# eval/run_eval.py
"""
Evaluation runner for the PO anomaly detection agent.

Loads the committed dataset from eval/dataset/, runs each PO through
the compiled graph, compares agent decision against ground truth,
and prints a precision/recall report.

Design decisions:
    - Ground truth is computed from RulesEngine at dataset generation time
      (deterministic, not LLM-judged)
    - Anomalous POs auto-resume with the LLM's own recommendation
      (we evaluate agent judgment, not human judgment)
    - Human interrupt is bypassed cleanly via Command(resume=...) using
      the LLM's recommendation from the interrupt payload
    - No external eval framework dependency — pure Python, zero extra deps

Run: uv run python eval/run_eval.py
Output: printed report + eval/eval_report.json
"""
import json
import sys
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("AUDIT_LOG_PATH", "eval/eval_audit.jsonl")

from langgraph.types import Command
from src.models.po import POInput
from src.graph.graph import build_graph
from src.graph.initial_state import build_initial_state

DATASET_DIR = Path(__file__).parent / "dataset"
REPORT_PATH = Path(__file__).parent / "eval_report.json"

graph = build_graph(db_path="eval/eval_checkpoints.db")


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    po_id: str
    category: str                   # clean | anomalous | edge_case
    ground_truth: str               # approve | escalate | reject
    agent_decision: str             # approve | escalate | reject
    ground_truth_violations: list[str]
    agent_violations_detected: int
    correct: bool
    failure_reason: str = ""


@dataclass
class EvalReport:
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    total: int = 0
    correct: int = 0
    incorrect: int = 0

    # Per-category accuracy
    clean_total: int = 0
    clean_correct: int = 0          # True Negative Rate (did not flag clean POs)

    anomalous_total: int = 0
    anomalous_correct: int = 0      # Recall (did catch real violations)

    edge_case_total: int = 0
    edge_case_correct: int = 0      # Boundary handling accuracy

    # Decision-level metrics
    false_positives: int = 0        # flagged a clean PO as anomalous
    false_negatives: int = 0        # missed a real violation

    results: list[dict] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)

    @property
    def overall_accuracy(self) -> float:
        return round(self.correct / self.total, 3) if self.total else 0.0

    @property
    def clean_precision(self) -> float:
        """
        Of POs the agent approved, what fraction were actually clean?
        False positives reduce this.
        """
        return round(self.clean_correct / self.clean_total, 3) if self.clean_total else 0.0

    @property
    def anomaly_recall(self) -> float:
        """
        Of real anomalous POs, what fraction did the agent catch?
        False negatives reduce this.
        """
        return round(self.anomalous_correct / self.anomalous_total, 3) \
            if self.anomalous_total else 0.0

    @property
    def edge_case_accuracy(self) -> float:
        return round(self.edge_case_correct / self.edge_case_total, 3) \
            if self.edge_case_total else 0.0


# ── Core runner ───────────────────────────────────────────────────────────────

def run_single_po(
    record: dict,
    category: str,
    thread_id: str,
) -> EvalResult:
    """
    Runs one PO through the graph end-to-end.

    For anomalous POs: auto-resumes the interrupt using the LLM's
    own recommendation from the interrupt payload. This evaluates
    whether the LLM correctly identifies and reasons about violations —
    the human layer is excluded from this measurement deliberately.

    Args:
        record:    A dict from the eval dataset (POInput fields + ground truth)
        category:  clean | anomalous | edge_case
        thread_id: Unique per-PO eval run ID

    Returns:
        EvalResult with comparison of agent vs ground truth
    """
    ground_truth = record["ground_truth_decision"]
    ground_truth_violations = record["ground_truth_rule_ids"]

    # Reconstruct POInput from the record (strip ground truth fields)
    po_fields = {
        k: v for k, v in record.items()
        if not k.startswith("ground_truth_")
    }
    po = POInput.model_validate(po_fields)

    config = {
        "configurable": {"thread_id": thread_id},
        "tags": ["eval"],
        "run_name": f"eval:{po.po_id}",
    }

    try:
        # First invoke
        result = graph.invoke(build_initial_state(po), config)

        # Check for interrupt (anomalous PO)
        interrupts = result.get("__interrupt__", [])
        if interrupts:
            payload = interrupts[0].value
            # Auto-resume with the LLM's recommendation
            # We evaluate agent judgment — bypass human decision
            llm_recommendation = payload.get("llm_recommendation", "reject")

            result = graph.invoke(
                Command(resume={
                    "decision": llm_recommendation,
                    "approver_id": "eval-runner",
                }),
                config,
            )

        final = result.get("final_result")
        if final is None:
            return EvalResult(
                po_id=po.po_id,
                category=category,
                ground_truth=ground_truth,
                agent_decision="error",
                ground_truth_violations=ground_truth_violations,
                agent_violations_detected=0,
                correct=False,
                failure_reason="Graph returned no final_result",
            )

        agent_decision = final.decision
        # Normalise: "escalate" and "reject" both mean "flagged as anomalous"
        # For precision/recall, we measure: did the agent correctly classify
        # clean vs anomalous? Escalate and reject are both "anomalous" flags.
        correct = _decisions_match(ground_truth, agent_decision)

        return EvalResult(
            po_id=po.po_id,
            category=category,
            ground_truth=ground_truth,
            agent_decision=agent_decision,
            ground_truth_violations=ground_truth_violations,
            agent_violations_detected=len(final.anomalies),
            correct=correct,
            failure_reason="" if correct else (
                f"Expected '{ground_truth}', got '{agent_decision}'"
            ),
        )

    except Exception as e:
        return EvalResult(
            po_id=po.po_id,
            category=category,
            ground_truth=ground_truth,
            agent_decision="error",
            ground_truth_violations=ground_truth_violations,
            agent_violations_detected=0,
            correct=False,
            failure_reason=f"Exception: {str(e)}",
        )


def _decisions_match(ground_truth: str, agent_decision: str) -> bool:
    """
    Matching logic: we collapse to binary correct/incorrect.

    Ground truth "approve" matches only agent "approve".
    Ground truth "escalate" or "reject" matches agent "escalate" or "reject".

    Rationale: the agent deciding escalate vs reject for a truly anomalous
    PO is a matter of severity judgment — both are correct in flagging it.
    A false negative (agent says approve for a bad PO) is the critical failure.
    """
    clean = {"approve"}
    flagged = {"escalate", "reject"}

    if ground_truth in clean:
        return agent_decision in clean
    if ground_truth in flagged:
        return agent_decision in flagged
    return False


# ── Report builder ────────────────────────────────────────────────────────────

def run_eval() -> EvalReport:
    report = EvalReport()

    categories = [
        ("clean_pos.json", "clean"),
        ("anomalous_pos.json", "anomalous"),
        ("edge_cases.json", "edge_case"),
    ]

    for filename, category in categories:
        path = DATASET_DIR / filename
        if not path.exists():
            print(f"  WARNING: {path} not found. Run generate_pos.py first.")
            continue

        records = json.loads(path.read_text())

        for i, record in enumerate(records):
            thread_id = f"eval-{category}-{i:03d}-{record['po_id']}"
            result = run_single_po(record, category, thread_id)

            # Update report counters
            report.total += 1
            if result.correct:
                report.correct += 1
            else:
                report.incorrect += 1
                report.failures.append(asdict(result))

            if category == "clean":
                report.clean_total += 1
                if result.correct:
                    report.clean_correct += 1
                elif result.agent_decision != "approve":
                    report.false_positives += 1

            elif category == "anomalous":
                report.anomalous_total += 1
                if result.correct:
                    report.anomalous_correct += 1
                elif result.agent_decision == "approve":
                    report.false_negatives += 1

            elif category == "edge_case":
                report.edge_case_total += 1
                if result.correct:
                    report.edge_case_correct += 1

            report.results.append(asdict(result))

            # Live progress
            status = "✓" if result.correct else "✗"
            print(
                f"  {status} [{category:10s}] {result.po_id:16s} "
                f"truth={result.ground_truth:8s} agent={result.agent_decision:8s} "
                + (f"← {result.failure_reason}" if not result.correct else "")
            )

    return report


# ── Report printer ────────────────────────────────────────────────────────────

def print_report(report: EvalReport) -> None:
    print("\n" + "═" * 65)
    print("  PO ANOMALY AGENT — EVALUATION REPORT")
    print("═" * 65)
    print(f"  Timestamp:          {report.timestamp}")
    print(f"  Total POs evaluated: {report.total}")
    print()
    print(f"  Overall Accuracy:   {report.overall_accuracy:.1%}  ({report.correct}/{report.total})")
    print()
    print("  ── By Category ─────────────────────────────────────────")
    print(f"  Clean (FP Rate):    {report.clean_precision:.1%}  ({report.clean_correct}/{report.clean_total}) [lower FP = better]")
    print(f"  Anomaly Recall:     {report.anomaly_recall:.1%}  ({report.anomalous_correct}/{report.anomalous_total}) [higher recall = better]")
    print(f"  Edge Case Accuracy: {report.edge_case_accuracy:.1%}  ({report.edge_case_correct}/{report.edge_case_total})")
    print()
    print("  ── Decision Quality ────────────────────────────────────")
    print(f"  False Positives:    {report.false_positives}  (clean POs incorrectly flagged)")
    print(f"  False Negatives:    {report.false_negatives}  (violations missed — critical)")
    print()

    if report.failures:
        print("  ── Failures ────────────────────────────────────────────")
        for f in report.failures:
            print(f"  ✗ {f['po_id']} [{f['category']}]")
            print(f"    Ground truth: {f['ground_truth']}  |  Agent: {f['agent_decision']}")
            print(f"    Reason: {f['failure_reason']}")
            print(f"    Violations expected: {f['ground_truth_violations']}")
        print()

    print("  ── Assessment ──────────────────────────────────────────")
    if report.false_negatives == 0 and report.overall_accuracy >= 0.8:
        print("  PASS — No missed violations. Overall accuracy acceptable.")
    elif report.false_negatives > 0:
        print(f"  WARN — {report.false_negatives} violation(s) missed (false negatives).")
        print("         Review anomaly_detect_node prompt and rules coverage.")
    else:
        print(f"  FAIL — Overall accuracy {report.overall_accuracy:.1%} below 80% threshold.")
        print("         Review rules engine and LLM reasoning prompt.")

    print("═" * 65)


if __name__ == "__main__":
    print("\nRunning evaluation suite...\n")
    report = run_eval()
    print_report(report)

    # Write JSON report for README and CI integration
    REPORT_PATH.write_text(
        json.dumps(
            {
                "timestamp": report.timestamp,
                "overall_accuracy": report.overall_accuracy,
                "clean_precision": report.clean_precision,
                "anomaly_recall": report.anomaly_recall,
                "edge_case_accuracy": report.edge_case_accuracy,
                "false_positives": report.false_positives,
                "false_negatives": report.false_negatives,
                "total": report.total,
                "correct": report.correct,
                "failures": report.failures,
            },
            indent=2
        )
    )
    print(f"\n  Report written to {REPORT_PATH}")