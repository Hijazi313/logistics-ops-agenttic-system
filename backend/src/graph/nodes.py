"""
All LangGraph node functions for the PO anomaly detection graph.

Contract every node must follow:
  - Input:  POAgentState (full state, read-only by convention)
  - Output: dict with ONLY the keys this node modifies
  - No node modifies state in-place — always return a delta dict
  - Only persist_node has side effects (file I/O)
  - All other nodes are pure functions — same input → same output
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage

from src.models.po import POInput
from src.models.result import AnomalyDetail, POAnalysisResult, AuditLog
from src.rules.engine import RulesEngine
from src.llm.providers import get_reasoning_llm
from src.graph.state import POAgentState

from langgraph.types import interrupt

# Module-level engine instance — loaded once, reused across all invocations
# Rules YAML is read once at startup, not on every PO
_engine = RulesEngine()


# ── Node 1: ingest ────────────────────────────────────────────────────────────

def ingest_node(state: POAgentState) -> dict:
    """
    Entry point of the graph. Validates the raw PO input.

    Why is this a separate node and not just API-layer validation?
    Because the graph owns its own input validation. If the graph is
    invoked directly (CLI, tests, other agents), it must still defend
    its own boundary. Never assume the caller validated for you.

    What it does:
    - Confirms po_input is a valid POInput (Pydantic already enforced
      this at API boundary, but we surface errors into graph state here)
    - Initializes list fields to empty lists (safe defaults for downstream)
    - Populates parse_errors if something is structurally wrong

    Returns delta: parse_errors only (po_input is already in state)
    """
    errors: list[str] = []

    po = state.get("po_input")

    if po is None:
        errors.append("po_input is missing from state — cannot proceed.")
        return {"parse_errors": errors}

    if not isinstance(po, POInput):
        errors.append(
            f"po_input must be a POInput model, got {type(po).__name__}. "
            "Ensure the caller passes a validated POInput instance."
        )
        return {"parse_errors": errors}

    if not po.line_items:
        errors.append(f"PO {po.po_id} has no line items.")

    return {"parse_errors": errors}


# ── Node 2: validate ──────────────────────────────────────────────────────────

def validate_node(state: POAgentState) -> dict:
    """
    Deterministic rules validation. No LLM involved.

    This is the most important node for correctness.
    The rules engine is the ground truth — the LLM in anomaly_detect
    enriches and explains violations, but cannot override them.

    Design decision: all violations are collected before returning.
    We do NOT short-circuit on the first violation. A compliance officer
    needs to see the full picture, not one issue at a time.

    Returns delta: rule_violations
    """
    po = state["po_input"]
    violations = _engine.validate(po)
    return {"rule_violations": violations}


# ── Node 3: anomaly_detect ────────────────────────────────────────────────────

def anomaly_detect_node(state: POAgentState) -> dict:
    """
    LLM reasoning layer — enriches the deterministic violations with
    business context, explanation depth, and a recommended action.

    Why LLM here and not in validate_node?
    The rules engine tells you WHAT is wrong (price exceeds $25 limit).
    The LLM tells you WHY it matters and WHAT to do (supplier X has a
    history of inflated pricing; recommend renegotiation or alternate vendor).
    These are different jobs. Keep them in different nodes.

    Model used: reasoning LLM (gpt-4o / claude-sonnet) — this node
    justifies the cost because it produces the human-facing explanation.

    Uses with_structured_output to guarantee Pydantic model output.
    temperature=0 (set in provider factory) for deterministic decisions.

    Returns delta: anomaly_analysis, awaiting_human
    """
    llm = get_reasoning_llm()
    structured_llm = llm.with_structured_output(POAnalysisResult)

    po = state["po_input"]
    violations = state["rule_violations"]

    # Serialize violations for the prompt — structured, not freeform
    violations_text = json.dumps(
        [v.model_dump() for v in violations],
        indent=2
    )

    system_prompt = """You are a procurement compliance analyst.
You will receive a Purchase Order and a list of rule violations that were
detected by a deterministic rules engine.

Your job is to:
1. Enrich each violation with a business-context explanation
2. Assess the overall risk and recommend a decision
3. Write a concise reasoning summary for the human reviewer

Rules:
- You cannot override the violations list — they are factual rule breaches
- Your decision must be 'escalate' when violations exist (never 'approve')
- 'reject' is appropriate for HIGH severity violations with no justification
- Keep reasoning_summary under 100 words — reviewers are busy
- recommended_action must be one clear sentence
"""

    human_prompt = f"""Purchase Order ID: {po.po_id}
Supplier: {po.supplier_name}
Total Value: ${po.total_value_usd:,.2f} USD
Lead Time: {po.lead_time_days} days
Line Items: {len(po.line_items)}

Rule Violations Detected:
{violations_text}

Analyze these violations and return a structured POAnalysisResult.
The po_id must be: {po.po_id}
"""

    result: POAnalysisResult = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ])

    # Compute confidence deterministically — don't trust LLM self-rating
    result.confidence = POAnalysisResult.compute_confidence(violations)

    return {
        "anomaly_analysis": result,
        "awaiting_human": True,
    }


# ── Node 4: human_review ──────────────────────────────────────────────────────

def human_review_node(state: POAgentState) -> dict:
    """
    Human-in-the-loop interrupt node.

    How interrupt() works:
    1. Graph reaches this node and calls interrupt(payload)
    2. LangGraph checkpoints full state to SqliteSaver
    3. Graph execution PAUSES — invoke() returns to the caller
    4. Caller reads the interrupt payload from result["__interrupt__"]
    5. Caller resumes with: graph.invoke(Command(resume={...}), config)
    6. THIS NODE RESTARTS from the top (LangGraph re-runs the node)
    7. interrupt() now returns the resume value instead of pausing
    8. Execution continues past interrupt() with human_decision in hand

    CRITICAL — why the node re-runs from top (Step 6):
    LangGraph replays the node on resume. This means any code BEFORE
    interrupt() runs twice. Keep pre-interrupt logic minimal and idempotent.
    We have none here — the interrupt is the first and only thing this node does.

    The payload surfaced to the human reviewer contains everything they need
    to make a decision without looking anywhere else.

    Returns delta: human_decision, approver_id, awaiting_human
    """
    analysis = state["anomaly_analysis"]
    po = state["po_input"]

    # Build the interrupt payload — what the human reviewer sees
    interrupt_payload = {
        "po_id": po.po_id,
        "supplier": po.supplier_name,
        "total_value_usd": po.total_value_usd,
        "violation_count": len(state["rule_violations"]),
        "llm_recommendation": analysis.decision,
        "confidence": analysis.confidence,
        "reasoning": analysis.reasoning_summary,
        "recommended_action": analysis.recommended_action,
        "anomalies": [v.model_dump() for v in analysis.anomalies],
        "instruction": "Respond with: {'decision': 'approve'|'reject', 'approver_id': 'your-id'}",
    }

    # This call pauses the graph on first run, returns resume value on second run
    human_response: dict = interrupt(interrupt_payload)

    decision = human_response.get("decision", "reject")
    approver = human_response.get("approver_id", "unknown")

    # Validate the human response — don't trust raw input
    if decision not in ("approve", "reject"):
        decision = "reject"  # safe default — never auto-approve on bad input

    return {
        "human_decision": decision,
        "approver_id": approver,
        "awaiting_human": False,
    }


# ── Node 5: resolve ───────────────────────────────────────────────────────────

def resolve_node(state: POAgentState) -> dict:
    """
    Assembles the final POAnalysisResult.

    Two paths converge here:
    Path A — Clean PO (no violations):
      rule_violations is empty, anomaly_analysis is None,
      human_decision is None → decision = "approve"

    Path B — Anomalous PO after human review:
      anomaly_analysis has the LLM result, human_decision is set
      Human can override to "approve" or confirm "reject"

    The final confidence is always computed deterministically here —
    never taken from LLM output or human input.

    Returns delta: final_result
    """
    po = state["po_input"]
    violations = state["rule_violations"]
    human_decision = state.get("human_decision")
    analysis = state.get("anomaly_analysis")

    # Path A: clean PO — no violations, no human review needed
    if not violations:
        final = POAnalysisResult(
            po_id=po.po_id,
            decision="approve",
            confidence="HIGH",
            anomalies=[],
            recommended_action="PO meets all compliance requirements. Approved for processing.",
            reasoning_summary="No rule violations detected. Deterministic validation passed.",
        )
        return {"final_result": final}

    # Path B: anomalous PO — use human decision, enrich from LLM analysis
    # Human can approve even anomalous POs (e.g., pre-authorised exceptions)
    final_decision = human_decision or "reject"

    # Map human decision to our three-way decision type
    # "approve" from human on an anomalous PO = override with documented anomalies
    # "reject" from human = reject
    # LLM may have said "escalate" — we resolve that to a final decision here
    if final_decision == "approve":
        decision_value = "approve"
        action = f"Human reviewer ({state.get('approver_id')}) approved with documented exceptions."
    else:
        decision_value = "reject"
        action = f"Human reviewer ({state.get('approver_id')}) rejected. Do not process this PO."

    final = POAnalysisResult(
        po_id=po.po_id,
        decision=decision_value,
        confidence=POAnalysisResult.compute_confidence(violations),
        anomalies=analysis.anomalies if analysis else [],
        recommended_action=action,
        reasoning_summary=analysis.reasoning_summary if analysis else "Manual review completed.",
    )

    return {"final_result": final}


# ── Node 6: persist ───────────────────────────────────────────────────────────

def persist_node(state: POAgentState) -> dict:
    """
    Writes the immutable audit log entry. The only node with side effects.

    Why append-only JSONL and not a database?
    For a portfolio project, JSONL is the right choice:
    - Zero infrastructure dependency
    - Human-readable
    - Trivially importable into any database later
    - Immutable by convention (append mode, never update)

    In production: swap to PostgreSQL INSERT with no DELETE privilege
    on the audit table — same immutability guarantee, better queryability.

    LangSmith trace ID: captured from the active run context.
    If tracing is disabled, trace_id is None — not an error.

    Guards against double-write: checks audit_written flag.

    Returns delta: audit_written
    """
    # Guard: never write twice for the same state
    if state.get("audit_written"):
        return {"audit_written": True}

    final = state["final_result"]
    if final is None:
        # Should never happen if graph is wired correctly
        # Defensive guard — log and continue rather than crash
        return {"audit_written": False}

    # Attempt to capture LangSmith trace ID — graceful fallback if unavailable
    trace_id: str | None = None
    try:
        from langsmith import get_current_run_tree
        run = get_current_run_tree()
        trace_id = str(run.id) if run else None
    except Exception:
        pass  # Tracing is optional — never fail the graph over observability

    audit = AuditLog(
        po_id=final.po_id,
        decision=final.decision,
        confidence=final.confidence,
        anomaly_count=len(final.anomalies),
        timestamp=datetime.now(timezone.utc).isoformat(),
        human_actor=state.get("approver_id"),
        langsmith_trace_id=trace_id,
        reasoning_snapshot=final.reasoning_summary,
    )

    # Append to JSONL — one JSON object per line
    audit_path = Path(os.getenv("AUDIT_LOG_PATH", "audit_log.jsonl"))
    with open(audit_path, "a", encoding="utf-8") as f:
        f.write(audit.model_dump_json() + "\n")

    return {"audit_written": True}


# ── Node 7: fail_fast ─────────────────────────────────────────────────────────

def fail_fast_node(state: POAgentState) -> dict:
    """
    Terminal error node for unrecoverable parse failures.

    Why a dedicated node instead of raising an exception?
    Exceptions in LangGraph nodes crash the graph and lose state.
    A fail_fast node keeps the graph in a known terminal state,
    writes a minimal audit entry, and returns cleanly.
    The caller gets a structured response, not a 500 error.

    Returns delta: final_result, audit_written
    """
    errors = state.get("parse_errors", [])
    po = state.get("po_input")
    po_id = po.po_id if po else "UNKNOWN"

    error_summary = "; ".join(errors) if errors else "Unknown parse error"

    final = POAnalysisResult(
        po_id=po_id,
        decision="reject",
        confidence="LOW",
        anomalies=[],
        recommended_action="PO rejected due to structural errors. Correct and resubmit.",
        reasoning_summary=f"Parse failure: {error_summary}",
    )

    # Write a minimal audit entry for failed POs too — full audit trail
    audit = AuditLog(
        po_id=po_id,
        decision="reject",
        confidence="LOW",
        anomaly_count=0,
        timestamp=datetime.now(timezone.utc).isoformat(),
        human_actor=None,
        langsmith_trace_id=None,
        reasoning_snapshot=f"PARSE_FAILURE: {error_summary}",
    )

    audit_path = Path(os.getenv("AUDIT_LOG_PATH", "audit_log.jsonl"))
    with open(audit_path, "a", encoding="utf-8") as f:
        f.write(audit.model_dump_json() + "\n")

    return {
        "final_result": final,
        "audit_written": True,
    }