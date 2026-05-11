"""
Integration tests for the compiled graph — end-to-end execution.

What we test:
  Path A — Clean PO:
    ingest → validate → resolve → persist → END
    No LLM call, no interrupt, auto-approved, audit log written.

  Path B — Anomalous PO:
    ingest → validate → anomaly_detect → human_review (INTERRUPT)
    → resume → resolve → persist → END
    LLM mocked, interrupt payload verified, resume verified.

What we do NOT test here:
  - Individual node logic (test_nodes.py covers that)
  - Rules engine (test_rules_engine.py covers that)
  - LLM output quality (not unit-testable)

Why real SqliteSaver in tests and not MemorySaver:
  We want to test the actual checkpoint/resume cycle, which requires
  a real persisted state. MemorySaver would work functionally but
  SqliteSaver confirms the serialisation path is correct end-to-end.
  We use tmp_path (pytest fixture) so tests never touch the real DB.
"""
import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from langgraph.types import Command

from src.models.po import POInput, POLineItem
from src.models.result import AnomalyDetail, POAnalysisResult
from src.graph.graph import build_graph
from src.graph.initial_state import build_initial_state
from config import load_env

load_env()


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def graph(tmp_path):
    """
    Compiled graph with isolated SQLite DB per test.
    tmp_path is a pytest built-in — unique temp directory per test function.
    This ensures tests never share checkpoint state.
    """
    db_path = str(tmp_path / "test_checkpoints.db")
    return build_graph(db_path=db_path)


@pytest.fixture
def audit_log_path(tmp_path):
    """Isolated audit log path per test."""
    return str(tmp_path / "test_audit.jsonl")


@pytest.fixture
def clean_po():
    return POInput(
        po_id="PO-2026-0010",
        supplier_name="ACME Corp",
        currency="USD",
        lead_time_days=30,
        total_value_usd=1000.00,
        line_items=[
            POLineItem(
                sku="SKU-001",
                description="Widget A",
                quantity=50,
                unit_price_usd=20.00,
                total_price_usd=1000.00,
            )
        ],
    )


@pytest.fixture
def anomalous_po():
    return POInput(
        po_id="PO-2026-0011",
        supplier_name="ShadyDeals Inc",
        currency="USD",
        lead_time_days=30,
        total_value_usd=3000.00,
        line_items=[
            POLineItem(
                sku="SKU-999",
                description="Overpriced Widget",
                quantity=100,
                unit_price_usd=30.00,
                total_price_usd=3000.00,
            )
        ],
    )


def _mock_analysis(po_id: str, violations: list[AnomalyDetail]) -> POAnalysisResult:
    """Builds a realistic mock LLM result for anomaly_detect_node."""
    return POAnalysisResult(
        po_id=po_id,
        decision="escalate",
        confidence="LOW",
        anomalies=violations,
        recommended_action="Escalate to procurement manager for review.",
        reasoning_summary="Unapproved supplier and unit price violation detected.",
    )


# ── Path A: Clean PO ──────────────────────────────────────────────────────────

def test_clean_po_full_path(graph, clean_po, audit_log_path):
    """
    Full graph execution for a clean PO.

    Expected path: ingest → validate → resolve → persist → END
    Expected outcome:
      - No interrupt
      - decision = "approve"
      - confidence = "HIGH"
      - audit log written with correct fields
    """
    config = {"configurable": {"thread_id": "clean-po-001"}}
    initial_state = build_initial_state(clean_po)

    with patch.dict(os.environ, {"AUDIT_LOG_PATH": audit_log_path}):
        result = graph.invoke(initial_state, config)

    # No interrupt should occur for a clean PO
    assert "__interrupt__" not in result or result.get("__interrupt__") == []

    # Final result must be present and correct
    final = result["final_result"]
    assert final is not None
    assert final.decision == "approve"
    assert final.confidence == "HIGH"
    assert final.anomalies == []
    assert final.po_id == "PO-2026-0010"

    # Audit log must exist with one entry
    log_path = Path(audit_log_path)
    assert log_path.exists(), "Audit log was not written"

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert entry["po_id"] == "PO-2026-0010"
    assert entry["decision"] == "approve"
    assert entry["anomaly_count"] == 0
    assert entry["human_actor"] is None  # no human needed for clean PO


# ── Path B: Anomalous PO — interrupt then resume ──────────────────────────────

def test_anomalous_po_pauses_at_interrupt(graph, anomalous_po, audit_log_path):
    """
    Verify graph pauses at human_review for an anomalous PO.

    First invoke should:
      - Run: ingest → validate → anomaly_detect → human_review (PAUSE)
      - Return interrupt payload in result["__interrupt__"]
      - NOT write audit log yet (decision not made)
    """
    config = {"configurable": {"thread_id": "anomalous-po-001"}}
    initial_state = build_initial_state(anomalous_po)

    violations = [
        AnomalyDetail(
            rule_id="approved_suppliers",
            description="Supplier check",
            expected="Approved vendor list",
            actual="ShadyDeals Inc",
            explanation="Not on approved vendor list.",
            severity="HIGH",
        ),
        AnomalyDetail(
            rule_id="price_per_unit",
            description="Unit price check",
            expected="<= $25.00 USD/unit",
            actual="$30.00 USD/unit (SKU: SKU-999)",
            explanation="Price exceeds limit by $5.00/unit.",
            severity="HIGH",
        ),
    ]
    mock_result = _mock_analysis("PO-2026-0011", violations)
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.return_value = mock_result

    with patch("src.graph.nodes.get_reasoning_llm", return_value=mock_llm):
        with patch.dict(os.environ, {"AUDIT_LOG_PATH": audit_log_path}):
            result = graph.invoke(initial_state, config)

    # Graph must have paused — interrupt payload present
    assert "__interrupt__" in result
    interrupts = result["__interrupt__"]
    assert len(interrupts) > 0

    # Interrupt payload must contain what the human reviewer needs
    payload = interrupts[0].value
    assert payload["po_id"] == "PO-2026-0011"
    assert payload["violation_count"] == 2
    assert payload["llm_recommendation"] == "escalate"
    assert "instruction" in payload

    # Audit log must NOT exist yet — decision not finalised
    assert not Path(audit_log_path).exists(), \
        "Audit log should not be written before human decision"


def test_anomalous_po_resumes_with_approval(graph, anomalous_po, audit_log_path):
    """
    Full interrupt/resume cycle — human approves the anomalous PO.

    Two invokes:
      1. First invoke → pauses at interrupt
      2. Second invoke with Command(resume=...) → completes graph

    Same thread_id must be used for both — this is the resume contract.
    """
    config = {"configurable": {"thread_id": "anomalous-po-002"}}
    initial_state = build_initial_state(anomalous_po)

    violations = [
        AnomalyDetail(
            rule_id="approved_suppliers",
            description="Supplier check",
            expected="Approved vendor list",
            actual="ShadyDeals Inc",
            explanation="Not on approved vendor list.",
            severity="HIGH",
        )
    ]
    mock_result = _mock_analysis("PO-2026-0011", violations)
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.return_value = mock_result

    with patch("src.graph.nodes.get_reasoning_llm", return_value=mock_llm):
        with patch.dict(os.environ, {"AUDIT_LOG_PATH": audit_log_path}):

            # ── First invoke: reaches interrupt ───────────────────────────
            first_result = graph.invoke(initial_state, config)
            assert "__interrupt__" in first_result

            # ── Second invoke: resume with human approval ─────────────────
            resume_payload = {
                "decision": "approve",
                "approver_id": "muhammad"
            }
            final_result = graph.invoke(
                Command(resume=resume_payload),
                config,
            )

    # Graph must have completed — no more interrupts
    interrupts = final_result.get("__interrupt__", [])
    assert len(interrupts) == 0

    # Final decision reflects human approval
    final = final_result["final_result"]
    assert final is not None
    assert final.decision == "approve"
    assert "muhammad" in final.recommended_action

    # Audit log must now exist
    log_path = Path(audit_log_path)
    assert log_path.exists(), "Audit log must be written after resume"

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert entry["decision"] == "approve"
    assert entry["human_actor"] == "muhammad"
    assert entry["anomaly_count"] == 1


def test_anomalous_po_resumes_with_rejection(graph, anomalous_po, audit_log_path):
    """
    Full interrupt/resume cycle — human rejects the anomalous PO.
    """
    config = {"configurable": {"thread_id": "anomalous-po-003"}}
    initial_state = build_initial_state(anomalous_po)

    violations = [
        AnomalyDetail(
            rule_id="price_per_unit",
            description="Unit price check",
            expected="<= $25.00",
            actual="$30.00 (SKU: SKU-999)",
            explanation="Overpriced.",
            severity="HIGH",
        )
    ]
    mock_result = _mock_analysis("PO-2026-0011", violations)
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.return_value = mock_result

    with patch("src.graph.nodes.get_reasoning_llm", return_value=mock_llm):
        with patch.dict(os.environ, {"AUDIT_LOG_PATH": audit_log_path}):
            graph.invoke(initial_state, config)
            final_result = graph.invoke(
                Command(resume={"decision": "reject", "approver_id": "admin"}),
                config,
            )

    final = final_result["final_result"]
    assert final.decision == "reject"

    entry = json.loads(Path(audit_log_path).read_text().strip())
    assert entry["decision"] == "reject"
    assert entry["human_actor"] == "admin"


def test_malformed_po_routes_to_fail_fast(graph, audit_log_path):
    """
    A state with no po_input routes to fail_fast and writes an error audit log.
    Graph must not crash — it must return a structured reject result.
    """
    config = {"configurable": {"thread_id": "malformed-po-001"}}

    # Build a state with no po_input — simulates a corrupted API call
    from src.graph.state import POAgentState
    bad_state = POAgentState(
        po_input=None,
        parse_errors=[],
        rule_violations=[],
        anomaly_analysis=None,
        awaiting_human=False,
        human_decision=None,
        approver_id=None,
        final_result=None,
        audit_written=False,
    )

    with patch.dict(os.environ, {"AUDIT_LOG_PATH": audit_log_path}):
        result = graph.invoke(bad_state, config)

    final = result["final_result"]
    assert final.decision == "reject"
    assert "UNKNOWN" in final.po_id or final.po_id == "UNKNOWN"

    # Audit log still written — even failures are audited
    assert Path(audit_log_path).exists()