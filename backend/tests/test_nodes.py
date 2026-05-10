# tests/test_nodes.py
"""
Unit tests for graph nodes.

Strategy:
- Nodes 1, 2, 5, 6, 7: pure logic — test directly with mock state
- Node 3 (anomaly_detect): mock the LLM — we test node logic, not the model
- Node 4 (human_review): not unit-testable in isolation — interrupt()
  requires a running checkpointed graph. Covered in integration tests (Day 4).
- Edge functions: tested here since they're pure routing logic
"""
import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.models.po import POInput, POLineItem
from src.models.result import AnomalyDetail, POAnalysisResult
from src.graph.nodes import (
    ingest_node,
    validate_node,
    anomaly_detect_node,
    resolve_node,
    persist_node,
    fail_fast_node,
)
from src.graph.edges import route_after_ingest, route_after_validate


# ── Shared Fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def clean_po():
    return POInput(
        po_id="PO-2026-0001",
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
        po_id="PO-2026-0002",
        supplier_name="ShadyDeals Inc",   # unapproved
        currency="USD",
        lead_time_days=30,
        total_value_usd=3000.00,
        line_items=[
            POLineItem(
                sku="SKU-999",
                description="Overpriced Widget",
                quantity=100,
                unit_price_usd=30.00,    # exceeds $25 limit
                total_price_usd=3000.00,
            )
        ],
    )


def _base_state(po: POInput) -> dict:
    """Minimal valid state for testing individual nodes."""
    return {
        "po_input": po,
        "parse_errors": [],
        "rule_violations": [],
        "anomaly_analysis": None,
        "awaiting_human": False,
        "human_decision": None,
        "approver_id": None,
        "final_result": None,
        "audit_written": False,
    }


# ── ingest_node ───────────────────────────────────────────────────────────────

def test_ingest_clean_po_no_errors(clean_po):
    state = _base_state(clean_po)
    result = ingest_node(state)
    assert result["parse_errors"] == []


def test_ingest_missing_po_input():
    state = _base_state(None)
    state["po_input"] = None
    result = ingest_node(state)
    assert len(result["parse_errors"]) > 0
    assert "missing" in result["parse_errors"][0].lower()


# ── validate_node ─────────────────────────────────────────────────────────────

def test_validate_clean_po_no_violations(clean_po):
    state = _base_state(clean_po)
    result = validate_node(state)
    assert result["rule_violations"] == []


def test_validate_anomalous_po_has_violations(anomalous_po):
    state = _base_state(anomalous_po)
    result = validate_node(state)
    violations = result["rule_violations"]
    assert len(violations) >= 1
    rule_ids = [v.rule_id for v in violations]
    assert "approved_suppliers" in rule_ids
    assert "price_per_unit" in rule_ids


# ── anomaly_detect_node ───────────────────────────────────────────────────────

def test_anomaly_detect_calls_reasoning_llm(anomalous_po):
    """
    We mock the LLM — testing node logic, not the model.
    The mock returns a valid POAnalysisResult so the node can proceed.
    """
    mock_result = POAnalysisResult(
        po_id="PO-2026-0002",
        decision="escalate",
        confidence="LOW",
        anomalies=[
            AnomalyDetail(
                rule_id="approved_suppliers",
                description="Supplier not approved",
                expected="Approved vendor list",
                actual="ShadyDeals Inc",
                explanation="Unapproved supplier poses procurement risk.",
                severity="HIGH",
            )
        ],
        recommended_action="Escalate to procurement manager.",
        reasoning_summary="Two violations detected. HIGH severity supplier issue.",
    )

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.return_value = mock_result

    state = _base_state(anomalous_po)
    state["rule_violations"] = [
        AnomalyDetail(
            rule_id="approved_suppliers",
            description="Supplier check",
            expected="Approved list",
            actual="ShadyDeals Inc",
            explanation="Not approved.",
            severity="HIGH",
        )
    ]

    with patch("src.graph.nodes.get_reasoning_llm", return_value=mock_llm):
        result = anomaly_detect_node(state)

    assert result["anomaly_analysis"] is not None
    assert result["awaiting_human"] is True
    assert result["anomaly_analysis"].decision == "escalate"
    # Confidence must be deterministically computed, not from mock
    assert result["anomaly_analysis"].confidence == "LOW"


# ── resolve_node ──────────────────────────────────────────────────────────────

def test_resolve_clean_po_auto_approves(clean_po):
    """Clean PO with no violations resolves to approve without human."""
    state = _base_state(clean_po)
    state["rule_violations"] = []
    state["anomaly_analysis"] = None
    state["human_decision"] = None

    result = resolve_node(state)
    assert result["final_result"].decision == "approve"
    assert result["final_result"].confidence == "HIGH"
    assert result["final_result"].anomalies == []


def test_resolve_human_approved_anomalous_po(anomalous_po):
    """Human approves anomalous PO — exception override."""
    violations = [
        AnomalyDetail(
            rule_id="price_per_unit",
            description="Price check",
            expected="<= $25.00",
            actual="$30.00",
            explanation="Overpriced.",
            severity="HIGH",
        )
    ]
    analysis = POAnalysisResult(
        po_id="PO-2026-0002",
        decision="escalate",
        confidence="LOW",
        anomalies=violations,
        recommended_action="Escalate.",
        reasoning_summary="Price violation detected.",
    )

    state = _base_state(anomalous_po)
    state["rule_violations"] = violations
    state["anomaly_analysis"] = analysis
    state["human_decision"] = "approve"
    state["approver_id"] = "muhammad"

    result = resolve_node(state)
    assert result["final_result"].decision == "approve"
    assert "muhammad" in result["final_result"].recommended_action


def test_resolve_human_rejected_anomalous_po(anomalous_po):
    """Human rejects anomalous PO."""
    violations = [
        AnomalyDetail(
            rule_id="approved_suppliers",
            description="Supplier check",
            expected="Approved list",
            actual="ShadyDeals Inc",
            explanation="Not approved.",
            severity="HIGH",
        )
    ]
    analysis = POAnalysisResult(
        po_id="PO-2026-0002",
        decision="escalate",
        confidence="LOW",
        anomalies=violations,
        recommended_action="Reject.",
        reasoning_summary="Unapproved supplier.",
    )

    state = _base_state(anomalous_po)
    state["rule_violations"] = violations
    state["anomaly_analysis"] = analysis
    state["human_decision"] = "reject"
    state["approver_id"] = "muhammad"

    result = resolve_node(state)
    assert result["final_result"].decision == "reject"


# ── persist_node ──────────────────────────────────────────────────────────────

def test_persist_writes_audit_log(clean_po, tmp_path):
    """Verify audit log is written as valid JSONL."""
    audit_file = tmp_path / "test_audit.jsonl"

    final = POAnalysisResult(
        po_id="PO-2026-0001",
        decision="approve",
        confidence="HIGH",
        anomalies=[],
        recommended_action="Approved.",
        reasoning_summary="Clean PO.",
    )

    state = _base_state(clean_po)
    state["final_result"] = final

    with patch.dict(os.environ, {"AUDIT_LOG_PATH": str(audit_file)}):
        result = persist_node(state)

    assert result["audit_written"] is True
    assert audit_file.exists()

    lines = audit_file.read_text().strip().splitlines()
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert entry["po_id"] == "PO-2026-0001"
    assert entry["decision"] == "approve"


def test_persist_does_not_double_write(clean_po, tmp_path):
    """audit_written=True guard prevents writing twice."""
    audit_file = tmp_path / "test_audit.jsonl"

    final = POAnalysisResult(
        po_id="PO-2026-0001",
        decision="approve",
        confidence="HIGH",
        anomalies=[],
        recommended_action="Approved.",
        reasoning_summary="Clean PO.",
    )

    state = _base_state(clean_po)
    state["final_result"] = final
    state["audit_written"] = True   # already written

    with patch.dict(os.environ, {"AUDIT_LOG_PATH": str(audit_file)}):
        result = persist_node(state)

    # File should not exist — guard prevented write
    assert not audit_file.exists()
    assert result["audit_written"] is True


# ── fail_fast_node ────────────────────────────────────────────────────────────

def test_fail_fast_produces_reject_result(tmp_path):
    audit_file = tmp_path / "test_audit.jsonl"
    state = {
        "po_input": None,
        "parse_errors": ["po_input is missing from state — cannot proceed."],
        "rule_violations": [],
        "anomaly_analysis": None,
        "awaiting_human": False,
        "human_decision": None,
        "approver_id": None,
        "final_result": None,
        "audit_written": False,
    }

    with patch.dict(os.environ, {"AUDIT_LOG_PATH": str(audit_file)}):
        result = fail_fast_node(state)

    assert result["final_result"].decision == "reject"
    assert result["audit_written"] is True
    assert audit_file.exists()


# ── Edge functions ────────────────────────────────────────────────────────────

def test_route_after_ingest_clean(clean_po):
    state = _base_state(clean_po)
    state["parse_errors"] = []
    assert route_after_ingest(state) == "validate"


def test_route_after_ingest_errors(clean_po):
    state = _base_state(clean_po)
    state["parse_errors"] = ["something went wrong"]
    assert route_after_ingest(state) == "fail_fast"


def test_route_after_validate_violations(anomalous_po):
    state = _base_state(anomalous_po)
    state["rule_violations"] = [
        AnomalyDetail(
            rule_id="currency",
            description="Currency check",
            expected="USD",
            actual="EUR",
            explanation="Wrong currency.",
            severity="HIGH",
        )
    ]
    assert route_after_validate(state) == "anomaly_detect"


def test_route_after_validate_clean(clean_po):
    state = _base_state(clean_po)
    state["rule_violations"] = []
    assert route_after_validate(state) == "resolve"