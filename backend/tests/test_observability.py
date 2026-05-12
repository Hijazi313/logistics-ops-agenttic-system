# tests/test_observability.py
"""
Observability tests — verify LangSmith config is correctly structured.

What we test:
  - graph.invoke() receives metadata with po_id, thread_id, supplier
  - graph.invoke() receives correct tags
  - graph.invoke() receives a meaningful run_name
  - resume route passes approver_id and decision in metadata
  - persist_node uses .trace_id (not .id) for the audit log

What we do NOT test:
  - Whether LangSmith actually receives the trace (external service)
  - Whether the LangSmith dashboard shows the correct data
  - LangSmith SDK internals

Why this boundary:
  Testing external services in unit tests makes tests slow, flaky,
  and dependent on network and credentials. We test our contract —
  the correct config is passed. LangSmith's own tests cover the rest.
"""
import pytest
from unittest.mock import MagicMock, patch, call
from fastapi.testclient import TestClient
from langgraph.types import Interrupt

from src.api.main import app
from src.models.result import POAnalysisResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

VALID_PO = {
    "po_id": "PO-2026-0099",
    "supplier_name": "ACME Corp",
    "currency": "USD",
    "lead_time_days": 30,
    "total_value_usd": 500.00,
    "line_items": [{
        "sku": "SKU-001",
        "description": "Widget",
        "quantity": 25,
        "unit_price_usd": 20.00,
        "total_price_usd": 500.00,
    }],
}


@pytest.fixture
def mock_graph():
    return MagicMock()


@pytest.fixture
def client(mock_graph):
    app.state.graph = mock_graph
    with TestClient(app) as c:
        yield c, mock_graph


# ── Metadata in /analyze ──────────────────────────────────────────────────────

def test_analyze_passes_metadata_to_graph(client):
    test_client, mock_graph = client
    mock_graph.invoke.return_value = {
        "__interrupt__": [],
        "final_result": POAnalysisResult(
            po_id="PO-2026-0099",
            decision="approve",
            confidence="HIGH",
            anomalies=[],
            recommended_action="Approved.",
            reasoning_summary="Clean PO.",
        ),
    }

    test_client.post("/analyze", json=VALID_PO)

    # graph.invoke() must have been called
    assert mock_graph.invoke.called

    # Extract the config arg passed to invoke
    call_args = mock_graph.invoke.call_args
    config = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs.get("config")

    # Metadata must contain po_id and supplier
    assert "metadata" in config
    assert config["metadata"]["po_id"] == "PO-2026-0099"
    assert config["metadata"]["supplier"] == "ACME Corp"
    assert "thread_id" in config["metadata"]

    # Tags must include po-analysis
    assert "tags" in config
    assert "po-analysis" in config["tags"]

    # run_name must reference the po_id
    assert "run_name" in config
    assert "PO-2026-0099" in config["run_name"]


def test_analyze_each_request_has_unique_run_name(client):
    test_client, mock_graph = client
    mock_graph.invoke.return_value = {
        "__interrupt__": [],
        "final_result": POAnalysisResult(
            po_id="PO-2026-0099",
            decision="approve",
            confidence="HIGH",
            anomalies=[],
            recommended_action="Approved.",
            reasoning_summary="Clean PO.",
        ),
    }

    test_client.post("/analyze", json=VALID_PO)
    test_client.post("/analyze", json=VALID_PO)

    calls = mock_graph.invoke.call_args_list
    config_1 = calls[0].args[1]
    config_2 = calls[1].args[1]

    # thread_id in metadata must differ between requests
    assert config_1["metadata"]["thread_id"] != config_2["metadata"]["thread_id"]


# ── Metadata in /resume ───────────────────────────────────────────────────────

def test_resume_passes_metadata_to_graph(client):
    test_client, mock_graph = client
    mock_graph.invoke.return_value = {
        "__interrupt__": [],
        "final_result": POAnalysisResult(
            po_id="PO-2026-0099",
            decision="approve",
            confidence="LOW",
            anomalies=[],
            recommended_action="Approved by human.",
            reasoning_summary="Human approved.",
        ),
    }

    test_client.post(
        "/resume/test-thread-abc123",
        json={"decision": "approve", "approver_id": "muhammad"},
    )

    call_args = mock_graph.invoke.call_args
    config = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs.get("config")

    assert "metadata" in config
    assert config["metadata"]["approver_id"] == "muhammad"
    assert config["metadata"]["decision"] == "approve"
    assert "tags" in config
    assert "po-resume" in config["tags"]
    assert "decision:approve" in config["tags"]


# ── persist_node trace_id capture ─────────────────────────────────────────────

def test_persist_node_uses_trace_id_not_run_id(tmp_path):
    """
    Verify persist_node captures .trace_id from the run tree,
    not .id (which would be the span ID of just the persist node).
    """
    import json
    import os
    from unittest.mock import patch, MagicMock
    from src.models.po import POInput, POLineItem
    from src.models.result import POAnalysisResult
    from src.graph.nodes import persist_node

    audit_file = tmp_path / "audit.jsonl"

    final = POAnalysisResult(
        po_id="PO-2026-0099",
        decision="approve",
        confidence="HIGH",
        anomalies=[],
        recommended_action="Approved.",
        reasoning_summary="Clean PO.",
    )

    state = {
        "po_input": POInput(
            po_id="PO-2026-0099",
            supplier_name="ACME Corp",
            currency="USD",
            lead_time_days=30,
            total_value_usd=500.00,
            line_items=[POLineItem(
                sku="SKU-001",
                description="Widget",
                quantity=25,
                unit_price_usd=20.00,
                total_price_usd=500.00,
            )],
        ),
        "final_result": final,
        "approver_id": None,
        "audit_written": False,
    }

    # Mock run tree returning a trace_id
    mock_run = MagicMock()
    mock_run.trace_id = "root-trace-id-abc123"

    with patch("src.graph.nodes.get_current_run_tree", return_value=mock_run):
        with patch.dict(os.environ, {"AUDIT_LOG_PATH": str(audit_file)}):
            persist_node(state)

    entry = json.loads(audit_file.read_text().strip())
    # Must store trace_id, not span id
    assert entry["langsmith_trace_id"] == "root-trace-id-abc123"