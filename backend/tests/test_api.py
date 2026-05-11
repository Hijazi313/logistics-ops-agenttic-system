"""
API layer tests using FastAPI's TestClient.

What we test:
  - /health returns correct structure
  - /analyze with clean PO returns completed AnalyzeResponse
  - /analyze with anomalous PO returns InterruptResponse
  - /resume with valid thread_id completes the graph
  - /resume with invalid decision returns 422
  - /resume with unknown thread_id returns 404

What we mock:
  - The compiled graph — we test the API contract, not graph execution.
    Graph execution is covered by test_graph_integration.py.
  - This keeps API tests fast, deterministic, and free of LLM calls.

Why TestClient and not AsyncClient:
  TestClient wraps the ASGI app synchronously — simpler for unit tests.
  AsyncClient (httpx) is needed for true async testing, which this layer
  doesn't require since we're mocking the graph anyway.
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from langgraph.types import Interrupt

from src.api.main import app
from src.models.result import POAnalysisResult, AnomalyDetail


# ── Shared mock helpers ───────────────────────────────────────────────────────

def _clean_graph_result() -> dict:
    """Graph result for a clean PO — no interrupt, completed."""
    return {
        "__interrupt__": [],
        "final_result": POAnalysisResult(
            po_id="PO-2026-0001",
            decision="approve",
            confidence="HIGH",
            anomalies=[],
            recommended_action="PO meets all compliance requirements.",
            reasoning_summary="No violations detected.",
        ),
    }


def _anomalous_graph_result() -> dict:
    """Graph result for an anomalous PO — interrupt present."""
    interrupt_payload = {
        "po_id": "PO-2026-0002",
        "supplier": "ShadyDeals Inc",
        "violation_count": 2,
        "llm_recommendation": "escalate",
        "confidence": "LOW",
        "reasoning": "Two HIGH severity violations detected.",
        "recommended_action": "Escalate to procurement manager.",
        "anomalies": [
            {
                "rule_id": "approved_suppliers",
                "description": "Supplier check",
                "expected": "Approved vendor list",
                "actual": "ShadyDeals Inc",
                "explanation": "Not approved.",
                "severity": "HIGH",
            }
        ],
        "instruction": "Respond with: {'decision': 'approve'|'reject', 'approver_id': 'your-id'}",
    }
    return {
        "__interrupt__": [Interrupt(value=interrupt_payload, resumable=True, ns=[], when="during")],
        "final_result": None,
    }


def _resumed_graph_result(approver_id: str) -> dict:
    """Graph result after human resume — completed."""
    return {
        "__interrupt__": [],
        "final_result": POAnalysisResult(
            po_id="PO-2026-0002",
            decision="approve",
            confidence="LOW",
            anomalies=[
                AnomalyDetail(
                    rule_id="approved_suppliers",
                    description="Supplier check",
                    expected="Approved vendor list",
                    actual="ShadyDeals Inc",
                    explanation="Not approved.",
                    severity="HIGH",
                )
            ],
            recommended_action=f"Human reviewer ({approver_id}) approved with documented exceptions.",
            reasoning_summary="Approved by human after review.",
        ),
    }


# ── Valid PO payload ──────────────────────────────────────────────────────────

VALID_PO_PAYLOAD = {
    "po_id": "PO-2026-0001",
    "supplier_name": "ACME Corp",
    "currency": "USD",
    "lead_time_days": 30,
    "total_value_usd": 1000.00,
    "line_items": [
        {
            "sku": "SKU-001",
            "description": "Widget A",
            "quantity": 50,
            "unit_price_usd": 20.00,
            "total_price_usd": 1000.00,
        }
    ],
}


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_graph():
    """Mock compiled graph — injected into app.state for all API tests."""
    return MagicMock()


@pytest.fixture
def client(mock_graph):
    """
    TestClient with mock graph injected via dependency_overrides.
    This is the standard FastAPI way to mock dependencies for testing.
    It bypasses the GraphProvider used by the actual routes.
    """
    from src.api.dependencies import get_graph
    app.dependency_overrides[get_graph] = lambda: mock_graph
    
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    
    # Cleanup overrides after test
    app.dependency_overrides.clear()


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["graph_ready"] is True
    assert body["version"] == "1.0.0"


def test_health_degraded_when_graph_missing():
    """Without a graph in the provider, health reports degraded."""
    from src.api.dependencies import GraphProvider
    
    # Ensure provider is empty for this test
    # We don't use dependency_overrides here because health check uses GraphProvider directly
    # OR we could update health check to use get_graph dependency.
    # Actually, health check is better as a dependency-less route that checks the singleton.
    with patch("src.api.dependencies.GraphProvider.get_graph", return_value=None):
        with TestClient(app) as c:
            response = c.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "degraded"


# ── POST /analyze — clean PO ──────────────────────────────────────────────────

def test_analyze_clean_po_returns_completed(client, mock_graph):
    mock_graph.invoke.return_value = _clean_graph_result()

    response = client.post("/analyze", json=VALID_PO_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert "thread_id" in body
    assert body["result"]["decision"] == "approve"
    assert body["result"]["confidence"] == "HIGH"


def test_analyze_returns_unique_thread_ids(client, mock_graph):
    """Each /analyze call must produce a unique thread_id."""
    mock_graph.invoke.return_value = _clean_graph_result()

    r1 = client.post("/analyze", json=VALID_PO_PAYLOAD)
    r2 = client.post("/analyze", json=VALID_PO_PAYLOAD)

    assert r1.json()["thread_id"] != r2.json()["thread_id"]


# ── POST /analyze — anomalous PO ──────────────────────────────────────────────

def test_analyze_anomalous_po_returns_interrupt(client, mock_graph):
    mock_graph.invoke.return_value = _anomalous_graph_result()

    response = client.post("/analyze", json=VALID_PO_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "awaiting_human_review"
    assert "thread_id" in body
    assert body["violation_count"] == 2
    assert body["llm_recommendation"] == "escalate"
    assert body["confidence"] == "LOW"
    assert isinstance(body["anomalies"], list)


# ── POST /analyze — validation ────────────────────────────────────────────────

def test_analyze_rejects_malformed_payload(client, mock_graph):
    """Missing required fields → 422 from Pydantic, before graph is called."""
    response = client.post("/analyze", json={"po_id": "bad"})
    assert response.status_code == 422
    mock_graph.invoke.assert_not_called()


def test_analyze_rejects_invalid_po_id_format(client, mock_graph):
    """PO ID not matching PO-YYYY-NNNN format → 422."""
    bad_payload = {**VALID_PO_PAYLOAD, "po_id": "INVALID-ID"}
    response = client.post("/analyze", json=bad_payload)
    assert response.status_code == 422
    mock_graph.invoke.assert_not_called()


# ── POST /resume/{thread_id} ──────────────────────────────────────────────────

def test_resume_approve_returns_completed(client, mock_graph):
    mock_graph.invoke.return_value = _resumed_graph_result("muhammad")

    response = client.post(
        "/resume/some-thread-id-123",
        json={"decision": "approve", "approver_id": "muhammad"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["thread_id"] == "some-thread-id-123"
    assert body["result"]["decision"] == "approve"
    assert "muhammad" in body["result"]["recommended_action"]


def test_resume_reject_returns_completed(client, mock_graph):
    mock_graph.invoke.return_value = _resumed_graph_result("admin")
    mock_graph.invoke.return_value["final_result"].decision = "reject"

    response = client.post(
        "/resume/some-thread-id-456",
        json={"decision": "reject", "approver_id": "admin"},
    )

    assert response.status_code == 200
    assert response.json()["result"]["decision"] == "reject"


def test_resume_invalid_decision_returns_422(client, mock_graph):
    """Decision must be approve or reject — anything else is 422."""
    response = client.post(
        "/resume/some-thread-id",
        json={"decision": "maybe", "approver_id": "muhammad"},
    )
    assert response.status_code == 422
    mock_graph.invoke.assert_not_called()


def test_resume_unknown_thread_returns_404(client, mock_graph):
    """Unknown thread_id → graph raises, API returns 404."""
    mock_graph.invoke.side_effect = Exception("Thread not found in checkpointer")

    response = client.post(
        "/resume/nonexistent-thread",
        json={"decision": "approve", "approver_id": "muhammad"},
    )
    assert response.status_code == 404