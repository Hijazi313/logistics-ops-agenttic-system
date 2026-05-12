"""
Manual smoke test — run this directly to see the graph in action.
Not a pytest test — this is for human verification during development.

Run: uv run python scripts/smoke_test.py
"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from langgraph.types import Command

load_dotenv()

def print_trace_url(thread_id: str):
    """
    Prints the LangSmith trace URL for a given thread.
    Only works when LANGSMITH_TRACING=true and LANGSMITH_PROJECT is set.
    """
    project = os.getenv("LANGSMITH_PROJECT", "default")
    print(f"\n  LangSmith: https://smith.langchain.com/projects/{project}")
    print(f"  Filter by tag: thread_id:{thread_id}")

from src.models.po import POInput, POLineItem
from src.graph.graph import build_graph
from src.graph.initial_state import build_initial_state

os.environ.setdefault("AUDIT_LOG_PATH", "smoke_audit.jsonl")

graph = build_graph(db_path="smoke_checkpoints.db")


def separator(label: str):
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print('─' * 60)


# ── Smoke Test A: Clean PO ────────────────────────────────────────────────────
separator("SMOKE TEST A: Clean PO — expect auto-approve")

clean_po = POInput(
    po_id="PO-2026-9001",
    supplier_name="ACME Corp",
    currency="USD",
    lead_time_days=30,
    total_value_usd=500.00,
    line_items=[
        POLineItem(
            sku="SKU-A1",
            description="Standard Widget",
            quantity=25,
            unit_price_usd=20.00,
            total_price_usd=500.00,
        )
    ],
)

config_a = {"configurable": {"thread_id": "smoke-clean-001"}}
result_a = graph.invoke(build_initial_state(clean_po), config_a)

print(f"Decision:    {result_a['final_result'].decision}")
print(f"Confidence:  {result_a['final_result'].confidence}")
print(f"Anomalies:   {len(result_a['final_result'].anomalies)}")
print(f"Action:      {result_a['final_result'].recommended_action}")
print(f"Interrupted: {'__interrupt__' in result_a and bool(result_a['__interrupt__'])}")


# ── Smoke Test B: Anomalous PO — interrupt/resume ─────────────────────────────
separator("SMOKE TEST B: Anomalous PO — expect interrupt then resume")

anomalous_po = POInput(
    po_id="PO-2026-9002",
    supplier_name="ShadyDeals Inc",
    currency="USD",
    lead_time_days=120,
    total_value_usd=3500.00,
    line_items=[
        POLineItem(
            sku="SKU-B1",
            description="Overpriced Part",
            quantity=100,
            unit_price_usd=35.00,
            total_price_usd=3500.00,
        )
    ],
)

config_b = {"configurable": {"thread_id": "smoke-anomalous-001"}}
result_b = graph.invoke(build_initial_state(anomalous_po), config_b)

if "__interrupt__" in result_b and result_b["__interrupt__"]:
    payload = result_b["__interrupt__"][0].value
    print(f"Graph paused. Interrupt payload:")
    print(json.dumps(payload, indent=2, default=str))

    print("\nResuming with human decision: reject")
    final_b = graph.invoke(
        Command(resume={"decision": "reject", "approver_id": "smoke-tester"}),
        config_b,
    )
    print(f"Decision:   {final_b['final_result'].decision}")
    print(f"Human:      {final_b['final_result'].recommended_action}")
else:
    print("ERROR: Expected interrupt but graph completed without pausing")

separator("SMOKE TESTS COMPLETE — check smoke_audit.jsonl for audit entries")

