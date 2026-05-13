# eval/generate_pos.py
"""
One-time synthetic PO generator.

Run once to populate eval/dataset/.
Do NOT re-run after committing the dataset — changing the dataset
changes ground truth, which invalidates historical eval comparisons.

Uses the LLM to generate realistic PO data, but ground_truth_decision
is always computed deterministically from the RulesEngine — never
from the LLM's judgment.

Run: uv run python eval/generate_pos.py
"""
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.rules.engine import RulesEngine
from src.models.po import POInput, POLineItem

engine = RulesEngine()
DATASET_DIR = Path(__file__).parent / "dataset"
DATASET_DIR.mkdir(exist_ok=True)


def compute_ground_truth(po: POInput) -> dict:
    """
    Compute ground truth deterministically from RulesEngine.
    This is the entire point: ground truth is never guessed or LLM-judged.

    Returns:
        dict with ground_truth_decision and ground_truth_violations
    """
    violations = engine.validate(po)
    if not violations:
        decision = "approve"
    else:
        high_severity = any(v.severity == "HIGH" for v in violations)
        decision = "reject" if high_severity else "escalate"

    return {
        "ground_truth_decision": decision,
        "ground_truth_violation_count": len(violations),
        "ground_truth_rule_ids": [v.rule_id for v in violations],
    }


def make_po_record(po: POInput) -> dict:
    """Combines POInput dict with deterministic ground truth."""
    record = po.model_dump()
    record.update(compute_ground_truth(po))
    return record


# ── 5 Clean POs ───────────────────────────────────────────────────────────────
# All fields within rules. Ground truth: approve. Tests False Positive Rate.

clean_pos = [
    POInput(
        po_id="PO-2026-C001",
        supplier_name="ACME Corp",
        currency="USD",
        lead_time_days=30,
        total_value_usd=1000.00,
        line_items=[POLineItem(
            sku="SKU-C001", description="Standard Widget",
            quantity=50, unit_price_usd=20.00, total_price_usd=1000.00
        )],
    ),
    POInput(
        po_id="PO-2026-C002",
        supplier_name="GlobalParts Ltd",
        currency="USD",
        lead_time_days=45,
        total_value_usd=2400.00,
        line_items=[POLineItem(
            sku="SKU-C002", description="Industrial Fastener",
            quantity=800, unit_price_usd=3.00, total_price_usd=2400.00
        )],
    ),
    POInput(
        po_id="PO-2026-C003",
        supplier_name="FastShip Inc",
        currency="USD",
        lead_time_days=14,
        total_value_usd=12200.00,
        line_items=[
            POLineItem(
                sku="SKU-C003A", description="Precision Gear",
                quantity=200, unit_price_usd=24.99, total_price_usd=4998.00
            ),
            POLineItem(
                sku="SKU-C003B", description="Mounting Bracket",
                quantity=500, unit_price_usd=14.00, total_price_usd=7000.00
            ),
            POLineItem(
                sku="SKU-C003C", description="Hex Bolt Set",
                quantity=100, unit_price_usd=2.02, total_price_usd=202.00
            ),
        ],
        # total: 4998 + 7000 + 202 = 12200 — recalculate
    ),
    POInput(
        po_id="PO-2026-C004",
        supplier_name="ACME Corp",
        currency="USD",
        lead_time_days=60,
        total_value_usd=500.00,
        line_items=[POLineItem(
            sku="SKU-C004", description="Cable Assembly",
            quantity=20, unit_price_usd=25.00, total_price_usd=500.00
            # unit price exactly at limit — should NOT trigger violation
        )],
    ),
    POInput(
        po_id="PO-2026-C005",
        supplier_name="GlobalParts Ltd",
        currency="USD",
        lead_time_days=90,          # exactly at lead time limit
        total_value_usd=49940.01,   # corrected sum of lines
        line_items=[
            POLineItem(
                sku="SKU-C005A", description="Hydraulic Pump",
                quantity=999, unit_price_usd=24.99, total_price_usd=24965.01
            ),
            POLineItem(
                sku="SKU-C005B", description="Pressure Valve",
                quantity=999, unit_price_usd=25.00, total_price_usd=24975.00
            ),
        ],
    ),
]

# ── 5 Anomalous POs ───────────────────────────────────────────────────────────
# Clear violations. Ground truth: escalate or reject. Tests Recall.

anomalous_pos = [
    POInput(
        po_id="PO-2026-A001",
        supplier_name="ShadyDeals Inc",       # unapproved supplier
        currency="USD",
        lead_time_days=30,
        total_value_usd=1000.00,
        line_items=[POLineItem(
            sku="SKU-A001", description="Generic Part",
            quantity=50, unit_price_usd=20.00, total_price_usd=1000.00
        )],
    ),
    POInput(
        po_id="PO-2026-A002",
        supplier_name="ACME Corp",
        currency="EUR",                        # wrong currency
        lead_time_days=30,
        total_value_usd=1000.00,
        line_items=[POLineItem(
            sku="SKU-A002", description="Electronic Module",
            quantity=40, unit_price_usd=25.00, total_price_usd=1000.00
        )],
    ),
    POInput(
        po_id="PO-2026-A003",
        supplier_name="ACME Corp",
        currency="USD",
        lead_time_days=30,
        total_value_usd=3500.00,
        line_items=[POLineItem(
            sku="SKU-A003", description="Premium Sensor",
            quantity=100, unit_price_usd=35.00,   # $10 over unit price limit
            total_price_usd=3500.00
        )],
    ),
    POInput(
        po_id="PO-2026-A004",
        supplier_name="QuickSupply Co",        # unapproved
        currency="USD",
        lead_time_days=120,                    # over lead time limit
        total_value_usd=75000.00,              # over total value limit
        line_items=[POLineItem(
            sku="SKU-A004", description="Industrial Robot Arm",
            quantity=50, unit_price_usd=1500.00,  # massively over unit price
            total_price_usd=75000.00
        )],
    ),
    POInput(
        po_id="PO-2026-A005",
        supplier_name="FastShip Inc",
        currency="USD",
        lead_time_days=30,
        total_value_usd=4995.00,
        line_items=[POLineItem(
            sku="SKU-A005", description="Bulk Connector",
            quantity=1500,                     # over 1000 unit per line limit
            unit_price_usd=3.33,
            total_price_usd=4995.00
        )],
    ),
]

# ── 5 Edge Cases ──────────────────────────────────────────────────────────────
# Boundary values and multi-rule interactions. Tests threshold handling.

edge_case_pos = [
    # Boundary: unit price exactly at limit ($25.00) — should pass
    POInput(
        po_id="PO-2026-E001",
        supplier_name="ACME Corp",
        currency="USD",
        lead_time_days=30,
        total_value_usd=2500.00,
        line_items=[POLineItem(
            sku="SKU-E001", description="Boundary Widget",
            quantity=100, unit_price_usd=25.00, total_price_usd=2500.00
        )],
    ),
    # Boundary: unit price one cent over ($25.01) — should fail
    POInput(
        po_id="PO-2026-E002",
        supplier_name="ACME Corp",
        currency="USD",
        lead_time_days=30,
        total_value_usd=2501.00,
        line_items=[POLineItem(
            sku="SKU-E002", description="Slightly Overpriced Widget",
            quantity=100, unit_price_usd=25.01, total_price_usd=2501.00
        )],
    ),
    # Boundary: lead time exactly 90 days — should pass
    POInput(
        po_id="PO-2026-E003",
        supplier_name="GlobalParts Ltd",
        currency="USD",
        lead_time_days=90,
        total_value_usd=1000.00,
        line_items=[POLineItem(
            sku="SKU-E003", description="Long Lead Part",
            quantity=40, unit_price_usd=25.00, total_price_usd=1000.00
        )],
    ),
    # Boundary: lead time 91 days — should fail
    POInput(
        po_id="PO-2026-E004",
        supplier_name="GlobalParts Ltd",
        currency="USD",
        lead_time_days=91,
        total_value_usd=1000.00,
        line_items=[POLineItem(
            sku="SKU-E004", description="Over Lead Time Part",
            quantity=40, unit_price_usd=25.00, total_price_usd=1000.00
        )],
    ),
    # Multi-rule: two MEDIUM violations, no HIGH — should escalate not reject
    POInput(
        po_id="PO-2026-E005",
        supplier_name="FastShip Inc",
        currency="USD",
        lead_time_days=91,          # MEDIUM violation
        total_value_usd=5050.00,
        line_items=[POLineItem(
            sku="SKU-E005", description="High Volume Part",
            quantity=1010,          # MEDIUM violation (over 1000 unit limit)
            unit_price_usd=5.00,
            total_price_usd=5050.00
        )],
    ),
]


def fix_total(po: POInput) -> POInput:
    """
    Recalculate total_value_usd from line items.
    Defensive fix for manual PO construction where totals might drift.
    """
    computed_total = round(sum(item.total_price_usd for item in po.line_items), 2)
    return po.model_copy(update={"total_value_usd": computed_total})


def write_dataset(pos: list[POInput], filename: str) -> None:
    fixed = [fix_total(po) for po in pos]
    records = [make_po_record(po) for po in fixed]
    path = DATASET_DIR / filename
    path.write_text(json.dumps(records, indent=2))
    print(f"  Written: {path} ({len(records)} records)")
    for r in records:
        violations = r["ground_truth_rule_ids"]
        print(f"    {r['po_id']} → {r['ground_truth_decision']:8s} | violations: {violations or 'none'}")


if __name__ == "__main__":
    print("\nGenerating evaluation dataset...\n")
    write_dataset(clean_pos, "clean_pos.json")
    print()
    write_dataset(anomalous_pos, "anomalous_pos.json")
    print()
    write_dataset(edge_case_pos, "edge_cases.json")
    print("\nDataset written to eval/dataset/")
    print("Commit these files. Do not regenerate unless rules.yaml changes.")