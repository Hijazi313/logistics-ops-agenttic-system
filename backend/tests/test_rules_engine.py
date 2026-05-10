"""
Unit tests for the RulesEngine.
These are fast, deterministic, and require no API keys.
Run with: uv run pytest tests/test_rules_engine.py -v
"""
import pytest
from src.models.po import POInput, POLineItem
from src.rules.engine import RulesEngine


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def engine():
    """Shared engine instance for all tests."""
    return RulesEngine()


@pytest.fixture
def clean_po():
    """A fully compliant PO — should produce zero violations."""
    return POInput(
        po_id="PO-2026-0001",
        supplier_name="ACME Corp",
        currency="USD",
        lead_time_days=30,
        total_value_usd=1000.00,
        submitted_by="muhammad@example.com",
        line_items=[
            POLineItem(
                sku="SKU-001",
                description="Industrial Widget A",
                quantity=50,
                unit_price_usd=20.00,
                total_price_usd=1000.00,
            )
        ],
    )


# ── Clean PO ──────────────────────────────────────────────────────────────────

def test_clean_po_produces_no_violations(engine, clean_po):
    violations = engine.validate(clean_po)
    assert violations == [], f"Expected no violations, got: {violations}"


# ── Currency Rule ─────────────────────────────────────────────────────────────

def test_invalid_currency_flagged(engine, clean_po):
    po = clean_po.model_copy(update={"currency": "EUR"})
    violations = engine.validate(po)
    rule_ids = [v.rule_id for v in violations]
    assert "currency" in rule_ids

def test_valid_currency_passes(engine, clean_po):
    violations = engine.validate(clean_po)
    rule_ids = [v.rule_id for v in violations]
    assert "currency" not in rule_ids


# ── Supplier Rule ─────────────────────────────────────────────────────────────

def test_unapproved_supplier_flagged(engine, clean_po):
    po = clean_po.model_copy(update={"supplier_name": "ShadyDeals Inc"})
    violations = engine.validate(po)
    rule_ids = [v.rule_id for v in violations]
    assert "approved_suppliers" in rule_ids

def test_approved_supplier_passes(engine, clean_po):
    violations = engine.validate(clean_po)
    rule_ids = [v.rule_id for v in violations]
    assert "approved_suppliers" not in rule_ids


# ── Lead Time Rule ────────────────────────────────────────────────────────────

def test_lead_time_over_limit_flagged(engine, clean_po):
    po = clean_po.model_copy(update={"lead_time_days": 91})
    violations = engine.validate(po)
    rule_ids = [v.rule_id for v in violations]
    assert "lead_time" in rule_ids

def test_lead_time_at_boundary_passes(engine, clean_po):
    """Boundary test: exactly 90 days should pass."""
    po = clean_po.model_copy(update={"lead_time_days": 90})
    violations = engine.validate(po)
    rule_ids = [v.rule_id for v in violations]
    assert "lead_time" not in rule_ids


# ── Total Order Value Rule ────────────────────────────────────────────────────

def test_total_value_over_limit_flagged(engine, clean_po):
    po = clean_po.model_copy(update={"total_value_usd": 50001.00})
    violations = engine.validate(po)
    rule_ids = [v.rule_id for v in violations]
    assert "total_order_value" in rule_ids


# ── Unit Price Rule ───────────────────────────────────────────────────────────

def test_unit_price_over_limit_flagged(engine):
    po = POInput(
        po_id="PO-2026-0002",
        supplier_name="ACME Corp",
        currency="USD",
        lead_time_days=30,
        total_value_usd=2950.00,
        line_items=[
            POLineItem(
                sku="SKU-999",
                description="Premium Widget",
                quantity=100,
                unit_price_usd=29.50,   # exceeds $25.00 limit
                total_price_usd=2950.00,
            )
        ],
    )
    violations = engine.validate(po)
    rule_ids = [v.rule_id for v in violations]
    assert "price_per_unit" in rule_ids


# ── Multiple Violations ───────────────────────────────────────────────────────

def test_multiple_violations_all_returned(engine):
    """
    Verify the engine collects ALL violations, not just the first one.
    This is the 'collect all' design — critical for usability.
    """
    po = POInput(
        po_id="PO-2026-0003",
        supplier_name="ShadyDeals Inc",  # bad supplier
        currency="EUR",                  # bad currency
        lead_time_days=120,              # bad lead time
        total_value_usd=500.00,
        line_items=[
            POLineItem(
                sku="SKU-001",
                description="Widget",
                quantity=5,
                unit_price_usd=100.00,   # bad unit price
                total_price_usd=500.00,
            )
        ],
    )
    violations = engine.validate(po)
    rule_ids = [v.rule_id for v in violations]
    assert "approved_suppliers" in rule_ids
    assert "currency" in rule_ids
    assert "lead_time" in rule_ids
    assert "price_per_unit" in rule_ids
    assert len(violations) == 4


# ── AnomalyDetail Structure ───────────────────────────────────────────────────

def test_anomaly_detail_fully_populated(engine, clean_po):
    """Verify every AnomalyDetail field is populated — no empty strings."""
    po = clean_po.model_copy(update={"currency": "EUR"})
    violations = engine.validate(po)
    assert len(violations) == 1
    v = violations[0]
    assert v.rule_id
    assert v.description
    assert v.expected
    assert v.actual
    assert v.explanation
    assert v.severity in ("HIGH", "MEDIUM", "LOW")