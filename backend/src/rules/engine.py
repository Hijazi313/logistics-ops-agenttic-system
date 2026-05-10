"""
Deterministic rules engine.
This is intentionally pure Python — no LLM, no async, no side effects.
It takes a POInput, returns a list of AnomalyDetail.
Testable, predictable, fast.
"""
import yaml
from pathlib import Path
from src.models.po import POInput
from src.models.result import AnomalyDetail


class RulesEngine:
    """
    Loads business rules from YAML config and validates POs against them.

    Design principle: every method here is deterministic.
    Given the same input, it always produces the same output.
    The LLM layer above this adds reasoning and explanation.
    This layer adds correctness.
    """

    def __init__(self, rules_path: str | Path = "config/rules.yaml"):
        self.rules_path = Path(rules_path)
        self._rules = self._load_rules()

    def _load_rules(self) -> dict:
        if not self.rules_path.exists():
            raise FileNotFoundError(
                f"Rules config not found at {self.rules_path}. "
                "Did you create config/rules.yaml?"
            )
        with open(self.rules_path) as f:
            config = yaml.safe_load(f)
        return config.get("rules", {})

    def validate(self, po: POInput) -> list[AnomalyDetail]:
        """
        Run all rules against the PO.
        Returns a list of violations. Empty list = clean PO.

        Why collect all violations instead of fail-fast?
        Because an approver needs to see ALL issues at once,
        not fix one and resubmit five times.
        """
        violations: list[AnomalyDetail] = []

        violations.extend(self._check_currency(po))
        violations.extend(self._check_supplier(po))
        violations.extend(self._check_lead_time(po))
        violations.extend(self._check_total_order_value(po))
        violations.extend(self._check_line_items(po))

        return violations

    def _check_currency(self, po: POInput) -> list[AnomalyDetail]:
        rule = self._rules.get("currency", {})
        allowed = rule.get("allowed", ["USD"])
        if po.currency not in allowed:
            return [AnomalyDetail(
                rule_id="currency",
                description=rule.get("description", "Currency check"),
                expected=f"One of: {', '.join(allowed)}",
                actual=po.currency,
                explanation=f"PO uses {po.currency} which is not an accepted currency.",
                severity=rule.get("severity_if_violated", "HIGH"),
            )]
        return []

    def _check_supplier(self, po: POInput) -> list[AnomalyDetail]:
        rule = self._rules.get("approved_suppliers", {})
        approved = rule.get("list", [])
        if po.supplier_name not in approved:
            return [AnomalyDetail(
                rule_id="approved_suppliers",
                description=rule.get("description", "Supplier check"),
                expected=f"One of: {', '.join(approved)}",
                actual=po.supplier_name,
                explanation=(
                    f"'{po.supplier_name}' is not on the approved vendor list. "
                    "Unapproved suppliers require procurement review before proceeding."
                ),
                severity=rule.get("severity_if_violated", "HIGH"),
            )]
        return []

    def _check_lead_time(self, po: POInput) -> list[AnomalyDetail]:
        rule = self._rules.get("lead_time", {})
        max_days = rule.get("max_days", 90)
        if po.lead_time_days > max_days:
            return [AnomalyDetail(
                rule_id="lead_time",
                description=rule.get("description", "Lead time check"),
                expected=f"<= {max_days} days",
                actual=f"{po.lead_time_days} days",
                explanation=(
                    f"Lead time of {po.lead_time_days} days exceeds the {max_days}-day policy. "
                    "Extended lead times impact inventory planning."
                ),
                severity=rule.get("severity_if_violated", "MEDIUM"),
            )]
        return []

    def _check_total_order_value(self, po: POInput) -> list[AnomalyDetail]:
        rule = self._rules.get("total_order_value", {})
        max_usd = rule.get("max_usd", 50000.00)
        if po.total_value_usd > max_usd:
            return [AnomalyDetail(
                rule_id="total_order_value",
                description=rule.get("description", "Order value check"),
                expected=f"<= ${max_usd:,.2f} USD",
                actual=f"${po.total_value_usd:,.2f} USD",
                explanation=(
                    f"Total value of ${po.total_value_usd:,.2f} exceeds the "
                    f"${max_usd:,.2f} approval threshold. Requires CFO sign-off."
                ),
                severity=rule.get("severity_if_violated", "HIGH"),
            )]
        return []

    def _check_line_items(self, po: POInput) -> list[AnomalyDetail]:
        """
        Check each line item individually.
        Two rules apply per line: price_per_unit and quantity_per_line.
        We check all lines, not just the first violation.
        """
        violations = []
        price_rule = self._rules.get("price_per_unit", {})
        qty_rule = self._rules.get("quantity_per_line", {})
        max_price = price_rule.get("max_usd", 25.00)
        max_qty = qty_rule.get("max_units", 1000)

        for item in po.line_items:
            if item.unit_price_usd > max_price:
                violations.append(AnomalyDetail(
                    rule_id="price_per_unit",
                    description=price_rule.get("description", "Unit price check"),
                    expected=f"<= ${max_price:.2f} USD/unit",
                    actual=f"${item.unit_price_usd:.2f} USD/unit (SKU: {item.sku})",
                    explanation=(
                        f"SKU {item.sku} priced at ${item.unit_price_usd:.2f}/unit, "
                        f"which is ${item.unit_price_usd - max_price:.2f} above limit. "
                        "Consider renegotiating or splitting the order."
                    ),
                    severity=price_rule.get("severity_if_violated", "HIGH"),
                ))

            if item.quantity > max_qty:
                violations.append(AnomalyDetail(
                    rule_id="quantity_per_line",
                    description=qty_rule.get("description", "Quantity check"),
                    expected=f"<= {max_qty} units per line",
                    actual=f"{item.quantity} units (SKU: {item.sku})",
                    explanation=(
                        f"SKU {item.sku} ordered in quantity {item.quantity}, "
                        f"which exceeds the {max_qty}-unit per-line limit. "
                        "Split into multiple line items."
                    ),
                    severity=qty_rule.get("severity_if_violated", "MEDIUM"),
                ))

        return violations