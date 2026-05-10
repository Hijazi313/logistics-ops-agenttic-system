"""
Input schema for Purchase Orders.
This is the contract at the system boundary — what the outside world sends us.
We validate aggressively here so the rest of the system can trust the data.
"""
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional


class POLineItem(BaseModel):
    """A single line in the purchase order."""
    sku: str = Field(..., description="Stock Keeping Unit identifier")
    description: str = Field(..., description="Human-readable item description")
    quantity: int = Field(..., gt=0, description="Number of units ordered")
    unit_price_usd: float = Field(..., gt=0, description="Price per unit in USD")
    total_price_usd: float = Field(..., gt=0, description="quantity * unit_price_usd")

    @model_validator(mode="after")
    def validate_line_total(self) -> "POLineItem":
        """
        Cross-field validation: total must equal quantity * unit_price.
        We allow a small float tolerance for rounding.
        """
        expected = round(self.quantity * self.unit_price_usd, 2)
        actual = round(self.total_price_usd, 2)
        if abs(expected - actual) > 0.05:
            raise ValueError(
                f"Line total mismatch for SKU {self.sku}: "
                f"expected {expected}, got {actual}"
            )
        return self


class POInput(BaseModel):
    """
    Top-level Purchase Order input.
    This is the entry point into the agent system.
    """
    po_id: str = Field(..., description="Unique PO identifier")
    supplier_name: str = Field(..., description="Name of the supplier")
    currency: str = Field(..., description="Order currency code (e.g. USD)")
    lead_time_days: int = Field(..., gt=0, description="Expected delivery time in days")
    line_items: list[POLineItem] = Field(..., min_length=1, description="Line items")
    total_value_usd: float = Field(..., gt=0, description="Sum of all line totals")
    submitted_by: Optional[str] = Field(None, description="User who submitted the PO")

    @field_validator("po_id")
    @classmethod
    def po_id_format(cls, v: str) -> str:
        """
        Enforce PO ID format: PO-YYYY-NNNN
        Deterministic validation at the boundary — not the LLM's job.
        """
        import re
        if not re.match(r"^PO-\d{4}-\d{4}$", v):
            raise ValueError(
                f"PO ID must match format PO-YYYY-NNNN, got: {v}"
            )
        return v

    @model_validator(mode="after")
    def validate_total(self) -> "POInput":
        """
        Cross-field validation: total_value_usd must match sum of line totals.
        Again, data integrity — not the LLM's concern.
        """
        computed = round(sum(item.total_price_usd for item in self.line_items), 2)
        declared = round(self.total_value_usd, 2)
        if abs(computed - declared) > 0.10:
            raise ValueError(
                f"PO total mismatch: line items sum to {computed}, "
                f"but total_value_usd declared as {declared}"
            )
        return self

    @field_validator("currency")
    @classmethod
    def currency_uppercase(cls, v: str) -> str:
        return v.upper().strip()