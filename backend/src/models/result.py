"""
Output schemas for analysis results and audit logging.
These are the contracts the downstream systems (API, audit log) consume.
"""
from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import datetime, timezone


class AnomalyDetail(BaseModel):
    """
    A single rule violation with full audit trail.
    Structured so a non-technical compliance officer can read it.
    """
    rule_id: str = Field(..., description="Machine-readable rule identifier")
    description: str = Field(..., description="Human-readable rule description")
    expected: str = Field(..., description="What the rule expected (as string)")
    actual: str = Field(..., description="What was found in the PO (as string)")
    explanation: str = Field(..., description="LLM-generated reasoning about why this matters")
    severity: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        ..., description="Impact severity of this violation"
    )


class POAnalysisResult(BaseModel):
    """
    The complete analysis output from the agent.
    This is what gets returned by the API and stored in the audit log.
    """
    po_id: str
    decision: Literal["approve", "escalate", "reject"] = Field(
        ...,
        description=(
            "approve: clean PO, no issues. "
            "escalate: anomalies detected, needs human review. "
            "reject: clear violations, should not proceed."
        )
    )
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        ...,
        description="Derived from number and severity of anomalies"
    )
    anomalies: list[AnomalyDetail] = Field(
        default_factory=list,
        description="All rule violations found. Empty list means clean PO."
    )
    recommended_action: str = Field(
        ...,
        description="Plain-English action for the reviewer"
    )
    reasoning_summary: str = Field(
        ...,
        description="LLM-generated summary of the analysis"
    )

    @classmethod
    def compute_confidence(cls, anomalies: list[AnomalyDetail]) -> Literal["HIGH", "MEDIUM", "LOW"]:
        """
        Deterministic confidence scoring — no LLM needed for this.
        HIGH severity violations → LOW confidence in the PO.
        """
        if not anomalies:
            return "HIGH"
        high_count = sum(1 for a in anomalies if a.severity == "HIGH")
        if high_count >= 1:
            return "LOW"
        if len(anomalies) >= 2:
            return "MEDIUM"
        return "HIGH"


class AuditLog(BaseModel):
    """
    Immutable audit record written after every decision.
    Append-only to audit_log.jsonl — never modified after write.
    """
    po_id: str
    decision: str
    confidence: str
    anomaly_count: int
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    human_actor: Optional[str] = Field(
        None,
        description="Approver ID if human reviewed. None if auto-approved."
    )
    langsmith_trace_id: Optional[str] = Field(
        None,
        description="LangSmith run ID for this trace. Use for debugging."
    )
    reasoning_snapshot: str = Field(
        ...,
        description="Full reasoning_summary from POAnalysisResult"
    )