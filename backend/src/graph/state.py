"""
POAgentState — the single source of shared state for the entire graph.

Design rules:
1. Every field must have a type annotation — no bare 'Any'
2. Optional fields default to None — nodes only populate them when relevant
3. No business logic here — this is pure data structure
4. Fields are grouped by lifecycle: input → intermediate → LLM → human → output
"""
from typing import TypedDict, Optional
from src.models.po import POInput
from src.models.result import AnomalyDetail, POAnalysisResult


class POAgentState(TypedDict):
    """
    Shared state flowing through every node in the PO anomaly graph.

    Lifecycle of a PO through this state:
    1. po_input arrives at ingest node
    2. parse_errors populated if PO is malformed → graph ends at fail_fast
    3. rule_violations populated by validate node (deterministic)
    4. anomaly_analysis populated by LLM reasoning node
    5. Graph pauses at human_review interrupt — awaiting_human = True
    6. human_decision + approver_id injected via Command(resume=...)
    7. final_result assembled by resolve node
    8. audit_written = True after persist node writes the log
    """

    # ── Input ──────────────────────────────────────────────────────────────
    po_input: POInput
    """The validated PO coming in from the API boundary."""

    # ── Ingest / Parse ─────────────────────────────────────────────────────
    parse_errors: list[str]
    """
    Pydantic validation errors from ingest.
    Non-empty → graph routes to fail_fast, bypasses all other nodes.
    """

    # ── Deterministic Validation ───────────────────────────────────────────
    rule_violations: list[AnomalyDetail]
    """
    Output of the RulesEngine — deterministic, no LLM involved.
    Empty list → PO is clean → routes directly to resolve (approve).
    Non-empty → routes to anomaly_detect for LLM enrichment.
    """

    # ── LLM Reasoning ─────────────────────────────────────────────────────
    anomaly_analysis: Optional[POAnalysisResult]
    """
    Structured output from the reasoning LLM.
    Populated only when rule_violations is non-empty.
    None for clean POs.
    """

    # ── Human-in-the-Loop ─────────────────────────────────────────────────
    awaiting_human: bool
    """
    Flag set to True when the graph reaches the interrupt node.
    The API layer reads this to know the graph is paused.
    """

    human_decision: Optional[str]
    """
    Value injected via Command(resume={"decision": "approve"|"reject"}).
    Only populated after human acts on the interrupt.
    """

    approver_id: Optional[str]
    """
    ID of the human who reviewed. Stored in the audit log.
    None for auto-approved clean POs.
    """

    # ── Output ─────────────────────────────────────────────────────────────
    final_result: Optional[POAnalysisResult]
    """
    The complete analysis result assembled by the resolve node.
    This is what the API returns to the caller.
    """

    audit_written: bool
    """
    Set to True after persist_node writes the audit log.
    Guards against double-writes if graph is re-invoked accidentally.
    """