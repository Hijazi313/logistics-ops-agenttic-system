"""
Factory for building the initial POAgentState.

Why this exists:
- POAgentState has 9 fields. Constructing it inline in 3+ places
  (API, CLI, tests) creates drift when fields change.
- This is the single definition of "what an empty state looks like".
- All mutable fields initialised to empty lists/False/None —
  never left undefined, which causes KeyError in node logic.
"""
from src.models.po import POInput
from src.graph.state import POAgentState


def build_initial_state(po_input: POInput) -> POAgentState:
    """
    Returns a fully initialised POAgentState for a new PO run.

    Every field is explicitly set — no implicit defaults.
    This makes the contract visible and prevents KeyError bugs
    when nodes call state.get() on an uninitialised key.
    """
    return POAgentState(
        # Input
        po_input=po_input,

        # Ingest
        parse_errors=[],

        # Validation
        rule_violations=[],

        # LLM
        anomaly_analysis=None,

        # Human-in-the-loop
        awaiting_human=False,
        human_decision=None,
        approver_id=None,

        # Output
        final_result=None,
        audit_written=False,
    )