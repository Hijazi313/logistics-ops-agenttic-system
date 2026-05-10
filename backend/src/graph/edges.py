"""
Conditional edge functions for the PO anomaly graph.

An edge function receives the full state and returns a string —
the name of the next node to route to.

Rules for writing edge functions:
- They must be exhaustive: every possible state must map to a node
- Never return a node name that doesn't exist in the graph
- Keep logic minimal — edges route, nodes act
- The 'default' branch should be the safe/conservative choice
"""
from src.graph.state import POAgentState


def route_after_ingest(state: POAgentState) -> str:
    """
    After ingest: did parsing succeed?

    Clean  → validate (proceed with rules check)
    Errors → fail_fast (structured error response, no further processing)
    """
    if state.get("parse_errors"):
        return "fail_fast"
    return "validate"


def route_after_validate(state: POAgentState) -> str:
    """
    After validate: did the rules engine find violations?

    Violations found → anomaly_detect (LLM enrichment + human review)
    Clean PO         → resolve (auto-approve, skip LLM and human entirely)

    This is the key cost-saving routing decision:
    Clean POs never touch the LLM. Zero tokens spent. Instant approval.
    """
    if state.get("rule_violations"):
        return "anomaly_detect"
    return "resolve"