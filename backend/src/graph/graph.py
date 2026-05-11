"""
Graph assembly — the only place where nodes and edges are wired together.

Design rules applied here:
1. Nodes are registered by name — the name is the routing key used in edges
2. Unconditional edges use add_edge() — no function needed
3. Conditional edges use add_conditional_edges() with the router function
4. SqliteSaver is the checkpointer — required for interrupt/resume to work
5. The compiled graph is returned from a factory function, not a module global
   Reason: module globals initialise at import time; a factory lets callers
   control when the DB connection is opened (important for testing with tmp_path)

Graph topology (matches the architecture diagram in the requirements doc):

  START
    │
    ▼
  ingest ──[parse_errors?]──▶ fail_fast ──▶ END
    │
    │ [clean]
    ▼
  validate ──[violations?]──▶ anomaly_detect ──▶ human_review ──▶ resolve ──▶ persist ──▶ END
    │
    │ [clean]
    ▼
  resolve ──▶ persist ──▶ END
"""
import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from src.graph.state import POAgentState
from src.graph.nodes import (
    ingest_node,
    validate_node,
    anomaly_detect_node,
    human_review_node,
    resolve_node,
    persist_node,
    fail_fast_node,
)
from src.graph.edges import (
    route_after_ingest,
    route_after_validate,
)


def build_graph(db_path: str = "checkpoints.db") -> object:
    """
    Builds and compiles the PO anomaly detection graph.

    Args:
        db_path: Path to the SQLite checkpoint database.
                 Override in tests to use a tmp_path location.
                 Override in production to use a mounted volume path.

    Returns:
        A compiled LangGraph CompiledStateGraph ready for invoke/stream.

    Why SqliteSaver over MemorySaver:
        MemorySaver loses all state on process restart.
        If the process dies while a PO is awaiting human review,
        that thread is unrecoverable with MemorySaver.
        SqliteSaver persists to disk — the thread survives restarts.

    Why sqlite3.connect() and not from_conn_string():
        from_conn_string() is a convenience wrapper that opens its own
        connection. Passing our own connection gives us explicit control
        over the connection lifecycle — we can close it cleanly on shutdown.
        Both work; explicit is better for production code.
    """
    # ── Build the graph ───────────────────────────────────────────────────
    builder = StateGraph(POAgentState)

    # Register nodes — string name must match exactly what edges reference
    builder.add_node("ingest", ingest_node)
    builder.add_node("validate", validate_node)
    builder.add_node("anomaly_detect", anomaly_detect_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("resolve", resolve_node)
    builder.add_node("persist", persist_node)
    builder.add_node("fail_fast", fail_fast_node)

    # ── Wire edges ────────────────────────────────────────────────────────

    # Entry point
    builder.add_edge(START, "ingest")

    # After ingest: conditional — route on parse_errors
    builder.add_conditional_edges(
        "ingest",
        route_after_ingest,
        {
            "validate": "validate",
            "fail_fast": "fail_fast",
        }
    )

    # After validate: conditional — route on rule_violations
    builder.add_conditional_edges(
        "validate",
        route_after_validate,
        {
            "anomaly_detect": "anomaly_detect",
            "resolve": "resolve",
        }
    )

    # Unconditional from anomaly_detect onward (violation path)
    builder.add_edge("anomaly_detect", "human_review")
    builder.add_edge("human_review", "resolve")

    # Both paths converge at resolve → persist → END
    builder.add_edge("resolve", "persist")
    builder.add_edge("persist", END)

    # Fail-fast terminal
    builder.add_edge("fail_fast", END)

    # ── Attach checkpointer and compile ───────────────────────────────────
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    return builder.compile(checkpointer=checkpointer)
