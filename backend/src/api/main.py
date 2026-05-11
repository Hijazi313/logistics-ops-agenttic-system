"""
FastAPI application entry point.

Responsibilities:
  - Lifespan: build the compiled graph once at startup, store on app.state
  - Router registration: mount all route modules under their prefixes
  - Nothing else — no business logic lives here

Run:
    uv run fastapi dev src/api/main.py        # development (auto-reload)
    uv run fastapi run src/api/main.py        # production
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI

from config import load_env
from src.graph.graph import build_graph
from src.api.routes.health import router as health_router
from src.api.routes.analyze import router as analyze_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan — startup and shutdown logic.

    Startup (before yield):
      1. Load env vars from .env
      2. Build and compile the LangGraph graph
      3. Attach to app.state — available to all routes via dependency

    Shutdown (after yield):
      - app.state is cleared automatically by FastAPI
      - SQLite connection held by SqliteSaver closes with the process

    Why build_graph() is called here and not at module level:
      Module-level calls happen at import time — before env vars are loaded.
      Lifespan runs after FastAPI initialises — env vars are guaranteed loaded.
    """
    # ── Startup ───────────────────────────────────────────────────────────
    load_env()

    db_path = os.getenv("CHECKPOINT_DB_PATH", "checkpoints.db")
    graph = build_graph(db_path=db_path)
    
    from src.api.dependencies import GraphProvider
    GraphProvider.set_graph(graph)

    print(f"[startup] Graph compiled. Checkpoint DB: {db_path}")
    print(f"[startup] Audit log: {os.getenv('AUDIT_LOG_PATH', 'audit_log.jsonl')}")


    yield

    # ── Shutdown ──────────────────────────────────────────────────────────
    print("[shutdown] Application shutting down.")


app = FastAPI(
    title="PO Anomaly Detection Agent",
    description=(
        "LangGraph-powered Purchase Order compliance agent with "
        "human-in-the-loop review and immutable audit logging."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(analyze_router)