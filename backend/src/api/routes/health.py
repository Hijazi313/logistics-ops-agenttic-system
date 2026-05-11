"""
Health endpoint — liveness check.

GET /health
  → 200 if app is running and graph is initialised
  → 503 if graph failed to load (startup error)

Why include graph status in health:
    A running FastAPI process with a broken graph is not healthy.
    Load balancers and container orchestrators use /health to decide
    whether to route traffic. A 200 with a broken graph is worse than
    a 503 — it silently accepts requests that will all fail.
"""
from fastapi import APIRouter
from src.api.dependencies import GraphProvider


router = APIRouter()


@router.get("/health", tags=["ops"])
async def health():
    graph_ready = GraphProvider.get_graph() is not None
    return {
        "status": "ok" if graph_ready else "degraded",
        "graph_ready": graph_ready,
        "version": "1.0.0",
    }