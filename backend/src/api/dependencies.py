"""
FastAPI dependencies for the PO anomaly agent API.

The compiled graph is a shared resource — built once at startup,
injected into routes via dependency injection.

Why app.state and not a module global:
    Module globals initialise at import time — before lifespan runs
    and before env vars are loaded. app.state is populated during
    lifespan startup, guaranteeing correct initialisation order.

Why Annotated[..., Depends(...)]:
    The modern FastAPI pattern for type-safe dependency injection.
    Avoids the older `= Depends(get_graph)` inline syntax.
"""
from typing import Optional
from fastapi import HTTPException


class GraphProvider:
    """
    Singleton provider for the compiled graph.
    Decoupled from app.state to align with dependency injection best practices.
    """
    _graph: Optional[object] = None

    @classmethod
    def set_graph(cls, graph: object) -> None:
        cls._graph = graph

    @classmethod
    def get_graph(cls) -> Optional[object]:
        return cls._graph


def get_graph() -> object:
    """
    Dependency to retrieve the compiled graph.
    Used by routes to access the graph singleton.
    Raises 503 if the graph hasn't been initialized during lifespan startup.
    """
    graph = GraphProvider.get_graph()
    if graph is None:
        raise HTTPException(
            status_code=503,
            detail="Graph not initialised. Application is still starting up."
        )
    return graph