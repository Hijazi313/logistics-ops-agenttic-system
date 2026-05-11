"""
PO analysis endpoints.

POST /analyze
    Accepts a POInput payload.
    Runs the graph synchronously (offloaded to thread pool).
    Returns either:
      - POAnalysisResult  (clean PO — graph completed)
      - InterruptResponse (anomalous PO — graph paused awaiting human)

POST /resume/{thread_id}
    Accepts a ResumeRequest payload.
    Resumes a paused graph thread with the human decision.
    Returns POAnalysisResult.

Design decisions:
    - graph.invoke() is blocking — we use asyncio.to_thread() to avoid
      blocking the FastAPI event loop on every request.
    - thread_id is generated server-side for /analyze (UUID4) so callers
      don't need to manage IDs. It's returned in the response so they can
      use it for /resume.
    - We never expose internal graph state to the caller — only the
      final result or the interrupt payload. The graph's TypedDict is
      an internal contract.
"""
import asyncio
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from langgraph.types import Command

from src.api.dependencies import get_graph
from src.models.po import POInput
from src.models.result import POAnalysisResult
from src.graph.initial_state import build_initial_state

router = APIRouter()


# ── Response schemas ──────────────────────────────────────────────────────────

class InterruptResponse(BaseModel):
    """
    Returned when the graph pauses at human_review.
    The caller uses thread_id to resume via POST /resume/{thread_id}.
    """
    status: str = "awaiting_human_review"
    thread_id: str
    po_id: str
    violation_count: int
    llm_recommendation: str
    confidence: str
    reasoning: str
    recommended_action: str
    anomalies: list[dict]
    instruction: str


class AnalyzeResponse(BaseModel):
    """
    Returned when the graph completes without interrupt (clean PO).
    """
    status: str = "completed"
    thread_id: str
    result: POAnalysisResult


class ResumeRequest(BaseModel):
    """
    Human reviewer's decision to resume a paused graph thread.
    """
    decision: str
    """Must be 'approve' or 'reject'."""

    approver_id: str
    """ID of the human reviewer. Stored in the audit log."""


class ResumeResponse(BaseModel):
    """
    Returned after a successful graph resume and completion.
    """
    status: str = "completed"
    thread_id: str
    result: POAnalysisResult


# ── POST /analyze ─────────────────────────────────────────────────────────────

@router.post(
    "/analyze",
    response_model=AnalyzeResponse | InterruptResponse,
    summary="Analyze a Purchase Order",
    description=(
        "Submits a PO for compliance analysis. "
        "Returns a completed result for clean POs, or an interrupt payload "
        "requiring human review for anomalous POs."
    ),
    tags=["analysis"],
)
async def analyze(
    po_input: POInput,
    graph: Annotated[object, Depends(get_graph)],
) -> AnalyzeResponse | InterruptResponse:
    """
    Entry point for PO analysis.

    Flow:
    1. Generate a unique thread_id for this PO run
    2. Build initial graph state from the POInput
    3. Invoke graph in a thread pool (blocking call → non-blocking route)
    4. Inspect result:
       - No interrupt → return completed AnalyzeResponse
       - Interrupt present → return InterruptResponse with thread_id
    """
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = build_initial_state(po_input)

    try:
        # graph.invoke() is synchronous — offload to thread pool
        # This prevents blocking the FastAPI event loop
        result = await asyncio.to_thread(
            graph.invoke,
            initial_state,
            config,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Graph execution failed: {str(e)}"
        )

    # ── Check for interrupt ───────────────────────────────────────────────
    interrupts = result.get("__interrupt__", [])

    if interrupts:
        payload = interrupts[0].value
        return InterruptResponse(
            thread_id=thread_id,
            po_id=payload.get("po_id", po_input.po_id),
            violation_count=payload.get("violation_count", 0),
            llm_recommendation=payload.get("llm_recommendation", "escalate"),
            confidence=payload.get("confidence", "LOW"),
            reasoning=payload.get("reasoning", ""),
            recommended_action=payload.get("recommended_action", ""),
            anomalies=payload.get("anomalies", []),
            instruction=payload.get("instruction", ""),
        )

    # ── Clean PO — graph completed ────────────────────────────────────────
    final = result.get("final_result")
    if final is None:
        raise HTTPException(
            status_code=500,
            detail="Graph completed but produced no final_result. Check graph wiring."
        )

    return AnalyzeResponse(
        thread_id=thread_id,
        result=final,
    )


# ── POST /resume/{thread_id} ──────────────────────────────────────────────────

@router.post(
    "/resume/{thread_id}",
    response_model=ResumeResponse,
    summary="Resume a paused PO review",
    description=(
        "Resumes a graph thread paused at human_review. "
        "Provide the thread_id from the /analyze response and the human decision."
    ),
    tags=["analysis"],
)
async def resume(
    thread_id: str,
    resume_request: ResumeRequest,
    graph: Annotated[object, Depends(get_graph)],
) -> ResumeResponse:
    """
    Resumes a paused graph with a human decision.

    Flow:
    1. Validate the decision value
    2. Build Command(resume=...) with the human decision
    3. Invoke graph with the same thread_id — checkpointer restores state
    4. Return the final POAnalysisResult

    Why we validate decision here and not only in the graph:
        Fail fast at the API boundary. If the decision is invalid,
        return a 422 immediately — don't waste a graph invocation.
    """
    if resume_request.decision not in ("approve", "reject"):
        raise HTTPException(
            status_code=422,
            detail=f"decision must be 'approve' or 'reject', got: '{resume_request.decision}'"
        )

    config = {"configurable": {"thread_id": thread_id}}
    resume_payload = {
        "decision": resume_request.decision,
        "approver_id": resume_request.approver_id,
    }

    try:
        result = await asyncio.to_thread(
            graph.invoke,
            Command(resume=resume_payload),
            config,
        )
    except Exception as e:
        # Common failure: thread_id not found in checkpointer
        # This happens if the ID is wrong or the DB was wiped
        raise HTTPException(
            status_code=404,
            detail=f"Thread '{thread_id}' not found or already completed. Error: {str(e)}"
        )

    final = result.get("final_result")
    if final is None:
        raise HTTPException(
            status_code=500,
            detail="Graph resumed but produced no final_result."
        )

    return ResumeResponse(
        thread_id=thread_id,
        result=final,
    )