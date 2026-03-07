"""
Brain Agent REST API endpoints.

POST /api/v1/brain/ask   — read-only brain query (natural language → synthesized answer)
POST /api/v1/brain/do    — read-write brain operation (with confirmation gate for destructive ops)
GET  /api/v1/brain/audit — brain audit log
GET  /api/v1/brain/status — brain agent status (model, config)
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from app.brain.agent import brain_agent, BrainResult
from app.brain.tools import _log_audit
from app.database import engine
from fastapi import APIRouter, Query
from pydantic import BaseModel
from sqlalchemy import text as sql_text

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/brain", tags=["brain"])


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class BrainAskRequest(BaseModel):
    request: str
    user_id: str


class BrainDoRequest(BaseModel):
    request: str
    user_id: str
    confirmed: bool = False
    # When confirmed=True and this was a dry-run re-send, include the planned actions
    planned_actions: Optional[List[Dict[str, Any]]] = None


class BrainResponse(BaseModel):
    answer: str
    steps: int
    tools_called: List[str]
    user_id: str
    success: bool
    error: Optional[str] = None
    requires_confirmation: bool = False
    planned_actions: Optional[List[Dict[str, Any]]] = None
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/ask", response_model=BrainResponse)
async def brain_ask(req: BrainAskRequest):
    """
    Ask the brain a natural-language question about stored memories.
    Read-only mode — the brain can search but cannot modify data.

    Examples:
    - "What hobbies does Steven have?"
    - "What GPU does Steven use?"
    - "Summarize everything you know about Echoes of the Fallen"
    - "How are Steven and Docker related?"
    """
    result = brain_agent.run(
        request=req.request,
        user_id=req.user_id,
        read_only=True,
    )
    return _result_to_response(result)


@router.post("/do", response_model=BrainResponse)
async def brain_do(req: BrainDoRequest):
    """
    Perform a natural-language memory operation (store, update, delete, reorganize).
    Destructive operations require confirmation.

    Flow:
    1. Send with confirmed=false → brain plans the operation, returns requires_confirmation=true
       with planned_actions showing what will be done.
    2. Re-send with confirmed=true and the planned_actions from step 1 → brain executes.

    For non-destructive operations (e.g., storing a new fact), executes immediately.

    Examples:
    - "Remember that Steven switched from UE5 to Godot"
    - "Delete all memories about dark mode"
    - "Update Steven's job title to Senior Developer"
    """
    confirm_destructive = os.environ.get("BRAIN_CONFIRM_DESTRUCTIVE", "true").lower() not in ("false", "0", "no")

    if req.confirmed and req.planned_actions:
        # Execute previously planned actions
        result = brain_agent.run_confirmed(
            request=req.request,
            user_id=req.user_id,
            planned_actions=req.planned_actions,
        )
        return _result_to_response(result)

    # First pass: run the brain
    # If confirmation is required, do a dry run to collect destructive actions
    if confirm_destructive and not req.confirmed:
        result = brain_agent.run(
            request=req.request,
            user_id=req.user_id,
            read_only=False,
            dry_run=True,
        )
        # If the brain used destructive tools, gate with confirmation
        if result.requires_confirmation and result.planned_actions:
            return _result_to_response(result)
        # No destructive tools → the dry run is equivalent to real run,
        # but we need to re-run for real since dry_run doesn't execute
        # Only re-run if there were no destructive actions
        # (read + store is fine to execute directly)

    # Execute for real
    result = brain_agent.run(
        request=req.request,
        user_id=req.user_id,
        read_only=False,
        dry_run=False,
    )
    return _result_to_response(result)


@router.get("/audit")
async def brain_audit(
    user_id: str = Query(None, description="Filter by user ID"),
    limit: int = Query(50, description="Number of rows to return"),
):
    """
    Return the last N brain_audit rows, optionally filtered by user_id.
    """
    try:
        with engine.connect() as conn:
            if user_id:
                result = conn.execute(
                    sql_text(
                        "SELECT id, ts, tool, user_id, input, output, error "
                        "FROM brain_audit WHERE user_id = :user_id "
                        "ORDER BY id DESC LIMIT :limit"
                    ),
                    {"user_id": user_id, "limit": limit},
                )
            else:
                result = conn.execute(
                    sql_text(
                        "SELECT id, ts, tool, user_id, input, output, error "
                        "FROM brain_audit ORDER BY id DESC LIMIT :limit"
                    ),
                    {"limit": limit},
                )
            rows = [dict(row._mapping) for row in result]
        return {"rows": rows, "count": len(rows)}
    except Exception as e:
        logger.error(f"Audit query failed: {e}")
        return {"rows": [], "count": 0, "error": str(e)}


@router.get("/status")
async def brain_status():
    """Return brain agent configuration and status."""
    return {
        "model": brain_agent.model,
        "ollama_url": brain_agent.ollama_url,
        "max_steps": brain_agent.max_steps,
        "confirm_destructive": os.environ.get("BRAIN_CONFIRM_DESTRUCTIVE", "true"),
        "tools_available": [t["name"] for t in from_tools()],
    }


def from_tools():
    from app.brain.tools import TOOL_DEFINITIONS
    return TOOL_DEFINITIONS


def _result_to_response(result: BrainResult) -> BrainResponse:
    """Convert BrainResult dataclass to BrainResponse Pydantic model."""
    return BrainResponse(
        answer=result.answer,
        steps=result.steps,
        tools_called=result.tools_called,
        user_id=result.user_id,
        success=result.success,
        error=result.error,
        requires_confirmation=result.requires_confirmation,
        planned_actions=result.planned_actions,
        elapsed_seconds=result.elapsed_seconds,
    )
