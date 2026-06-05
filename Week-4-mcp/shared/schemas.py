"""Pydantic schemas that define the tool contract.

The schema IS the API documentation the LLM reads at runtime.
The tighter the schema, the less the model has to guess.
"""
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


class PolicySearchInput(BaseModel):
    query: str = Field(..., description="User question about company policy")
    department: Literal["hr", "finance", "engineering", "all"] = "all"
    max_results: int = Field(default=3, ge=1, le=5)


class CreateTicketInput(BaseModel):
    user_id: str = Field(..., description="Unique user ID")
    issue_summary: str = Field(..., min_length=5)
    priority: Literal["low", "medium", "high"] = "medium"


class TicketStatusInput(BaseModel):
    ticket_id: str = Field(..., description="Ticket ID, for example TKT-1001")


class ToolMetadata(BaseModel):
    source: str
    latency_ms: int = 0
    cost_usd: float = 0.0


class ToolSuccessResponse(BaseModel):
    success: Literal[True] = True
    data: Dict[str, Any]
    metadata: ToolMetadata


class ToolErrorResponse(BaseModel):
    success: Literal[False] = False
    error_type: Literal[
        "validation_error",
        "rate_limit",
        "timeout",
        "auth_error",
        "provider_error",
        "budget_exceeded",
        "approval_required",
        "approval_rejected",
        "no_results",
        "all_sources_failed",
    ]
    message: str
    recoverable: bool = True
    retry_after_seconds: Optional[int] = None
    fallback_suggestion: Optional[str] = None
    details: Dict[str, Any] = {}
