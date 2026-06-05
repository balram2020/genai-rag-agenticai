"""Tool implementations exposed via the MCP-style server.

Each tool:
    - validates input through a Pydantic schema
    - returns a structured success or error envelope
    - documents its when-to-use and when-not-to-use rules

MCP does not make a bad tool production-ready.
It only makes the tool accessible.
The tool itself still needs a good contract.
"""
from shared.error_contracts import error_response, success_response
from shared.fake_data import POLICY_DOCS, TICKETS
from shared.schemas import (
    CreateTicketInput,
    PolicySearchInput,
    TicketStatusInput,
)


def search_policy_docs(
    query: str,
    department: str = "all",
    max_results: int = 3,
    force_empty: bool = False,
) -> dict:
    """
    Search internal policy documents.

    Use when:
    - user asks about HR, finance, or engineering policy
    - user asks about leave, reimbursement, access, or internal process

    Do NOT use when:
    - user asks for private employee data
    - user asks for payroll details
    - user asks for confidential documents

    Returns structured success or error response.
    """
    try:
        validated = PolicySearchInput(
            query=query, department=department, max_results=max_results
        )
    except Exception as exc:
        return error_response(
            error_type="validation_error",
            message="Invalid policy search input",
            recoverable=True,
            fallback_suggestion=(
                "Ask the user to provide a clearer policy question"
            ),
            details={"exception": str(exc)},
        )

    if force_empty:
        return error_response(
            error_type="no_results",
            message="No policy documents matched the query",
            recoverable=True,
            fallback_suggestion=(
                "Create a support ticket or ask a clarifying question"
            ),
        )

    import re

    stopwords = {
        "i", "a", "an", "the", "is", "to", "of", "in", "on", "at",
        "for", "and", "or", "with", "my", "me", "you", "your", "we",
        "it", "this", "that", "be", "do", "what", "how", "why",
        "need", "help", "please", "can", "could", "would", "should",
        "urgent", "asap", "immediately",
    }

    query_terms = [
        w for w in re.findall(r"[a-z]+", validated.query.lower())
        if w not in stopwords and len(w) > 2
    ]

    scored = []
    for doc in POLICY_DOCS:
        if (
            validated.department != "all"
            and doc["department"] != validated.department
        ):
            continue

        searchable_text = f"{doc['title']} {doc['content']}".lower()
        doc_words = set(re.findall(r"[a-z]+", searchable_text))
        score = sum(1 for term in query_terms if term in doc_words)
        if score > 0:
            scored.append((score, doc))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    matches = [doc for _, doc in scored]

    if not matches:
        return error_response(
            error_type="no_results",
            message="No policy documents matched the query",
            recoverable=True,
            fallback_suggestion=(
                "Create a support ticket if the user still needs help"
            ),
        )

    return success_response(
        data={
            "results": matches[: validated.max_results],
            "source_count": len(matches),
        },
        source="policy_docs",
        cost_usd=0.001,
        latency_ms=50,
    )


def create_support_ticket(
    user_id: str,
    issue_summary: str,
    priority: str = "medium",
) -> dict:
    """
    Create an internal support ticket.

    Use when:
    - policy search cannot resolve the user request
    - user needs follow-up from support team

    Do NOT use when:
    - user only asks an informational question that can be answered from policy docs
    - the action requires approval and approval has not been granted
    """
    try:
        validated = CreateTicketInput(
            user_id=user_id, issue_summary=issue_summary, priority=priority
        )
    except Exception as exc:
        return error_response(
            error_type="validation_error",
            message="Invalid ticket creation input",
            recoverable=True,
            fallback_suggestion=(
                "Ask for a clearer issue summary and valid priority"
            ),
            details={"exception": str(exc)},
        )

    ticket_id = f"TKT-{1000 + len(TICKETS) + 1}"
    ticket = {
        "ticket_id": ticket_id,
        "status": "open",
        "priority": validated.priority,
        "summary": validated.issue_summary,
        "user_id": validated.user_id,
    }
    TICKETS[ticket_id] = ticket

    return success_response(
        data=ticket,
        source="ticketing_system",
        cost_usd=0.002,
        latency_ms=80,
    )


def get_ticket_status(ticket_id: str) -> dict:
    """Get status of an existing support ticket."""
    try:
        validated = TicketStatusInput(ticket_id=ticket_id)
    except Exception as exc:
        return error_response(
            error_type="validation_error",
            message="Invalid ticket ID",
            recoverable=True,
            details={"exception": str(exc)},
        )

    ticket = TICKETS.get(validated.ticket_id)
    if not ticket:
        return error_response(
            error_type="no_results",
            message="Ticket not found",
            recoverable=True,
            fallback_suggestion=(
                "Check the ticket ID or create a new support ticket"
            ),
        )

    return success_response(
        data=ticket,
        source="ticketing_system",
        cost_usd=0.001,
        latency_ms=30,
    )
