"""Step 6 — production-grade guarded MCP tool calls inside LangGraph.

Every tool call goes through a boundary that checks:
    1. budget
    2. rate limit
    3. approval policy (for write actions)
    4. structured error normalization
    5. audit logging

Run from the project root:
    python clients/06_langgraph_with_guarded_mcp_tools.py
"""
import os
import sys
from typing import Any, Dict, List, Optional, TypedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import END, START, StateGraph

from mcp_server.server import build_mcp_server
from shared.approval import approve_write_action
from shared.audit_logger import AuditLogger
from shared.cost_tracker import CostTracker
from shared.error_contracts import error_response
from shared.rate_limiter import RateLimiter


class SupportAgentState(TypedDict):
    user_id: str
    query: str
    intent: Optional[str]
    policy_results: List[Dict[str, Any]]
    ticket_result: Optional[Dict[str, Any]]
    final_answer: Optional[str]
    errors: List[Dict[str, Any]]
    audit_events: List[Dict[str, Any]]
    cost_summary: Dict[str, Any]


server = build_mcp_server()
rate_limiter = RateLimiter(max_calls_per_minute=5)
cost_tracker = CostTracker(budget_usd=0.01)
audit_logger = AuditLogger()

TOOL_COSTS = {
    "search_policy_docs": 0.001,
    "create_support_ticket": 0.002,
    "get_ticket_status": 0.001,
}


def guarded_mcp_tool_call(tool_name: str, arguments: dict) -> dict:
    """Production boundary around MCP tool calls."""

    estimated_cost = TOOL_COSTS.get(tool_name, 0.001)

    if not cost_tracker.can_afford(tool_name, estimated_cost):
        result = error_response(
            error_type="budget_exceeded",
            message=f"Budget exceeded before calling {tool_name}",
            recoverable=False,
            fallback_suggestion=(
                "Stop or ask user to continue with higher budget"
            ),
        )
        audit_logger.record(
            "tool_blocked_budget",
            {"tool": tool_name, "arguments": arguments},
        )
        return result

    allowed, retry_after = rate_limiter.check(tool_name)
    if not allowed:
        result = error_response(
            error_type="rate_limit",
            message=f"Rate limit reached for {tool_name}",
            recoverable=True,
            retry_after_seconds=retry_after,
            fallback_suggestion="Try cached result or retry later",
        )
        audit_logger.record(
            "tool_blocked_rate_limit",
            {"tool": tool_name, "retry_after": retry_after},
        )
        return result

    if tool_name == "create_support_ticket":
        approval = approve_write_action(tool_name, arguments)
        if not approval["approved"]:
            result = error_response(
                error_type="approval_required",
                message=approval["reason"],
                recoverable=True,
                fallback_suggestion=(
                    "Ask human reviewer to approve the write action"
                ),
                details={"tool": tool_name, "arguments": arguments},
            )
            audit_logger.record(
                "tool_blocked_approval",
                {"tool": tool_name, "reason": approval["reason"]},
            )
            return result

    result = server.call_tool(tool_name, arguments)

    if result.get("success"):
        cost = result.get("metadata", {}).get("cost_usd", estimated_cost)
        cost_tracker.record(tool_name, cost)
        audit_logger.record(
            "tool_success", {"tool": tool_name, "cost": cost}
        )
    else:
        audit_logger.record(
            "tool_error", {"tool": tool_name, "error": result}
        )

    return result


def classify_query(state: SupportAgentState) -> SupportAgentState:
    query = state["query"].lower()

    if any(
        word in query
        for word in [
            "policy",
            "leave",
            "reimbursement",
            "access",
            "production",
        ]
    ):
        state["intent"] = "policy_question"
    else:
        state["intent"] = "support_request"

    return state


def search_policy_node(state: SupportAgentState) -> SupportAgentState:
    result = guarded_mcp_tool_call(
        "search_policy_docs",
        {
            "query": state["query"],
            "department": "all",
            "max_results": 3,
        },
    )

    if result["success"]:
        state["policy_results"] = result["data"]["results"]
    else:
        state["errors"].append(result)

    return state


def decide_resolution(state: SupportAgentState) -> str:
    if state["policy_results"]:
        return "final_response"

    if state["errors"]:
        last_error = state["errors"][-1]
        if last_error["error_type"] == "budget_exceeded":
            return "final_response"

    return "create_ticket"


def create_ticket_node(state: SupportAgentState) -> SupportAgentState:
    priority = "high" if "urgent" in state["query"].lower() else "medium"

    result = guarded_mcp_tool_call(
        "create_support_ticket",
        {
            "user_id": state["user_id"],
            "issue_summary": state["query"],
            "priority": priority,
        },
    )

    if result["success"]:
        state["ticket_result"] = result["data"]
    else:
        state["errors"].append(result)

    return state


def final_response_node(state: SupportAgentState) -> SupportAgentState:
    if state["policy_results"]:
        doc = state["policy_results"][0]
        state["final_answer"] = f"Based on {doc['title']}: {doc['content']}"
    elif state["ticket_result"]:
        ticket_id = state["ticket_result"]["ticket_id"]
        state["final_answer"] = (
            f"I could not resolve this from policy docs. "
            f"I created ticket {ticket_id}."
        )
    elif state["errors"]:
        last_error = state["errors"][-1]
        state["final_answer"] = (
            f"I could not complete the request. Reason: {last_error['message']}"
        )
    else:
        state["final_answer"] = "I could not resolve this request."

    state["audit_events"] = audit_logger.all_events()
    state["cost_summary"] = cost_tracker.summary()
    return state


def build_graph():
    graph = StateGraph(SupportAgentState)

    graph.add_node("classify_query", classify_query)
    graph.add_node("search_policy", search_policy_node)
    graph.add_node("create_ticket", create_ticket_node)
    graph.add_node("final_response", final_response_node)

    graph.add_edge(START, "classify_query")
    graph.add_edge("classify_query", "search_policy")

    graph.add_conditional_edges(
        "search_policy",
        decide_resolution,
        {
            "create_ticket": "create_ticket",
            "final_response": "final_response",
        },
    )

    graph.add_edge("create_ticket", "final_response")
    graph.add_edge("final_response", END)

    return graph.compile()


def run_case(query: str):
    app = build_graph()
    initial_state: SupportAgentState = {
        "user_id": "user-123",
        "query": query,
        "intent": None,
        "policy_results": [],
        "ticket_result": None,
        "final_answer": None,
        "errors": [],
        "audit_events": [],
        "cost_summary": {},
    }

    result = app.invoke(initial_state)
    print("\nQUERY:", query)
    print("FINAL ANSWER:", result["final_answer"])
    print("ERRORS:", result["errors"])
    print("COST:", result["cost_summary"])
    print("AUDIT EVENTS:", len(result["audit_events"]))
    print("-" * 80)


if __name__ == "__main__":
    run_case("What is the production access policy?")
    run_case("I need help with a laptop replacement")
    run_case("Urgent: my laptop is broken and I cannot work")
