"""Step 5 — call MCP tools from inside a LangGraph workflow.

Run from the project root:
    python clients/05_langgraph_with_mcp_tools.py
"""
import os
import sys
from typing import Any, Dict, List, Optional, TypedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import END, START, StateGraph

from mcp_server.server import build_mcp_server


class SupportAgentState(TypedDict):
    user_id: str
    query: str
    intent: Optional[str]
    policy_results: List[Dict[str, Any]]
    ticket_result: Optional[Dict[str, Any]]
    final_answer: Optional[str]
    errors: List[Dict[str, Any]]


server = build_mcp_server()


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
    result = server.call_tool(
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
    return "create_ticket"


def create_ticket_node(state: SupportAgentState) -> SupportAgentState:
    result = server.call_tool(
        "create_support_ticket",
        {
            "user_id": state["user_id"],
            "issue_summary": state["query"],
            "priority": "medium",
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
    else:
        state["final_answer"] = "I could not resolve this request."

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


if __name__ == "__main__":
    app = build_graph()

    initial_state: SupportAgentState = {
        "user_id": "user-123",
        "query": "What is the production access policy?",
        "intent": None,
        "policy_results": [],
        "ticket_result": None,
        "final_answer": None,
        "errors": [],
    }

    result = app.invoke(initial_state)

    print("Final answer:")
    print(result["final_answer"])
    print("\nFull state:")
    print(result)
