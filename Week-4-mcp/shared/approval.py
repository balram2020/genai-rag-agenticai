"""Mock approval policy for write actions.

In production, this could be a human approval interrupt in LangGraph.
For class, we simulate the policy locally.
"""


def approve_write_action(tool_name: str, arguments: dict) -> dict:
    priority = arguments.get("priority", "medium")

    if tool_name == "create_support_ticket" and priority == "high":
        return {
            "approved": False,
            "reason": "High priority write actions require human approval",
        }

    return {
        "approved": True,
        "reason": "Action approved by mock policy",
    }
