"""Build the MCP-style server that exposes tools to clients."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_server.tools import (
    create_support_ticket,
    get_ticket_status,
    search_policy_docs,
)
from shared.mcp_registry_fallback import MCPToolRegistry


def build_mcp_server() -> MCPToolRegistry:
    server = MCPToolRegistry(server_name="company-support-mcp-server")

    server.register_tool(
        name="search_policy_docs",
        description=(
            "Search internal company policy documents for HR, finance, "
            "and engineering policies."
        ),
        function=search_policy_docs,
        input_schema={
            "query": "string: user question about policy",
            "department": "one of: hr, finance, engineering, all",
            "max_results": "integer between 1 and 5",
        },
    )

    server.register_tool(
        name="create_support_ticket",
        description=(
            "Create a support ticket when the user's issue cannot be "
            "resolved from policy docs."
        ),
        function=create_support_ticket,
        input_schema={
            "user_id": "string",
            "issue_summary": "string, minimum 5 characters",
            "priority": "one of: low, medium, high",
        },
    )

    server.register_tool(
        name="get_ticket_status",
        description="Get the current status of an existing support ticket.",
        function=get_ticket_status,
        input_schema={
            "ticket_id": "string, for example TKT-1001",
        },
    )

    return server


if __name__ == "__main__":
    server = build_mcp_server()
    print("MCP-style server started:", server.server_name)
    print("Available tools:")
    for tool in server.list_tools():
        print("-", tool["name"], "::", tool["description"])
