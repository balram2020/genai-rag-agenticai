"""Step 4 — adapt MCP tools into LangChain-style callable tools.

Run from the project root:
    python clients/04_langchain_tools_from_mcp.py
"""
import os
import sys
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_server.server import build_mcp_server


class LangChainLikeTool:
    """Minimal adapter to explain the concept.

    In a real project, langchain-mcp-adapters can convert MCP tools
    into proper LangChain BaseTool instances.
    """

    def __init__(self, name: str, description: str, server):
        self.name = name
        self.description = description
        self.server = server

    def invoke(self, arguments: Dict[str, Any]) -> dict:
        return self.server.call_tool(self.name, arguments)


def load_mcp_tools_as_langchain_like_tools():
    server = build_mcp_server()
    tools = []

    for tool in server.list_tools():
        tools.append(
            LangChainLikeTool(
                name=tool["name"],
                description=tool["description"],
                server=server,
            )
        )

    return tools


if __name__ == "__main__":
    tools = load_mcp_tools_as_langchain_like_tools()

    print("Loaded MCP tools as LangChain-like tools:\n")
    for tool in tools:
        print("-", tool.name, "::", tool.description)

    search_tool = next(t for t in tools if t.name == "search_policy_docs")

    print("\nInvoking LangChain-like tool wrapper...\n")
    result = search_tool.invoke(
        {
            "query": "travel reimbursement",
            "department": "finance",
            "max_results": 2,
        }
    )
    print(result)
