"""Step 3 — discover tools via the MCP-style server and call them.

Run from the project root:
    python clients/03_list_and_call_mcp_tools.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_server.server import build_mcp_server


server = build_mcp_server()

print("Connected to MCP-style server:", server.server_name)

print("\nListing tools exposed by server:\n")
for tool in server.list_tools():
    print("Tool:", tool["name"])
    print("Description:", tool["description"])
    print("Input schema:", tool["input_schema"])
    print("-" * 60)

print("\nCalling search_policy_docs through MCP-style client...\n")
result = server.call_tool(
    "search_policy_docs",
    {
        "query": "production access",
        "department": "engineering",
        "max_results": 3,
    },
)
print(result)

print("\nCalling create_support_ticket through MCP-style client...\n")
result = server.call_tool(
    "create_support_ticket",
    {
        "user_id": "user-123",
        "issue_summary": "I could not find the policy for laptop replacement",
        "priority": "medium",
    },
)
print(result)
