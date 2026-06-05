"""Step 1 — call the tools as plain Python functions.

Run from the project root:
    python clients/01_call_python_tools.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_server.tools import create_support_ticket, search_policy_docs


print("Calling normal Python tools directly...\n")

result_1 = search_policy_docs(query="production access", department="engineering")
print("Policy search result:")
print(result_1)

print("\nCreating support ticket...")
result_2 = create_support_ticket(
    user_id="user-123",
    issue_summary="I need help getting production access",
    priority="medium",
)
print(result_2)
