"""Teaching-friendly fallback for MCP concepts.

This is NOT the full MCP protocol. It simulates the two ideas
that matter most for the class:
    1. list_tools()
    2. call_tool(name, arguments)

Swap this for the official MCP SDK in production.
"""
from typing import Any, Callable, Dict


class MCPToolRegistry:
    def __init__(self, server_name: str):
        self.server_name = server_name
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register_tool(
        self,
        name: str,
        description: str,
        function: Callable,
        input_schema: Dict[str, Any],
    ):
        self._tools[name] = {
            "name": name,
            "description": description,
            "function": function,
            "input_schema": input_schema,
        }

    def list_tools(self):
        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["input_schema"],
            }
            for tool in self._tools.values()
        ]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> dict:
        if name not in self._tools:
            return {
                "success": False,
                "error_type": "validation_error",
                "message": f"Tool '{name}' not found",
                "recoverable": True,
                "fallback_suggestion": "Call list_tools() and choose an available tool",
                "details": {"available_tools": list(self._tools.keys())},
            }

        tool = self._tools[name]
        return tool["function"](**arguments)
