"""MCP-style prompts: reusable prompt templates exposed to clients."""

PROMPTS = {
    "summarize_policy_answer": {
        "name": "summarize_policy_answer",
        "description": (
            "Summarize a policy document into a concise, employee-friendly answer."
        ),
        "template": (
            "You are an internal HR assistant.\n"
            "Given the following policy document, summarize it in 3 bullet points.\n\n"
            "Document title: {title}\n"
            "Document content: {content}\n"
        ),
        "input_variables": ["title", "content"],
    },
    "explain_ticket_status": {
        "name": "explain_ticket_status",
        "description": "Explain a ticket's current status to an end user.",
        "template": (
            "Ticket {ticket_id} is currently '{status}' with priority '{priority}'.\n"
            "Issue summary: {summary}\n"
            "Explain in two sentences what the user should expect next."
        ),
        "input_variables": ["ticket_id", "status", "priority", "summary"],
    },
}


def list_prompts():
    return [
        {"name": p["name"], "description": p["description"]}
        for p in PROMPTS.values()
    ]


def get_prompt(name: str, arguments: dict):
    if name not in PROMPTS:
        return {
            "success": False,
            "error_type": "no_results",
            "message": f"Prompt '{name}' not found",
        }
    template = PROMPTS[name]["template"]
    return {
        "success": True,
        "data": {"rendered": template.format(**arguments)},
        "metadata": {"source": "prompt_registry"},
    }
