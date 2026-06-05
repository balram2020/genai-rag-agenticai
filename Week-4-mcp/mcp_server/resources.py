"""MCP-style resources: read-only context the model can consult.

Resources are NOT actions. They are data surfaces — policy docs,
schemas, runbooks — that the model can fetch by URI.
"""
from shared.fake_data import POLICY_DOCS

RESOURCES = {
    "policy://leave-policy": next(
        d for d in POLICY_DOCS if d["title"] == "Leave Policy"
    ),
    "policy://travel-reimbursement": next(
        d for d in POLICY_DOCS if d["title"] == "Travel Reimbursement"
    ),
    "policy://production-access": next(
        d for d in POLICY_DOCS if d["title"] == "Production Access"
    ),
}


def list_resources():
    return [{"uri": uri, "title": doc["title"]} for uri, doc in RESOURCES.items()]


def read_resource(uri: str):
    if uri not in RESOURCES:
        return {
            "success": False,
            "error_type": "no_results",
            "message": f"Resource '{uri}' not found",
        }
    return {
        "success": True,
        "data": RESOURCES[uri],
        "metadata": {"source": "policy_resources"},
    }
