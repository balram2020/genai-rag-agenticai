"""In-memory fake data so the live class never depends on the network."""

POLICY_DOCS = [
    {
        "department": "hr",
        "title": "Leave Policy",
        "content": "Employees get 20 paid leaves per year. Leave requests must be submitted in advance.",
    },
    {
        "department": "finance",
        "title": "Travel Reimbursement",
        "content": "Travel claims must be submitted within 15 days with valid receipts.",
    },
    {
        "department": "engineering",
        "title": "Production Access",
        "content": "Production access requires manager approval, audit logging, and quarterly review.",
    },
]


TICKETS = {
    "TKT-1001": {
        "ticket_id": "TKT-1001",
        "status": "open",
        "priority": "medium",
        "summary": "Need help with production access",
    }
}
