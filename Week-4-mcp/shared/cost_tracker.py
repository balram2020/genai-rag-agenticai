"""Per-run budget tracker. Every autonomous loop is also a spending loop."""


class CostTracker:
    def __init__(self, budget_usd: float = 0.01):
        self.budget_usd = budget_usd
        self.total_spent = 0.0
        self.cost_by_tool = {}

    def can_afford(self, tool_name: str, estimated_cost: float):
        return self.total_spent + estimated_cost <= self.budget_usd

    def record(self, tool_name: str, cost_usd: float):
        self.total_spent += cost_usd
        self.cost_by_tool[tool_name] = (
            self.cost_by_tool.get(tool_name, 0.0) + cost_usd
        )

    def summary(self):
        return {
            "budget_usd": self.budget_usd,
            "total_spent": round(self.total_spent, 6),
            "remaining": round(self.budget_usd - self.total_spent, 6),
            "cost_by_tool": self.cost_by_tool,
        }
