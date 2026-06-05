"""Run a sequence of sources until one succeeds, with structured failure."""
from typing import Callable, List

from shared.error_contracts import error_response


class FallbackChain:
    def __init__(self, name: str, sources: List[Callable]):
        self.name = name
        self.sources = sources

    def execute(self, **kwargs) -> dict:
        errors = []

        for index, source in enumerate(self.sources):
            result = source(**kwargs)

            if result.get("success"):
                if index > 0:
                    result["metadata"]["fallback_used"] = source.__name__
                    result["metadata"]["quality_note"] = (
                        "Result came from fallback source"
                    )
                return result

            errors.append(result)

        return error_response(
            error_type="all_sources_failed",
            message=f"All sources failed for fallback chain: {self.name}",
            recoverable=True,
            fallback_suggestion=(
                "Return partial response or ask user for clarification"
            ),
            details={"errors": errors},
        )
