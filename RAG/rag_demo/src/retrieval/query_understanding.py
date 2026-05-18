from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI

from ..models import QueryType, QueryUnderstandingOutput


class QueryUnderstanding(Protocol):
    def understand(self, raw_query: str) -> QueryUnderstandingOutput: ...

    @property
    def name(self) -> str: ...


_FILTER_PATTERNS = {
    "document": re.compile(r"\bdoc(?:ument)?\s*:\s*([^\s]+)", re.IGNORECASE),
    "version": re.compile(r"\bversion\s*:\s*(\d+)", re.IGNORECASE),
    "section": re.compile(r"\bsection\s*:\s*\"([^\"]+)\"", re.IGNORECASE),
}


def _detect_query_type(q: str) -> QueryType:
    ql = q.lower()
    if any(w in ql for w in ["summarize", "summary", "tl;dr", "overview"]):
        return QueryType.summarization
    if any(w in ql for w in ["compare", "difference", "vs", "versus"]):
        return QueryType.comparison
    if any(w in ql for w in ["what is", "explain", "define", "how does"]):
        return QueryType.explain_concept
    if any(w in ql for w in ["who", "when", "where", "how many", "which", "what"]):
        return QueryType.factual_lookup
    return QueryType.other


def _normalize(q: str) -> str:
    q = q.strip()
    q = re.sub(r"\s+", " ", q)
    return q


def _expand_terms(q: str) -> List[str]:
    """
    Tiny, teachable query expansion (hand-built synonyms).
    """
    ql = q.lower()
    expansions: List[str] = []
    if "bert" in ql:
        expansions += ["bidirectional encoder representations from transformers", "masked language model", "mlm"]
    if "pretrain" in ql or "pre-training" in ql:
        expansions += ["pre-training", "pretraining", "self-supervised"]
    if "fine-tune" in ql or "finetune" in ql:
        expansions += ["fine-tuning", "transfer learning"]
    return list(dict.fromkeys(expansions))


def _extract_filters(q: str) -> Tuple[Dict, str, List[str]]:
    filters: Dict = {}
    notes: List[str] = []
    remaining = q
    for key, pat in _FILTER_PATTERNS.items():
        m = pat.search(remaining)
        if not m:
            continue
        val = m.group(1)
        filters[key] = val
        remaining = pat.sub("", remaining).strip()
        notes.append(f"Extracted filter {key}={val}")
    return filters, remaining, notes


@dataclass(frozen=True)
class RuleBasedQueryUnderstanding:
    @property
    def name(self) -> str:
        return "rule_based"

    def understand(self, raw_query: str) -> QueryUnderstandingOutput:
        raw_query = raw_query.strip()
        filters, stripped, notes = _extract_filters(raw_query)
        normalized = _normalize(stripped)
        qt = _detect_query_type(normalized)
        expanded = _expand_terms(normalized)
        return QueryUnderstandingOutput(
            raw_query=raw_query,
            normalized_query=normalized,
            expanded_terms=expanded,
            query_type=qt,
            extracted_filters=filters,
            notes=notes,
        )


@dataclass(frozen=True)
class LLMQueryUnderstanding:
    api_key: str
    model: str = "gpt-4.1-mini"

    def __post_init__(self) -> None:
        object.__setattr__(self, "_client", ChatGoogleGenerativeAI(model=self.model, api_key=self.api_key))

    @property
    def name(self) -> str:
        return f"google_rewrite:{self.model}"

    def understand(self, raw_query: str) -> QueryUnderstandingOutput:
        # Keep it very constrained for teaching: ask the model to rewrite a retrieval-friendly query.
        raw_query = raw_query.strip()
        rb = RuleBasedQueryUnderstanding().understand(raw_query)

        prompt = (
            "Rewrite the user's question into a concise retrieval query.\n"
            "Return ONLY the rewritten query, no explanation.\n\n"
            f"User question: {rb.normalized_query}\n"
        )
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            rewritten = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            rewritten = rb.normalized_query
            rb.notes.append(f"LLM rewrite failed; falling back to rule-based. Error: {type(e).__name__}")

        # Merge: keep structured filters/type from deterministic layer; update normalized query.
        return QueryUnderstandingOutput(
            raw_query=rb.raw_query,
            normalized_query=_normalize(rewritten) or rb.normalized_query,
            expanded_terms=rb.expanded_terms,
            query_type=rb.query_type,
            extracted_filters=rb.extracted_filters,
            notes=rb.notes,
        )


def make_query_understanding(google_api_key: Optional[str], model: str) -> QueryUnderstanding:
    if google_api_key:
        return LLMQueryUnderstanding(api_key=google_api_key, model=model)
    return RuleBasedQueryUnderstanding()

