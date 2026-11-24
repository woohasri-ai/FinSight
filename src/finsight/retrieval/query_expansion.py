import inspect
import json
import logging
import re

from finsight.llm.client import LLMClient

logger = logging.getLogger(__name__)


QUERY_EXPANSION_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 2,
        }
    },
    "required": ["queries"],
    "additionalProperties": False,
}

QUERY_REWRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
    },
    "required": ["query"],
    "additionalProperties": False,
}


METRIC_ALIAS_RULES = (
    (
        re.compile(r"\btotal\s+assets?\b", re.IGNORECASE),
        (
            "total assets consolidated balance sheet "
            "consolidated statement of financial position "
            "total assets as at March 31 balance sheet total assets"
        ),
        (
            "Do not substitute individual asset categories, investments, "
            "receivables, or deposits for total assets. Find the overall "
            "Total assets line."
        ),
    ),
    (
        re.compile(r"\btotal\s+liabilit(?:y|ies)\b", re.IGNORECASE),
        (
            "total liabilities consolidated balance sheet "
            "statement of financial position"
        ),
        (
            "Do not substitute borrowings, trade payables, provisions, "
            "or individual liability categories for total liabilities."
        ),
    ),
    (
        re.compile(r"\btotal\s+equity\b", re.IGNORECASE),
        (
            "total equity consolidated balance sheet "
            "statement of changes in equity equity attributable to owners"
        ),
        (
            "Do not substitute share capital, reserves, or individual "
            "equity components for total equity."
        ),
    ),
    (
        re.compile(
            r"\boperating\s+(?:profit\s+)?margin\b|\bebit\s+margin\b",
            re.IGNORECASE,
        ),
        (
            "operating margin operating profit margin EBIT margin "
            "operating margin percentage key financial ratios "
            "consolidated standalone annual report"
        ),
        "Do not substitute gross margin or net profit margin.",
    ),
    (
        re.compile(
            r"\bnet\s+(?:profit|income)\s+margin\b|\bnet\s+margin\b",
            re.IGNORECASE,
        ),
        (
            "net profit margin net margin PAT margin "
            "profit after tax as a percentage of revenue"
        ),
        "The requested metric is a percentage, not an absolute profit amount.",
    ),
    (
        re.compile(
            r"\breturn\s+on\s+equity\b|\bROE\b",
            re.IGNORECASE,
        ),
        (
            "return on equity ROE shareholders equity "
            "key financial ratios percentage"
        ),
        "Do not substitute return on assets or return on capital employed.",
    ),
    (
        re.compile(r"\bcurrent\s+ratio\b", re.IGNORECASE),
        (
            "current ratio current assets current liabilities "
            "key financial ratios times"
        ),
        "Do not substitute quick ratio or working capital.",
    ),
    (
        re.compile(
            r"\bdebt[\s-]+(?:to[\s-]+)?equity\s+ratio\b",
            re.IGNORECASE,
        ),
        (
            "debt equity ratio debt-to-equity ratio "
            "lease liabilities shareholders equity key financial ratios"
        ),
        "Do not substitute total debt or interest coverage ratio.",
    ),
    (
        re.compile(r"\brevenue\s+growth\b", re.IGNORECASE),
        (
            "revenue growth percentage year on year growth "
            "consolidated revenue growth"
        ),
        "Do not substitute absolute revenue or profit growth.",
    ),
    (
        re.compile(
            r"\bnet\s+income\b|\bnet\s+profit\b|"
            r"\bprofit\s+after\s+tax\b|\bPAT\b",
            re.IGNORECASE,
        ),
        (
            "net income net profit profit after tax PAT "
            "profit for the year statement of profit and loss "
            "income statement"
        ),
        (
            "Do not substitute operating cash flow, investing cash flow, "
            "financing cash flow, cash balances, revenue, or investments."
        ),
    ),
    (
        re.compile(
            r"\bprofit\s+before\s+tax\b|\bPBT\b",
            re.IGNORECASE,
        ),
        (
            "profit before tax PBT earnings before tax "
            "statement of profit and loss"
        ),
        "Do not substitute profit after tax or cash flow for PBT.",
    ),
    (
        re.compile(
            r"\boperating\s+profit\b|\bEBIT\b",
            re.IGNORECASE,
        ),
        (
            "operating profit EBIT earnings before interest and tax "
            "statement of profit and loss"
        ),
        "Do not substitute net profit, EBITDA, or operating cash flow.",
    ),
    (
        re.compile(r"\bEBITDA\b", re.IGNORECASE),
        (
            "EBITDA earnings before interest tax depreciation "
            "and amortisation EBITDA reconciliation"
        ),
        "Do not substitute EBIT, net profit, or operating cash flow.",
    ),
    (
        re.compile(
            r"\bEPS\b|\bearnings\s+per\s+share\b",
            re.IGNORECASE,
        ),
        (
            "earnings per share EPS basic earnings per share "
            "diluted earnings per share"
        ),
        "Do not substitute total net income or share price for EPS.",
    ),
    (
        re.compile(
            r"\brevenue\b|\bsales\b|\bturnover\b",
            re.IGNORECASE,
        ),
        (
            "revenue from operations revenue sales turnover "
            "statement of profit and loss"
        ),
        "Do not substitute cash receipts or cash flow for revenue.",
    ),
)


def _get_metric_guidance(
    question: str,
) -> tuple[str, str] | None:
    """Return accounting aliases for a recognized financial metric."""

    for pattern, aliases, exclusions in METRIC_ALIAS_RULES:
        if pattern.search(question):
            return aliases, exclusions

    return None


def _build_fallback_queries(question: str) -> list[str]:
    """Build deterministic metric-aware queries."""

    normalized_question = question.strip()
    guidance = _get_metric_guidance(normalized_question)

    if guidance is None:
        return [normalized_question]

    aliases, _ = guidance

    return [
        normalized_question,
        f"{normalized_question} {aliases}",
    ]


def _append_unique(
    queries: list[str],
    candidate: str,
) -> None:
    """Append a nonblank query without case-insensitive duplicates."""

    normalized = candidate.strip()

    if not normalized:
        return

    existing = {
        query.casefold()
        for query in queries
    }

    if normalized.casefold() not in existing:
        queries.append(normalized)


def _format_metric_guidance(question: str) -> str:
    """Build metric-specific instructions for the LLM."""

    guidance = _get_metric_guidance(question)

    if guidance is None:
        return ""

    aliases, exclusions = guidance

    return (
        "\nMetric-specific retrieval guidance:\n"
        f"- Search using these accounting labels: {aliases}.\n"
        f"- {exclusions}\n"
        "- Preserve the exact requested metric and reporting scope.\n"
    )


async def expand_retrieval_queries(
    question: str,
    llm: LLMClient,
) -> list[str]:
    """
    Generate retrieval alternatives.

    Synchronous/mock LLM interfaces retain the original single-query
    behavior. Real asynchronous LLM interfaces use deterministic aliases
    for recognized financial metrics without making an LLM expansion call.

    General questions retain LLM-based expansion.
    """

    if not question.strip():
        raise ValueError("Question cannot be blank.")

    normalized_question = question.strip()

    # Preserve compatibility with synchronous mocks and clients.
    generate_method = getattr(llm, "generate", None)

    if not inspect.iscoroutinefunction(generate_method):
        return [normalized_question]

    # Avoid an LLM expansion call for recognized financial metrics.
    if _get_metric_guidance(normalized_question) is not None:
        fallback_queries = _build_fallback_queries(
            normalized_question
        )

        logger.info(
            "FinSight deterministic query expansion | "
            "LLM skipped | query count: %d",
            len(fallback_queries),
        )

        return fallback_queries

    prompt = (
        "Generate exactly two alternative search queries for the question "
        "below. Preserve its original intent, company names, financial "
        "metrics, fiscal years, and reporting scope.\n\n"
        "Use relevant accounting terminology and financial statement labels. "
        "Do not broaden the question into unrelated financial topics. "
        "Do not answer the question or invent facts.\n"
        f"Question: {normalized_question}"
    )

    try:
        response = await llm.generate(
            prompt,
            system=(
                "You are a precise financial-document search assistant. "
                "Return only the requested JSON object."
            ),
            response_format=QUERY_EXPANSION_SCHEMA,
        )

        parsed = json.loads(response)

        if not isinstance(parsed, dict):
            return [normalized_question]

        alternatives = parsed.get("queries")

        if not isinstance(alternatives, list):
            return [normalized_question]

        queries = [normalized_question]

        for candidate in alternatives:
            if isinstance(candidate, str):
                _append_unique(queries, candidate)

        return queries[:3]

    except Exception:
        logger.warning(
            "Query expansion failed; using the original question.",
            exc_info=True,
        )
        return [normalized_question]


async def rewrite_retrieval_query(
    question: str,
    llm: LLMClient,
) -> str:
    """Rewrite general questions without changing their original intent."""

    if not question.strip():
        raise ValueError("Question cannot be blank.")

    normalized_question = question.strip()

    # Keep recognized financial questions unchanged.
    if _get_metric_guidance(normalized_question) is not None:
        logger.info(
            "FinSight metric query rewrite skipped | "
            "original query retained."
        )
        return normalized_question

    generate_method = getattr(llm, "generate", None)

    if not inspect.iscoroutinefunction(generate_method):
        return normalized_question

    prompt = (
        "Rewrite the question below as exactly one focused document-search "
        "query. Preserve its complete intent, company names, financial "
        "metrics, fiscal years, and reporting scope.\n"
        "Do not invent facts or broaden the question.\n"
        "Return only the requested JSON object.\n\n"
        f"Original question: {normalized_question}"
    )

    try:
        response = await llm.generate(
            prompt,
            system=(
                "You are a precise financial-document query rewriting "
                "assistant. Return one concise, faithful retrieval query "
                "as JSON."
            ),
            response_format=QUERY_REWRITE_SCHEMA,
        )

        parsed = json.loads(response)

        if not isinstance(parsed, dict):
            return normalized_question

        rewritten = parsed.get("query")

        if not isinstance(rewritten, str) or not rewritten.strip():
            return normalized_question

        rewritten = rewritten.strip()

        if rewritten.casefold() == normalized_question.casefold():
            return normalized_question

        return rewritten

    except Exception:
        logger.warning(
            "Query rewriting failed; using the original question.",
            exc_info=True,
        )
        return normalized_question