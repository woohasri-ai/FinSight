import re
from collections.abc import Sequence

LEGAL_SUFFIXES = {
    "limited",
    "ltd",
    "incorporated",
    "inc",
    "corporation",
    "corp",
    "company",
    "co",
}


def normalize_company_text(value: str) -> str:
    """Normalize company names and questions for matching."""
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def company_aliases(company_name: str) -> set[str]:
    """Build matching aliases from a company's metadata name."""
    normalized = normalize_company_text(company_name)

    if not normalized:
        return set()

    aliases = {normalized}
    words = normalized.split()

    while words and words[-1] in LEGAL_SUFFIXES:
        words.pop()

    if words:
        aliases.add(" ".join(words))

    return {alias for alias in aliases if len(alias) >= 2}


def resolve_company_symbols(
    question: str,
    chunks: Sequence[dict[str, object]],
) -> set[str] | None:
    """
    Resolve explicitly named companies using available chunk metadata.

    Returns a set of matching symbols, or None when no available company
    is recognized, preserving unrestricted retrieval behavior.
    """
    normalized_question = normalize_company_text(question)

    if not normalized_question:
        return None

    question_words = set(normalized_question.split())
    companies: dict[str, set[str]] = {}

    for chunk in chunks:
        symbol = str(chunk.get("symbol", "")).strip().upper()
        company_name = str(chunk.get("company_name", "")).strip()

        if not symbol:
            continue

        aliases = company_aliases(company_name)
        aliases.add(symbol.lower())

        companies.setdefault(symbol, set()).update(aliases)

    matched_symbols = set()

    for symbol, aliases in companies.items():
        for alias in aliases:
            if " " in alias:
                if re.search(
                    rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])",
                    normalized_question,
                ):
                    matched_symbols.add(symbol)
                    break
            elif alias in question_words:
                matched_symbols.add(symbol)
                break

    return matched_symbols or None
