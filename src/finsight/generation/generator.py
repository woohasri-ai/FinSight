
import json
import logging
import re
from copy import deepcopy
from decimal import Decimal
from typing import Any

from pydantic import ValidationError

from finsight.generation.financial_metrics import (
    _format_indian_integer,
    extract_financial_metric,
)
from finsight.generation.prompts import SYSTEM_PROMPT, build_generation_prompt
from finsight.generation.schemas import (
    Citation,
    EvidenceExcerpt,
    GeneratedAnswer,
    LLMGenerationResponse,
)
from finsight.llm.client import LLMClient


logger = logging.getLogger(__name__)


class GenerationError(RuntimeError):
    """Raised when the LLM response cannot be safely used."""


def handle_empty_evidence() -> GeneratedAnswer:
    """Return a deterministic abstention when retrieval provides no evidence."""
    return GeneratedAnswer(
        answer=(
            "The supplied evidence does not provide sufficient information "
            "to answer this question."
        ),
        citations=[],
        abstained=True,
    )


def build_evidence_map(
    evidence: list[dict[str, Any]],
    top_k: int = 5,
) -> dict[str, dict[str, Any]]:
    """Validate retrieved chunks and assign prompt-local evidence IDs."""

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    if not evidence:
        raise ValueError("Evidence cannot be empty")

    evidence_map: dict[str, dict[str, Any]] = {}
    required_fields = ("source_doc", "page_number", "chunk_id", "text")

    for index, chunk in enumerate(evidence[:top_k], start=1):
        if not isinstance(chunk, dict):
            raise ValueError("Each evidence item must be a dictionary")

        for field in required_fields:
            if field not in chunk:
                raise ValueError(f"Evidence item is missing '{field}'")

        for field in ("source_doc", "chunk_id", "text"):
            value = chunk[field]

            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Evidence field '{field}' must be a non-blank string"
                )

        page_number = chunk["page_number"]

        if (
            isinstance(page_number, bool)
            or not isinstance(page_number, int)
            or page_number <= 0
        ):
            raise ValueError("Evidence page_number must be a positive integer")

        evidence_map[f"S{index}"] = chunk

    if not evidence_map:
        raise ValueError("Evidence cannot be empty")

    return evidence_map


def build_response_schema(
    evidence_ids: list[str],
) -> dict[str, Any]:
    """Restrict model evidence references to retrieved evidence IDs."""

    if not evidence_ids:
        raise ValueError("evidence_ids cannot be empty")

    if any(not isinstance(item, str) or not item.strip() for item in evidence_ids):
        raise ValueError("evidence_ids must contain non-blank strings")

    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError("evidence_ids must be unique")

    schema = deepcopy(LLMGenerationResponse.model_json_schema())

    required_fields = schema.setdefault("required", [])

    if "excerpts" not in required_fields:
        required_fields.append("excerpts")

    evidence_id_schema = schema["$defs"]["EvidenceExcerpt"]["properties"][
        "evidence_id"
    ]
    evidence_id_schema["enum"] = evidence_ids

    return schema


def parse_llm_response(raw_response: str) -> LLMGenerationResponse:
    """Parse response and remove repeated references before schema validation."""

    try:
        payload = json.loads(raw_response)

        if isinstance(payload, dict) and isinstance(payload.get("excerpts"), list):
            unique_excerpts = []
            seen_ids = set()

            for excerpt in payload["excerpts"]:
                if not isinstance(excerpt, dict):
                    unique_excerpts.append(excerpt)
                    continue

                evidence_id = excerpt.get("evidence_id")

                if isinstance(evidence_id, str) and evidence_id in seen_ids:
                    logger.warning(
                        "Ignoring duplicate evidence reference returned by LLM: %s",
                        evidence_id,
                    )
                    continue

                if isinstance(evidence_id, str):
                    seen_ids.add(evidence_id)

                unique_excerpts.append(excerpt)

            payload["excerpts"] = unique_excerpts

        return LLMGenerationResponse.model_validate(payload)

    except (json.JSONDecodeError, TypeError, ValidationError) as exc:
        raise GenerationError(
            "The LLM returned an invalid generation response."
        ) from exc


def validate_evidence_excerpts(
    response: LLMGenerationResponse,
    evidence_map: dict[str, dict[str, Any]],
) -> None:
    """Verify that every returned evidence ID belongs to retrieved evidence."""

    for excerpt in response.excerpts:
        if excerpt.evidence_id not in evidence_map:
            logger.error(
                "Unknown evidence reference returned by LLM: %s",
                excerpt.evidence_id,
            )
            raise GenerationError(
                f"The LLM returned unknown evidence ID: {excerpt.evidence_id}"
            )


def build_citations(
    response: LLMGenerationResponse,
    evidence_map: dict[str, dict[str, Any]],
) -> list[Citation]:
    """Build one citation per distinct evidence chunk used by the response."""

    citations: list[Citation] = []
    used_evidence_ids: set[str] = set()

    for excerpt in response.excerpts:
        evidence_id = excerpt.evidence_id

        if evidence_id in used_evidence_ids:
            continue

        chunk = evidence_map[evidence_id]

        citations.append(
            Citation(
                document_name=chunk["source_doc"],
                page_number=chunk["page_number"],
                chunk_id=chunk["chunk_id"],
                evidence_text=chunk["text"],
            )
        )

        used_evidence_ids.add(evidence_id)

    return citations


def build_answer_from_excerpts(
    response: LLMGenerationResponse,
) -> str:
    """Retain the legacy helper using the model-generated answer."""
    return response.answer


def _is_internal_evidence_id_answer(answer: str) -> bool:
    """Detect answers that expose an internal evidence ID instead of content."""
    return bool(
        isinstance(answer, str)
        and re.fullmatch(r"\s*S\d+\s*[.!?]?\s*", answer, re.IGNORECASE)
    )


def _is_fy2026_total_assets_question(question: str) -> bool:
    """Identify FY2026 total-assets questions requiring exact row evidence."""

    if not isinstance(question, str):
        return False

    asks_total_assets = bool(
        re.search(r"\btotal\s+assets?\b", question, re.I)
    )

    asks_fy2026 = bool(
        re.search(
            r"\b(?:"
            r"(?:fy\s*)?2025\s*[-–—/]\s*(?:26|2026)"
            r"|(?:fy\s*)?2026"
            r")\b",
            question,
            re.I,
        )
    )

    asks_ratio = bool(
        re.search(
            r"\b(?:percentage|percent|ratio|proportion)\b|%",
            question,
            re.I,
        )
    )

    return asks_total_assets and asks_fy2026 and not asks_ratio


def _is_year_scoped_net_income_question(question: str) -> bool:
    """Identify net-income questions that require verified year columns."""

    if not isinstance(question, str):
        return False

    asks_metric = bool(
        re.search(
            r"\b(?:net\s+income|net\s+profit|profit\s+for\s+the\s+year)\b",
            question,
            re.I,
        )
    )

    asks_year = bool(
        re.search(
            r"\b(?:"
            r"(?:fy\s*)?20\d{2}\s*[-–—/]\s*(?:\d{2}|20\d{2})"
            r"|fy\s*20\d{2}"
            r")\b",
            question,
            re.I,
        )
    )

    return asks_metric and asks_year


def extract_fy2026_total_assets_excerpt(
    question: str,
    evidence_map: dict[str, dict[str, Any]],
) -> tuple[str, str, str] | None:
    """Extract an explicit FY2026 total-assets balance-sheet row."""

    if not _is_fy2026_total_assets_question(question):
        return None

    requested_basis = (
        "standalone"
        if re.search(r"\bstandalone\b", question, re.I)
        else "consolidated"
    )

    heading_pattern = re.compile(
        r"\b(consolidated|standalone)\s+balance\s+sheet\b",
        re.I,
    )
    total_assets_row_pattern = re.compile(
        r"^\s*TOTAL\s+ASSETS\s*(?P<tail>.*)$",
        re.I,
    )
    numeric_pattern = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")

    for evidence_id, chunk in evidence_map.items():
        text = chunk.get("text", "")

        if not isinstance(text, str) or not text.strip():
            continue

        chunk_year = str(chunk.get("fiscal_year", "")).strip()

        if chunk_year and chunk_year not in {
            "2025-26",
            "2025–26",
            "FY2025-26",
        }:
            continue

        headings = list(heading_pattern.finditer(text))

        if not any(
            heading.group(1).casefold() == requested_basis
            for heading in headings
        ):
            continue

        if not re.search(r"\b2026\b", text):
            continue

        if not re.search(r"\b2025\b", text):
            continue

        if re.search(
            r"statement\s+of\s+segment\s+assets\s+and\s+liabilities"
            r"|segment\s+assets\s+and\s+liabilities",
            text,
            re.I,
        ):
            continue

        lines = text.splitlines()
        position = 0
        line_positions: list[int] = []

        for line in lines:
            line_positions.append(position)
            position += len(line) + 1

        for line_index, line in enumerate(lines):
            row_match = total_assets_row_pattern.fullmatch(line)

            if not row_match:
                continue

            row_position = line_positions[line_index]

            matching_heading = any(
                heading.group(1).casefold() == requested_basis
                and abs(heading.start() - row_position) <= 5000
                for heading in headings
            )

            if not matching_heading:
                continue

            values = numeric_pattern.findall(row_match.group("tail") or "")

            if not values:
                for following_line in lines[line_index + 1:line_index + 5]:
                    candidate = following_line.strip()

                    if not candidate:
                        continue

                    if not re.fullmatch(
                        r"[\s₹$€£(),.\-*–—\d]+",
                        candidate,
                    ):
                        break

                    candidate_values = numeric_pattern.findall(candidate)

                    if candidate_values:
                        values.extend(candidate_values)
                        break

            if not values:
                continue

            formatted_value = _format_indian_integer(
                Decimal(values[0].replace(",", ""))
            )

            unit_match = re.search(
                r"\b(?:all\s+amounts|amounts|figures)\s+in\s+"
                r"(crores?|millions?|billions?|lakhs?)\b"
                r"|\bin\s+(crores?|millions?|billions?|lakhs?)\b",
                text,
                re.I,
            )

            unit = ""

            if unit_match:
                unit = next(
                    (group for group in unit_match.groups() if group),
                    "",
                ).casefold()

                if unit.endswith("s"):
                    unit = unit[:-1]

            currency = "₹" if (
                "₹" in text
                or re.search(r"\bINR\b|\bRs\.?", text, re.I)
            ) else ""

            display_value = (
                f"{currency}{formatted_value} {unit}".strip()
                if unit
                else f"{currency}{formatted_value}"
            )

            answer = (
                f"The {requested_basis} total assets for FY 2025–26 "
                f"were {display_value}."
            )

            logger.info(
                "FinSight deterministic metric extraction | "
                "metric: %s total assets | value: %s | evidence: %s",
                requested_basis,
                display_value,
                evidence_id,
            )

            return evidence_id, answer, line.strip()

    return None


def extract_fy2026_profit_table_excerpt(
    question: str,
    evidence_map: dict[str, dict[str, Any]],
) -> tuple[str, str] | None:
    """Extract a clearly labelled FY2026 profit row from a financial table."""

    question_lower = question.lower()

    profit_question = bool(
        re.search(
            r"\b(net profit|profit after tax|pat|profit attributable)\b",
            question_lower,
        )
    )

    asks_fy2026 = bool(
        re.search(
            r"\b(?:"
            r"fy\s*2026"
            r"|2026"
            r"|fy\s*2025\s*[-–—/]\s*(?:26|2026)"
            r")\b",
            question_lower,
            re.I,
        )
    )

    if not profit_question or not asks_fy2026:
        return None

    for evidence_id, chunk in evidence_map.items():
        text = chunk["text"]

        lines = text.splitlines(keepends=True)
        offsets = []
        position = 0

        for line in lines:
            offsets.append(position)
            position += len(line)

        matching_rows = [
            index
            for index, line in enumerate(lines)
            if re.search(
                r"\bprofit\s+after\s+tax\b|\bnet\s+profit\b",
                line,
                re.I,
            )
        ]

        if len(matching_rows) != 1:
            continue

        row_index = matching_rows[0]
        row_start = offsets[row_index]
        preceding_text = text[:row_start]

        table_heading = re.search(
            r"The following table gives an overview of the consolidated "
            r"financial results of the Company:",
            preceding_text,
            re.I,
        )

        if not table_heading:
            continue

        table_text = preceding_text[table_heading.start():]

        if not re.search(r"\bFY\s*2026\b", table_text, re.I):
            continue

        if not re.search(r"\bFY\s*2025\b", table_text, re.I):
            continue

        unit_match = re.search(
            r"\(\s*H\s+crore\s*\)",
            table_text,
            re.I,
        )

        if not unit_match:
            continue

        excerpt_start = table_heading.start() + unit_match.start()
        row_end_index = row_index + 1

        while row_end_index < len(lines):
            candidate = lines[row_end_index].strip()

            if not candidate:
                row_end_index += 1
                continue

            if not re.fullmatch(r"[\d,.\-%\s]+", candidate):
                break

            row_end_index += 1

        if row_end_index == row_index + 1:
            continue

        excerpt_end = (
            offsets[row_end_index]
            if row_end_index < len(lines)
            else len(text)
        )

        exact_excerpt = text[excerpt_start:excerpt_end].strip()

        if exact_excerpt and exact_excerpt in text:
            return evidence_id, exact_excerpt

    return None


def extract_fy2026_operating_margin_excerpt(
    question: str,
    evidence_map: dict[str, dict[str, Any]],
) -> tuple[str, str, str] | None:
    """Extract an explicitly reported consolidated FY2026 EBIT margin."""

    if not isinstance(question, str):
        return None

    asks_operating_margin = bool(
        re.search(r"\b(?:operating|ebit)\s+margin\b", question, re.I)
    )

    asks_fy2026 = bool(
        re.search(
            r"\b(?:"
            r"(?:fy\s*)?2025\s*[-–—/]\s*(?:26|2026)"
            r"|(?:fy\s*)?2026"
            r")\b",
            question,
            re.I,
        )
    )

    asks_standalone = bool(re.search(r"\bstandalone\b", question, re.I))

    if not asks_operating_margin or not asks_fy2026 or asks_standalone:
        return None

    margin_pattern = re.compile(
        r"(?P<sentence>"
        r"(?:the\s+)?(?:resultant\s+)?"
        r"EBIT\s+Margin\s+for\s+the\s+financial\s+year\s+"
        r"2025\s*[-–—/]\s*26\s+stood\s+at\s+"
        r"(?P<value>\d+(?:\.\d+)?)\s*%"
        r"[^.]*\.?"
        r")",
        re.I,
    )

    for evidence_id, chunk in evidence_map.items():
        text = chunk.get("text", "")

        if not isinstance(text, str) or not text.strip():
            continue

        normalized_text = " ".join(text.split())

        heading = re.search(
            r"\bConsolidated\s+Financial\s+Performance\b",
            normalized_text,
            re.I,
        )

        if not heading:
            continue

        match = margin_pattern.search(normalized_text[heading.end():])

        if not match:
            continue

        value = match.group("value")
        exact_excerpt = match.group("sentence").strip()

        logger.info(
            "FinSight deterministic metric extraction | "
            "metric: consolidated operating margin | value: %s%% | "
            "evidence: %s",
            value,
            evidence_id,
        )

        return evidence_id, exact_excerpt, value

    return None


def build_evidence_selection_prompt(
    question: str,
    answer: str,
    evidence_map: dict[str, dict[str, Any]],
) -> str:
    """Ask the model to identify evidence supporting an existing answer."""

    evidence_blocks = []

    for evidence_id, chunk in evidence_map.items():
        evidence_blocks.append(
            f"[{evidence_id}]\n"
            f"Source: {chunk['source_doc']}\n"
            f"Page: {chunk['page_number']}\n"
            f"Text: {chunk['text']}"
        )

    return (
        "Your task is to identify which supplied evidence IDs genuinely "
        "support the existing answer.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"EXISTING ANSWER:\n{answer}\n\n"
        "AVAILABLE EVIDENCE:\n"
        + "\n\n".join(evidence_blocks)
        + "\n\n"
        "RULES:\n"
        "- Do not rewrite or improve the existing answer.\n"
        "- Select only evidence that directly supports the answer.\n"
        "- Select at least one ID if the evidence supports the answer.\n"
        "- If none of the evidence supports the answer, set abstained=true "
        "and return an empty excerpts array.\n"
        "- Use only the supplied evidence IDs.\n"
        "- Return the required JSON structure exactly."
    )


async def recover_evidence_references(
    question: str,
    answer: str,
    evidence_map: dict[str, dict[str, Any]],
    llm: LLMClient,
) -> LLMGenerationResponse:
    """Make one focused attempt to recover missing evidence references."""

    prompt = build_evidence_selection_prompt(question, answer, evidence_map)
    schema = build_response_schema(list(evidence_map.keys()))

    raw_response = await llm.generate(
        prompt,
        system=(
            "You are an evidence-selection component for FinSight. "
            "Select only supplied evidence IDs that genuinely support "
            "the existing answer. Never invent references."
        ),
        response_format=schema,
    )

    response = parse_llm_response(raw_response)

    if response.abstained:
        raise GenerationError(
            "The answer could not be matched to supporting evidence."
        )

    if not response.excerpts:
        raise GenerationError(
            "A non-abstained response must contain evidence excerpts."
        )

    validate_evidence_excerpts(response, evidence_map)

    return response


async def generate_answer(
    question: str,
    evidence: list[dict[str, Any]],
    llm: LLMClient,
    top_k: int = 5,
) -> GeneratedAnswer:
    """Generate a natural-language answer using retrieved source evidence."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-blank string")

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    if not evidence:
        return handle_empty_evidence()

    evidence_map = build_evidence_map(evidence, top_k=top_k)

    # First: exact balance-sheet row. Never let the LLM substitute net assets.
    total_assets_result = extract_fy2026_total_assets_excerpt(
        question,
        evidence_map,
    )

    if total_assets_result is not None:
        evidence_id, exact_answer, row_label = total_assets_result

        response = LLMGenerationResponse(
            answer=exact_answer,
            excerpts=[EvidenceExcerpt(evidence_id=evidence_id)],
            abstained=False,
        )

        validate_evidence_excerpts(response, evidence_map)

        logger.info(
            "FinSight answer generated without LLM | "
            "metric: total assets | row: %s | evidence: %s",
            row_label,
            evidence_id,
        )

        return GeneratedAnswer(
            answer=response.answer,
            citations=build_citations(response, evidence_map),
            abstained=False,
        )

    if _is_fy2026_total_assets_question(question):
        logger.warning(
            "FinSight total-assets abstention | "
            "No valid TOTAL ASSETS row in supplied evidence."
        )
        return handle_empty_evidence()

    # Second: preserve the existing TCS consolidated PAT fast path.
    # It handles the explicit "Profit after tax attributable..." table,
    # which is different from Wipro's "Profit for the year" statement row.
    table_result = extract_fy2026_profit_table_excerpt(
        question,
        evidence_map,
    )

    if table_result is not None:
        evidence_id, exact_excerpt = table_result

        response = LLMGenerationResponse(
            answer=(
                "The FY2026 consolidated profit after tax information "
                "reported in the supplied financial results table is:\n\n"
                f"{exact_excerpt}"
            ),
            excerpts=[EvidenceExcerpt(evidence_id=evidence_id)],
            abstained=False,
        )

        validate_evidence_excerpts(response, evidence_map)

        return GeneratedAnswer(
            answer=response.answer,
            citations=build_citations(response, evidence_map),
            abstained=False,
        )

    # Third: generic year-aware statement-row extraction.
    financial_metric_result = extract_financial_metric(
        question,
        evidence_map,
    )

    if financial_metric_result is not None:
        evidence_id = financial_metric_result.evidence_id

        if evidence_id not in evidence_map:
            raise GenerationError(
                "Financial metric extractor returned unknown evidence ID."
            )

        response = LLMGenerationResponse(
            answer=financial_metric_result.answer,
            excerpts=[EvidenceExcerpt(evidence_id=evidence_id)],
            abstained=False,
        )

        validate_evidence_excerpts(response, evidence_map)

        logger.info(
            "FinSight deterministic metric answer | "
            "metric: %s | year: %s | basis: %s | value: %s | "
            "evidence: %s",
            financial_metric_result.metric_name,
            financial_metric_result.fiscal_year_end,
            financial_metric_result.reporting_basis,
            financial_metric_result.display_value,
            evidence_id,
        )

        return GeneratedAnswer(
            answer=response.answer,
            citations=build_citations(response, evidence_map),
            abstained=False,
        )

    # Only now fail closed if an exact, year-scoped net-income row is absent.
    if _is_year_scoped_net_income_question(question):
        logger.warning(
            "FinSight net-income abstention | "
            "No verified metric row for requested year and reporting basis."
        )
        return handle_empty_evidence()

    # Fourth: existing consolidated operating-margin extraction.
    margin_result = extract_fy2026_operating_margin_excerpt(
        question,
        evidence_map,
    )

    if margin_result is not None:
        evidence_id, exact_excerpt, value = margin_result

        response = LLMGenerationResponse(
            answer=(
                "The consolidated operating margin, reported as EBIT "
                f"margin, for FY 2025–26 was {value}%."
            ),
            excerpts=[EvidenceExcerpt(evidence_id=evidence_id)],
            abstained=False,
        )

        validate_evidence_excerpts(response, evidence_map)

        return GeneratedAnswer(
            answer=response.answer,
            citations=build_citations(response, evidence_map),
            abstained=False,
        )

    # Other questions continue to use the standard evidence-grounded LLM.
    prompt = build_generation_prompt(
        question,
        list(evidence_map.values()),
        top_k=top_k,
    )

    response_schema = build_response_schema(list(evidence_map.keys()))

    raw_response = await llm.generate(
        prompt,
        system=SYSTEM_PROMPT,
        response_format=response_schema,
    )

    parsed_response = parse_llm_response(raw_response)

    # Reject a response that exposes only an internal evidence label.
    if _is_internal_evidence_id_answer(parsed_response.answer):
        logger.warning(
            "FinSight rejected an answer containing only an internal "
            "evidence ID: %s",
            parsed_response.answer.strip(),
        )
        return handle_empty_evidence()

    if parsed_response.abstained:
        if parsed_response.excerpts:
            raise GenerationError(
                "An abstained response cannot contain evidence excerpts."
            )

        return handle_empty_evidence()

    if not parsed_response.excerpts:
        logger.warning(
            "LLM generated an answer without evidence IDs. "
            "Attempting focused evidence selection."
        )

        recovered_response = await recover_evidence_references(
            question,
            parsed_response.answer,
            evidence_map,
            llm,
        )

        parsed_response = LLMGenerationResponse(
            answer=parsed_response.answer,
            excerpts=recovered_response.excerpts,
            abstained=False,
        )

    validate_evidence_excerpts(parsed_response, evidence_map)

    citations = build_citations(parsed_response, evidence_map)

    if not citations:
        raise GenerationError(
            "The generated answer has no validated supporting citations."
        )

    return GeneratedAnswer(
        answer=parsed_response.answer,
        citations=citations,
        abstained=False,
    )
