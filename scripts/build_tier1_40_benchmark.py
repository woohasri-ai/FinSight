import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TCS_PATH = ROOT / "data" / "eval" / "retrieval_golden_set.json"
ADDITIONAL_PATH = ROOT / "data" / "eval" / "tier1_additional_24_draft.json"
CHUNKS_PATH = (
    ROOT / "data" / "processed" / "multi_company"
    / "tier1_combined_chunks.json"
)
OUTPUT_PATH = ROOT / "data" / "eval" / "tier1_retrieval_golden_set_40.json"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    tcs_questions = load_json(TCS_PATH)
    additional_questions = load_json(ADDITIONAL_PATH)
    chunks = load_json(CHUNKS_PATH)

    if len(tcs_questions) != 16:
        raise SystemExit(
            f"Expected 16 original TCS questions, found {len(tcs_questions)}."
        )

    if len(additional_questions) != 24:
        raise SystemExit(
            f"Expected 24 additional questions, found {len(additional_questions)}."
        )

    combined = tcs_questions + additional_questions

    ids = [question["id"] for question in combined]
    if len(ids) != len(set(ids)):
        raise SystemExit("Duplicate question IDs detected.")

    chunk_map = {chunk["chunk_id"]: chunk for chunk in chunks}
    errors = []

    for question in combined:
        for chunk_id in question["relevant_chunk_ids"]:
            chunk = chunk_map.get(chunk_id)

            if chunk is None:
                errors.append(
                    f"{question['id']}: missing chunk {chunk_id}"
                )
                continue

            if chunk["page_number"] not in question["expected_pages"]:
                errors.append(
                    f"{question['id']}: page mismatch for {chunk_id}"
                )

            if chunk.get("source_doc") != question["source_doc"]:
                errors.append(
                    f"{question['id']}: source document mismatch for {chunk_id}"
                )

    if errors:
        print("VALIDATION FAILED")
        for error in errors:
            print("-", error)
        raise SystemExit(1)

    OUTPUT_PATH.write_text(
        json.dumps(combined, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    company_counts = Counter(
        question["id"].split("-RET-")[0]
        for question in combined
    )

    print("VALIDATION PASSED")
    print(f"Total questions: {len(combined)}")
    print(f"Original TCS questions preserved: {len(tcs_questions)}")
    print(f"Additional questions: {len(additional_questions)}")
    print("Company distribution:")

    for company, count in sorted(company_counts.items()):
        print(f"  {company}: {count}")

    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
