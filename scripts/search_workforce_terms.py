import json
from pathlib import Path

report_path = Path("data/processed/tcs_annual_report_2025_26.json")
pages = json.loads(report_path.read_text(encoding="utf-8"))

terms = [
    "reskilling",
    "upskilling",
    "AI training",
    "learning and development",
]

matches = 0

for page in pages:
    page_text = page["text"]
    matched_terms = [term for term in terms if term.lower() in page_text.lower()]

    if matched_terms:
        matches += 1
        print(f"\nPAGE {page['page_number']}")
        print(f"Matched terms: {', '.join(matched_terms)}")
        print(page_text[:1200])

if matches == 0:
    print("No matching workforce-development terms found.")
else:
    print(f"\nTotal pages with matches: {matches}")
