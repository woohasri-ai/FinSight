import json
from pathlib import Path

report_path = Path("data/processed/tcs_annual_report_2025_26.json")
pages = json.loads(report_path.read_text(encoding="utf-8"))

keywords = [
    "consolidated revenue",
    "operating margin",
    "net profit",
    "revenue growth",
    "total revenue",
    "financial performance",
    "profit after tax",
]

for page in pages:
    text = page["text"]
    text_lower = text.lower()

    matches = [keyword for keyword in keywords if keyword in text_lower]

    if matches:
        print(f"\n{'=' * 60}")
        print(f"Page: {page['page_number']}")
        print(f"Matched terms: {', '.join(matches)}")
        print(text[:1200].replace("\n", " "))
