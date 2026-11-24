import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = PROJECT_ROOT / "data" / "reference" / "nifty100_constituents.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "reference" / "company_master.json"

REQUIRED_COLUMNS = {"Company Name", "Symbol", "Industry"}


def build_company_master() -> list[dict[str, str]]:
    """Build a validated company master from the official NSE CSV."""

    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"Nifty 100 CSV not found: {CSV_PATH}")

    companies = []
    seen_symbols = set()
    seen_names = set()

    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError("CSV does not contain a header row.")

        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames)

        if missing_columns:
            raise ValueError(
                f"CSV is missing required columns: {sorted(missing_columns)}"
            )

        for row_number, row in enumerate(reader, start=2):
            company_name = (row.get("Company Name") or "").strip()
            symbol = (row.get("Symbol") or "").strip()
            industry = (row.get("Industry") or "").strip()

            if not company_name or not symbol or not industry:
                raise ValueError(
                    f"Missing required company data at CSV row {row_number}."
                )

            normalized_symbol = symbol.upper()
            normalized_name = company_name.casefold()

            if normalized_symbol in seen_symbols:
                raise ValueError(f"Duplicate NSE symbol: {symbol}")

            if normalized_name in seen_names:
                raise ValueError(f"Duplicate company name: {company_name}")

            seen_symbols.add(normalized_symbol)
            seen_names.add(normalized_name)

            companies.append(
                {
                    "company_name": company_name,
                    "symbol": symbol,
                    "industry": industry,
                }
            )

    if not companies:
        raise ValueError("No company records found in the CSV.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH.write_text(
        json.dumps(companies, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Companies loaded: {len(companies)}")
    print(f"Industries covered: {len({c['industry'] for c in companies})}")
    print(f"Company master saved: {OUTPUT_PATH}")

    return companies


if __name__ == "__main__":
    build_company_master()
