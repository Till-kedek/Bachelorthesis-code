"""Prepare a traceable abstract corpus from a Scopus CSV export (offline)."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import date
import hashlib
import html
import json
from pathlib import Path
import re
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
FIELDS = {
    "paper_id": "EID",
    "title": "Title",
    "authors": "Authors",
    "year": "Year",
    "journal": "Source title",
    "doi": "DOI",
    "url": "Link",
    "document_language": "Language of Original Document",
    "document_type": "Document Type",
}
MISSING = {"", "[no abstract available]", "no abstract available", "n/a", "null"}
# Restrict markup recognition to actual formatting tags: <.001 is a p-value.
TAGS = re.compile(r"</?(?:p|br|div|span|b|i|em|strong|sub|sup|jats:p)\b[^>]*>", re.I)
NOTICE = re.compile(r"(?:\bCopyrights?\s*:?\s*(?:©|\d{4}|The Author)|©)", re.I)


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFC", html.unescape(value))
    value = value.replace("\u00ad", "").replace("\u200b", "").replace("\ufeff", "")
    return " ".join(value.split())


def clean_abstract(value: str) -> tuple[str, str]:
    """Keep prose intact; return the cleaned text and removed trailing notice."""
    value = normalize(TAGS.sub(" ", value))
    if value.casefold() in MISSING:
        return "", ""
    if value.startswith("Apart from any fair dealing for the purposes of research or private study,") and "Copyright, Designs and Patents Act 1988" in value:
        return "", value
    match = NOTICE.search(value)
    if match:
        # Scopus appends copyright/licensing statements. Preserve every removed
        # suffix in the audit so the decision can be inspected and reversed.
        return value[:match.start()].rstrip(), value[match.start():]
    return value, ""


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare(source: Path, destination: Path) -> dict:
    csv.field_size_limit(100_000_000)  # Large reference fields in the raw export.
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Abstract", *FIELDS.values()}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing Scopus columns: {sorted(missing)}")
        source_columns = list(reader.fieldnames)
        records = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in records):
        raise ValueError("Malformed CSV: at least one record has the wrong field count")
    ids = [normalize(row["EID"]) for row in records]
    if any(not key for key in ids) or len(ids) != len(set(ids)):
        raise ValueError("Scopus EIDs must be nonempty and unique for traceability")

    cutoff_year = date.today().year
    corpus, audit, seen_text, seen_doi = [], [], {}, {}
    for number, raw in enumerate(records, 1):
        abstract, notice = clean_abstract(raw["Abstract"])
        row = {field: normalize(raw[column]) for field, column in FIELDS.items()}
        row["doi"] = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", row["doi"], flags=re.I).lower()
        row.update(source_record=number, abstract=abstract, abstract_word_count=len(abstract.split()))
        row["embedding_text"] = "\n\n".join(part for part in (row["title"], abstract) if part)
        # Apply eligibility rules before deduplication. Only retained records
        # reserve a DOI or text, so excluded records cannot suppress valid ones.
        status, reason, duplicate_of = "kept", "", ""
        if not abstract:
            status, reason = "excluded", "publisher_notice_only" if notice else "missing_abstract"
        elif row["document_type"].casefold() == "retracted":
            status, reason = "excluded", "retracted"
        elif row["year"].isdigit() and int(row["year"]) > cutoff_year:
            status, reason = "excluded", "future_publication_year"
        elif row["doi"] and row["doi"] in seen_doi:
            status, reason = "excluded", "duplicate_doi"
            duplicate_of = seen_doi[row["doi"]]
        elif abstract in seen_text:
            status, reason = "excluded", "duplicate_cleaned_abstract"
            duplicate_of = seen_text[abstract]
        else:
            seen_text[abstract] = row["paper_id"]
            if row["doi"]:
                seen_doi[row["doi"]] = row["paper_id"]
            corpus.append(row)
        audit.append({
            "source_record": number, "paper_id": row["paper_id"],
            "title": row["title"], "status": status, "reason": reason,
            "duplicate_of": duplicate_of,
            "abstract_changed": abstract != raw["Abstract"],
            "removed_notice": notice,
        })

    title_counts = Counter(row["title"].casefold() for row in corpus if row["title"])
    review = []
    for row in corpus:
        flags = []
        if row["title"] and title_counts[row["title"].casefold()] > 1:
            flags.append("shared_title_different_abstract")
        if not row["year"].isdigit():
            flags.append("invalid_year")
        if row["document_type"].casefold() == "erratum":
            flags.append("erratum")
        if "©" in row["abstract"]:
            flags.append("remaining_copyright_symbol")
        if flags:
            review.append({**row, "review_flags": ";".join(flags)})

    fields = ["paper_id", "source_record", *[k for k in FIELDS if k != "paper_id"], "abstract", "abstract_word_count", "embedding_text"]
    outputs = {"abstracts_clean.csv", "cleaning_audit.csv", "review.csv", "cleaning_summary.json"}
    if source.resolve() in {(destination / name).resolve() for name in outputs}:
        raise ValueError("Output would overwrite the source CSV")
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / "abstracts_clean.csv", corpus, fields)
    write_csv(destination / "cleaning_audit.csv", audit, ["source_record", "paper_id", "title", "status", "reason", "duplicate_of", "abstract_changed", "removed_notice"])
    write_csv(destination / "review.csv", review, [*fields, "review_flags"])
    summary = {
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "publication_year_cutoff": cutoff_year,
        "embedding_text_column": "embedding_text",
        "source_rows": len(records), "source_column_count": len(source_columns),
        "retained_rows": len(corpus), "retained_columns": fields,
        "exclusions": dict(Counter(row["reason"] for row in audit if row["status"] == "excluded")),
        "removed_publisher_notices": sum(bool(row["removed_notice"]) for row in audit),
        "review_rows": len(review),
        "review_flags": dict(Counter(flag for row in review for flag in row["review_flags"].split(";"))),
        "retained_document_languages": dict(Counter(row["document_language"] for row in corpus)),
        "retained_document_types": dict(Counter(row["document_type"] for row in corpus)),
        "rules": [
            "Source preserved; source_record is the 1-based CSV data record, not a physical line number.",
            "Decode HTML entities; remove known formatting tags and invisible artifacts; normalize whitespace and Unicode NFC.",
            "Remove trailing copyright/licensing notices, preserving each removed suffix in the audit.",
            "Exclude missing/placeholder abstracts, notice-only entries and repeated exact cleaned abstracts; retain first source occurrence.",
            "Exclude records marked Retracted and publication years later than the current year, recorded as publication_year_cutoff.",
            "Among eligible records retain the first source occurrence per nonempty normalized DOI; audit later occurrences as duplicate_doi.",
            "Exclusion precedence: missing/notice-only abstract, retracted, future year, duplicate DOI, duplicate text. Only retained records reserve identifiers/text.",
            "Keep short abstracts without a length warning; keep missing DOIs. Flag shared titles, invalid years and errata for review.",
            "Keep all languages and subjects; remaining review flags are not exclusions.",
            "Document language describes the publication and need not be the abstract language.",
            "Preserve case, punctuation, numbers, stopwords and word order; no stemming, lemmatization or n-gram tokenization.",
            "Use embedding_text for vector embedding: normalized title, blank line, cleaned abstract. If title is empty, use the abstract alone.",
            "Keep title and abstract separately for inspection and later n-gram analysis; other metadata is not included in embedding_text.",
        ],
    }
    (destination / "cleaning_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/bibliography/All_private_equity_bib.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/abstracts")
    args = parser.parse_args()
    result = prepare(args.input, args.output_dir)
    print(json.dumps({key: result[key] for key in ["source_rows", "retained_rows", "exclusions", "removed_publisher_notices", "review_rows"]}, indent=2))


if __name__ == "__main__":
    main()
