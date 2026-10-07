"""Audit PE mentions in the original corpus without filtering or changing inputs.

Case-insensitive matches: the phrase 'private equity' (including hyphenated
forms), or the standalone abbreviation 'PE'. These are lexical matches only,
not judgements of a paper's substantive relevance. Also report phrase-only
counts so the effect of including the abbreviation remains visible.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
PHRASE = re.compile(r'\bprivate[\s-]+equity\b', re.IGNORECASE)
ABBREVIATION = re.compile(r'\bPE\b', re.IGNORECASE)
CATEGORIES = ('neither', 'abstract_only', 'title_only', 'both')


def normalize(text):
    return unicodedata.normalize('NFKC', text).translate(str.maketrans({
        '\u2010': '-', '\u2011': '-', '\u2012': '-', '\u2013': '-',
        '\u2014': '-', '\u2212': '-', '\u00ad': '',
    }))


def category(title_match, abstract_match):
    if title_match and abstract_match:
        return 'both'
    if title_match:
        return 'title_only'
    if abstract_match:
        return 'abstract_only'
    return 'neither'


def audit(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source in {destination / 'paper_mentions.csv', destination / 'summary.json'}:
        raise ValueError('Audit output must not overwrite the input.')
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    with source.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        required = {'paper_id', 'title', 'abstract'}
        if not required <= set(reader.fieldnames or []):
            raise ValueError('Expected paper_id, title and abstract columns.')
        papers = list(reader)
    ids = [row['paper_id'] for row in papers]
    if len(set(ids)) != len(ids) or any(not key.strip() for key in ids):
        raise ValueError('Paper IDs must be unique and nonempty.')
    rows = []
    for paper in papers:
        row = {field: paper[field] for field in ('paper_id', 'title', 'abstract')}
        for field in ('title', 'abstract'):
            text = normalize(paper[field])
            row[field + '_phrase'] = bool(PHRASE.search(text))
            row[field + '_abbreviation'] = bool(ABBREVIATION.search(text))
            row[field + '_any_pe'] = row[field + '_phrase'] or row[field + '_abbreviation']
        row['category'] = category(row['title_any_pe'], row['abstract_any_pe'])
        row['phrase_only_category'] = category(row['title_phrase'], row['abstract_phrase'])
        rows.append(row)
    summaries = {}
    for method, column in [('phrase_or_abbreviation', 'category'),
                           ('phrase_only', 'phrase_only_category')]:
        counts = Counter(row[column] for row in rows)
        summaries[method] = {name: counts[name] for name in CATEGORIES}
        assert sum(summaries[method].values()) == len(papers)
    summary = {
        'source': str(source), 'source_sha256': before, 'papers': len(papers),
        'matching': {
            'phrase': PHRASE.pattern, 'abbreviation': ABBREVIATION.pattern,
            'case_sensitive': False, 'unicode_and_hyphens_normalized': True,
            'fields': ['title', 'abstract'],
            'note': 'Lexical presence only; no relevance judgement and no filtering.',
        },
        'counts': summaries,
    }
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / 'paper_mentions.csv').open('w', encoding='utf-8', newline='') as handle:
        fields = ['paper_id', 'title', 'abstract', 'title_phrase', 'title_abbreviation',
                  'title_any_pe', 'abstract_phrase', 'abstract_abbreviation',
                  'abstract_any_pe', 'category', 'phrase_only_category']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (destination / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before, 'Input changed during audit.'
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/abstracts/abstracts_clean.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reports/abstract_pe_mentions')
    args = parser.parse_args()
    print(json.dumps(audit(args.input, args.output_dir), indent=2))


if __name__ == '__main__':
    main()
