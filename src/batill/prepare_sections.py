"""Prepare a separate section-based corpus; never extract PDFs or run inference.

Usage: python -m batill.prepare_sections --skip-empty
"""

import argparse
import csv
from pathlib import Path

from . import pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/pipeline.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/corpus_sections'))
    parser.add_argument('--max-tokens', type=int, default=32768,
                        help='Maximum tokens per section input, including title/instruction')
    parser.add_argument('--skip-empty', action='store_true', help='Report and exclude empty extractions')
    args = parser.parse_args()
    config = pipeline.load_config(args.config)
    source = config['output_dir']
    destination = args.output_dir.resolve()
    if destination == source.resolve():
        parser.error('Choose a separate output directory to preserve the original corpus')
    config = {**config, 'output_dir': destination, 'embedding': {
        **config['embedding'], 'chunking': 'sections', 'max_tokens': args.max_tokens}}
    plans = pipeline.prepare(config, skip_empty=args.skip_empty, extraction_dir=source / 'extraction')
    report = destination / 'prepared' / 'section_review.csv'
    columns = ['pdf_sha256', 'source_filename', 'title', 'segment', 'section_group',
               'section_name', 'token_count', 'first_page', 'last_page', 'review_flags']
    with report.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for plan in plans:
            for index, segment in enumerate(plan['segments'], 1):
                pages = [s['page'] for s in segment['sources'] if s['page'] is not None]
                writer.writerow({**{k: plan[k] for k in columns[:3]},
                    'segment': index, 'section_group': segment['section_group'],
                    'section_name': segment['section_name'], 'token_count': segment['token_count'],
                    'first_page': min(pages) if pages else '', 'last_page': max(pages) if pages else '',
                    'review_flags': '; '.join(segment['section_review_flags'])})
    print(f'Prepared {len(plans)} papers. Review section boundaries: {report}')
    print(f'Exact input text: {destination / "prepared" / "readable"}')
    print('No embedding model was loaded. Existing corpus outputs were not changed.')


if __name__ == '__main__':
    main()
