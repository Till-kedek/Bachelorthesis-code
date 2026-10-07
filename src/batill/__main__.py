"""Run `python -m batill --help` for the reproducible command-line interface."""
import argparse
from pathlib import Path

from .pipeline import load_config, extract, prepare, embed, validate_outputs
from .embedding import RuntimeConfig


def main():
    parser = argparse.ArgumentParser(description='PDF extraction and Qwen3 paper embeddings')
    parser.add_argument('stage', choices=['extract', 'prepare', 'embed', 'validate', 'all'])
    parser.add_argument('--config', type=Path, default=Path('configs/pipeline.json'))
    parser.add_argument('--output-dir', type=Path, help='Use an existing prepared output folder (embed/validate only)')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--attention', choices=['sdpa', 'batill_sdpa_repeat_kv'])
    parser.add_argument('--sdpa-backend', choices=['auto', 'efficient'])
    parser.add_argument('--workers', type=int, default=1, help='Independent CUDA GPU workers for embedding')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    config = load_config(args.config)
    if args.output_dir:
        if args.stage not in {'embed', 'validate'}:
            parser.error('--output-dir is only valid with embed/validate')
        config['output_dir'] = args.output_dir.resolve()
    runtime = dict(config['runtime'])
    for key in ['device', 'attention', 'sdpa_backend']:
        if getattr(args, key) is not None:
            runtime[key] = getattr(args, key)
    if args.stage in {'extract', 'all'}:
        report = extract(config)
        if any(row['status'] == 'error' for row in report):
            raise SystemExit('Extraction failed for one or more PDFs; inspect extraction/report.json')
    if args.stage in {'prepare', 'all'}:
        prepare(config)
    if args.stage in {'embed', 'all'}:
        embed(config['output_dir'], RuntimeConfig(**runtime), workers=args.workers)
    if args.stage in {'validate', 'all'}:
        print(validate_outputs(config['output_dir']))


if __name__ == '__main__':
    main()
