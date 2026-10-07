"""Resumable native-text PDF extraction with explicit per-paper provenance."""

from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from .documents import normalize_document, embedding_blocks, SCHEMA_VERSION
from .storage import file_hash, fingerprint, read_json, write_json

MARKER_CONFIG = {"mode": "fast", "disable_ocr": True, "ocr_inline_math": False, "use_llm": False}


def discover_pdfs(pdf_dir):
    """Identify PDFs by content; filenames are display metadata, not identifiers."""
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.is_dir():
        raise FileNotFoundError(f"PDF folder does not exist: {pdf_dir}")
    records, seen = [], set()
    for path in sorted(pdf_dir.rglob('*')):
        if path.is_file() and path.suffix.lower() == '.pdf':
            sha = file_hash(path)
            records.append({'path': str(path), 'filename': path.relative_to(pdf_dir).as_posix(),
                            'pdf_sha256': sha, 'duplicate': sha in seen})
            seen.add(sha)
    if not records:
        raise ValueError(f"No PDFs found in {pdf_dir}. Add PDFs before running extraction.")
    return records


def make_converter(config):
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    return PdfConverter(artifact_dict=create_model_dict(),
                        renderer='marker.renderers.json.JSONRenderer', config=config)


def extract_papers(records, output_dir, config=None, *, converter=None, extractor_version=None):
    """Cache raw extraction by PDF hash, extractor version and options.

    Reports describe this invocation, including failures and duplicate skips. Paths
    inside reports are relative to output_dir so the output can move between machines.
    Heavy dependencies are loaded only for uncached papers.
    """
    output_dir = Path(output_dir)
    config = dict(MARKER_CONFIG if config is None else config)
    signature = {'extractor': 'marker-pdf', 'version': extractor_version or version('marker-pdf'), 'config': config}
    report = []
    for index, record in enumerate(records, 1):
        start = perf_counter()
        row = {k: record[k] for k in ('filename', 'pdf_sha256')}
        print(f"[{index}/{len(records)}] {record['filename']}", flush=True)
        try:
            if file_hash(record['path']) != record['pdf_sha256']:
                raise ValueError('PDF changed after discovery; rerun discovery')
            if record['duplicate']:
                row['status'] = 'duplicate_skipped'
            else:
                cache_id = fingerprint({**signature, 'pdf_sha256': record['pdf_sha256']})
                raw_path = output_dir / 'raw' / f'{cache_id}.json'
                doc_id = fingerprint({'raw': cache_id, 'schema': SCHEMA_VERSION})
                doc_path = output_dir / 'documents' / f'{doc_id}.json'
                cached = raw_path.exists()
                if cached:
                    raw = read_json(raw_path)
                else:
                    if converter is None:
                        converter = make_converter(config)
                    raw = converter(record['path']).model_dump(mode='json')
                    write_json(raw_path, raw)
                doc = normalize_document(raw, pdf_sha256=record['pdf_sha256'], source_filename=record['filename'])
                doc.update(extraction=signature, raw_path=raw_path.relative_to(output_dir).as_posix())
                write_json(doc_path, doc)
                retained = embedding_blocks(doc, 'omit')
                readable = output_dir / 'readable' / f"{record['pdf_sha256']}.md"
                readable.parent.mkdir(parents=True, exist_ok=True)
                readable.write_text('\n\n'.join(b['text'] for b in retained), encoding='utf-8')
                row.update(status='cached' if cached else 'extracted', title=doc['title'],
                           document_path=doc_path.relative_to(output_dir).as_posix(),
                           readable_path=readable.relative_to(output_dir).as_posix(),
                           quality_flags=doc['quality_flags'], retained_characters=sum(len(b['text']) for b in retained))
        except Exception as error:
            row.update(status='error', error=f'{type(error).__name__}: {error}')
        row['elapsed_seconds'] = round(perf_counter() - start, 3)
        report.append(row)
        write_json(output_dir / 'report.json', report)
        print(f"  {row['status']} ({row['elapsed_seconds']:.1f}s)", flush=True)
    return report


def load_documents(output_dir):
    """Load precisely the latest report's unique papers, never stale cached documents."""
    output_dir = Path(output_dir)
    rows = read_json(output_dir / 'report.json')
    failures = [r for r in rows if r['status'] == 'error']
    if failures:
        raise ValueError(f"{len(failures)} extraction(s) failed. Inspect extraction/report.json before preparation.")
    docs = []
    for row in rows:
        if row['status'] == 'duplicate_skipped':
            continue
        doc = read_json(output_dir / row['document_path'])
        if doc['pdf_sha256'] != row['pdf_sha256']:
            raise ValueError('Document identity does not match extraction report')
        docs.append(doc)
    if not docs or len({d['pdf_sha256'] for d in docs}) != len(docs):
        raise ValueError('Empty extraction or duplicate document IDs')
    return docs
