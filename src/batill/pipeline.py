"""Public pipeline stages used by both the notebook and command-line runner."""

from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from .embedding import (EmbeddingConfig, RuntimeConfig, load_tokenizer, prepare_corpus,
                        load_plans, load_model, embed_corpus, assemble_corpus, validate_result)
from .extraction import discover_pdfs, extract_papers, load_documents
from .storage import file_hash, fingerprint, read_json, write_json


def load_config(path):
    """Resolve paths relative to the project, not the notebook's working directory."""
    path = Path(path).resolve()
    config = read_json(path)
    root = path.parent.parent
    result = {**config, 'pdf_dir': root / config['pdf_dir'], 'output_dir': root / config['output_dir']}
    if result['pdf_dir'].resolve() == result['output_dir'].resolve():
        raise ValueError('PDF input and generated output folders must differ')
    EmbeddingConfig(**result['embedding'])
    RuntimeConfig(**result['runtime'])
    return result


def extract(config):
    """Discover a fresh folder, extract unique PDFs and persist timing provenance."""
    started = perf_counter()
    records = discover_pdfs(config['pdf_dir'])
    report = extract_papers(records, config['output_dir'] / 'extraction')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    write_json(config['output_dir'] / 'extraction' / 'timings' / f'{stamp}.json', {
        'wall_seconds': perf_counter() - started, 'papers': report})
    return report


def prepare(config, selected_ids=None, *, skip_empty=False, extraction_dir=None):
    """Prepare reviewed paper text; selection defaults to all successful unique PDFs.

    This stage verifies the input folder still matches the extraction report. It
    never treats a successful extraction as a scientific quality assessment.
    Set skip_empty=True to explicitly exclude papers without retained text; these
    are listed in prepared/preparation_report.json and the prepared manifest.
    """
    output = config['output_dir']
    documents = load_documents(Path(extraction_dir) if extraction_dir is not None else output / 'extraction')
    current = {r['pdf_sha256'] for r in discover_pdfs(config['pdf_dir'])}
    if current != {d['pdf_sha256'] for d in documents}:
        raise ValueError('PDF folder changed since extraction. Run extraction again before preparation.')
    if selected_ids is not None:
        selected_ids = list(selected_ids)
        if len(set(selected_ids)) != len(selected_ids) or not set(selected_ids) <= current:
            raise ValueError('Selection contains duplicate or unknown PDF IDs')
        by_id = {d['pdf_sha256']: d for d in documents}
        documents = [by_id[key] for key in selected_ids]
    if not documents:
        raise ValueError('Select at least one paper')
    embedding = EmbeddingConfig(**config['embedding'])
    tokenizer, revision = load_tokenizer(embedding, output / 'prepared')
    return prepare_corpus(documents, tokenizer, embedding, revision, output / 'prepared',
                          skip_empty=skip_empty)


def embed(output_dir, runtime=None, workers=1):
    """Run the complete embedding stage on a suitably resourced local machine.

    When every prepared paper already has a valid cache (including results copied
    from another machine), assemble directly without loading model weights.
    """
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError('workers must be a positive integer')
    output_dir = Path(output_dir)
    plans = load_plans(output_dir / 'prepared')
    destination = output_dir / 'embeddings'
    complete = True
    for plan in plans:
        path = destination / 'papers' / f"{plan['id']}.json"
        if path.exists():
            validate_result(plan, read_json(path))
        else:
            complete = False
    if complete:
        return assemble_corpus(plans, destination)
    if workers > 1:
        from .parallel import embed_parallel
        return embed_parallel(output_dir, runtime, workers)
    model = load_model(EmbeddingConfig(**plans[0]['config']), runtime=runtime or RuntimeConfig())
    return embed_corpus(plans, model, destination)


def validate_outputs(output_dir):
    """Verify all cached vectors, corpus identity, row order and the final archive."""
    output_dir = Path(output_dir)
    plans = load_plans(output_dir / 'prepared')
    directory = output_dir / 'embeddings'
    manifest = read_json(directory / 'manifest.json')
    if manifest['corpus_id'] != fingerprint([p['id'] for p in plans]):
        raise ValueError('Output corpus does not match prepared inputs')
    if manifest['config'] != plans[0]['config']:
        raise ValueError('Output model/configuration does not match prepared inputs')
    if manifest['matrix_sha256'] != file_hash(directory / 'embeddings.npz'):
        raise ValueError('Output matrix checksum mismatch')
    if [r['input_id'] for r in manifest['rows']] != [p['id'] for p in plans]:
        raise ValueError('Manifest order does not match prepared inputs')
    expected = []
    for plan, row in zip(plans, manifest['rows']):
        if row['pdf_sha256'] != plan['pdf_sha256'] or row['title'] != plan['title']:
            raise ValueError('Manifest paper identity mismatch')
        result = read_json(directory / 'papers' / f"{plan['id']}.json")
        expected.append(validate_result(plan, result))
        if row['execution'] != result['execution']:
            raise ValueError('Execution provenance mismatch')
    with np.load(directory / 'embeddings.npz', allow_pickle=False) as saved:
        vectors = saved['embeddings']
        if saved['pdf_sha256'].tolist() != [p['pdf_sha256'] for p in plans]:
            raise ValueError('Matrix rows do not match paper IDs')
        if vectors.shape != (len(plans), plans[0]['config']['embedding_dimension']):
            raise ValueError('Unexpected matrix dimensions')
        if not np.isfinite(vectors).all() or not np.allclose(vectors, expected, atol=1e-6):
            raise ValueError('Matrix differs from per-paper vectors')
    return {'papers': len(plans), 'dimensions': vectors.shape[1],
            'corpus_id': manifest['corpus_id'], 'matrix': str(directory / 'embeddings.npz')}
