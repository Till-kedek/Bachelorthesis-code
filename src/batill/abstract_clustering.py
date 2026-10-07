"""Load an audited full-corpus abstract embedding run without PDF dependencies."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .storage import file_hash


def load_abstract_embeddings(run_dir, source_csv):
    """Return metadata, vectors and provenance in saved embedding-row order.

    The saved full run is explicit: never choose a sample or a different run by
    modification time. Check its original source checksum, every metadata/text
    row, both saved ID tables, token audit, shape and normalization before use.
    No vectors are inferred, truncated, padded, filtered or silently normalized.
    """
    run_dir, source_csv = Path(run_dir), Path(source_csv)
    paths = [source_csv, *(run_dir / name for name in (
        'run_summary.json', 'embeddings.npy', 'paper_ids.csv',
        'selected_papers.csv', 'token_lengths.csv',
    ))]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f'Missing abstract input: {path}. Check the configured paths.')
    summary = json.loads((run_dir / 'run_summary.json').read_text(encoding='utf-8'))
    if summary.get('status') != 'complete' or summary.get('run_mode') != 'full':
        raise ValueError('Choose a completed full abstract run, not a sample or interrupted run.')
    expected_settings = {
        'model': 'BAAI/bge-m3', 'input_column': 'embedding_text',
        'input_format': 'title + blank line + abstract', 'normalization': 'L2',
        'truncation': False, 'chunking': False,
    }
    for name, expected in expected_settings.items():
        if summary.get(name) != expected:
            raise ValueError(f'Unexpected abstract run setting {name}: {summary.get(name)!r}')
    if file_hash(source_csv) != summary.get('source_sha256'):
        raise ValueError('Abstract source checksum differs from the embedded source; do not mix runs.')

    source = pd.read_csv(source_csv, dtype=str, keep_default_na=False)
    selected = pd.read_csv(run_dir / 'selected_papers.csv', dtype=str, keep_default_na=False)
    ids = pd.read_csv(run_dir / 'paper_ids.csv', dtype=str, keep_default_na=False)
    audit = pd.read_csv(run_dir / 'token_lengths.csv', dtype=str, keep_default_na=False)
    required = {'paper_id', 'title', 'abstract', 'embedding_text', 'doi', 'year',
                'journal', 'document_type', 'document_language', 'abstract_word_count'}
    for name, table, columns in (
        ('source', source, required),
        ('selected_papers', selected, set(source.columns) | {'embedding_row', 'token_count', 'over_limit'}),
        ('paper_ids', ids, {'embedding_row', 'paper_id'}),
        ('token_lengths', audit, {'paper_id', 'title', 'token_count', 'over_limit'}),
    ):
        if not columns.issubset(table.columns):
            raise ValueError(f'{name} is missing columns: {sorted(columns - set(table.columns))}')
        if table['paper_id'].str.strip().eq('').any() or not table['paper_id'].is_unique:
            raise ValueError(f'{name} must contain unique nonempty paper IDs.')

    n = len(source)
    if n < 4 or summary.get('corpus_size') != n or summary.get('selected_size') != n:
        raise ValueError('Full run counts must match the source and contain at least four abstracts.')
    if any(len(table) != n for table in (selected, ids, audit)):
        raise ValueError('Full run ID, metadata and token-audit row counts differ.')
    expected_rows = [str(i) for i in range(n)]
    if any(table['embedding_row'].tolist() != expected_rows for table in (selected, ids)):
        raise ValueError('embedding_row must be contiguous and in matrix row order in both ID tables.')
    saved_ids = ids['paper_id'].tolist()
    if selected['paper_id'].tolist() != saved_ids:
        raise ValueError('Saved ID order differs between paper_ids and selected_papers.')
    if set(saved_ids) != set(source['paper_id']) or set(audit['paper_id']) != set(saved_ids):
        raise ValueError('Full run IDs do not cover exactly the source corpus.')
    aligned_source = source.set_index('paper_id', drop=False).loc[saved_ids].reset_index(drop=True)
    if not selected[source.columns].equals(aligned_source):
        raise ValueError('Saved metadata or text differs from the embedded source for its paper IDs.')
    expected_texts = ['\n\n'.join(part for part in (title, abstract) if part)
                      for title, abstract in zip(selected['title'], selected['abstract'])]
    if (selected['abstract'].str.strip().eq('').any()
            or selected['embedding_text'].tolist() != expected_texts):
        raise ValueError('Each embedding input must be its title plus the complete cleaned abstract.')

    def positive_integers(values, name):
        numbers = pd.to_numeric(values, errors='raise')
        if not np.isfinite(numbers).all() or (numbers <= 0).any() or (numbers % 1 != 0).any():
            raise ValueError(f'{name} must contain positive finite integers.')
        return numbers.astype(np.int64)

    token_counts = positive_integers(selected['token_count'], 'token_count')
    aligned_audit = audit.set_index('paper_id').loc[saved_ids].reset_index()
    if (aligned_audit['title'].tolist() != selected['title'].tolist()
            or not np.array_equal(positive_integers(aligned_audit['token_count'], 'audit token_count'), token_counts)):
        raise ValueError('Saved token audit does not match the selected abstracts.')
    max_tokens = summary.get('max_tokens')
    if not isinstance(max_tokens, int) or max_tokens < 1 or (token_counts > max_tokens).any():
        raise ValueError('Abstract token counts exceed the recorded model input limit.')
    if any(table['over_limit'].str.casefold().ne('false').any() for table in (selected, audit)):
        raise ValueError('The saved run contains over-limit inputs.')
    for field in ('selected_over_limit', 'corpus_over_limit'):
        if summary.get(field) != 0:
            raise ValueError(f'Run summary reports {field}; expected zero.')
    for field in ('maximum_sample_tokens', 'maximum_corpus_tokens'):
        if summary.get(field) != int(token_counts.max()):
            raise ValueError(f'Run summary {field} differs from the token audit.')
    word_counts = positive_integers(selected['abstract_word_count'], 'abstract_word_count')
    if not np.array_equal(word_counts, selected['abstract'].str.split().str.len()):
        raise ValueError('Abstract word counts differ from the saved text.')

    embeddings = np.load(run_dir / 'embeddings.npy', allow_pickle=False)
    if (embeddings.ndim != 2 or embeddings.shape[0] != n or embeddings.shape[1] < 2
            or list(embeddings.shape) != summary.get('shape')):
        raise ValueError('Embedding matrix shape differs from the saved run or its paper IDs.')
    if embeddings.dtype.kind != 'f' or str(embeddings.dtype) != summary.get('dtype'):
        raise ValueError('Embedding dtype differs from the saved floating-point format.')
    if not np.isfinite(embeddings).all():
        raise ValueError('Embedding matrix contains non-finite values.')
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5, rtol=0):
        raise ValueError('Abstract embeddings must be nonzero L2-normalized vectors.')

    papers = selected.copy()
    papers['embedding_row'] = np.arange(n)
    papers['token_count'] = token_counts
    papers['abstract_word_count'] = word_counts
    papers['over_limit'] = False
    papers['Key'] = papers['paper_id']  # Scopus EIDs, never PDF content hashes.
    papers['Paper ID'] = papers['paper_id']
    papers['Title'] = papers['title']
    papers['text_preview'] = papers['abstract'].str.slice(stop=1500)
    provenance = {**summary, 'run_dir': str(run_dir.resolve()),
                  'source_csv': str(source_csv.resolve()),
                  'input_hashes': {path.name: file_hash(path) for path in paths}}
    return papers, embeddings, provenance
