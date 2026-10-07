"""Explicit paper exclusions for analysis, without modifying stored embeddings."""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .storage import fingerprint, read_json


def filter_analysis_corpus(papers, embeddings, exclusion_file, *, key_column='Key'):
    """Apply reviewed PDF-hash exclusions to aligned paper rows and vectors.

    Preserve manifest Paper IDs and row order; reset only the DataFrame index.
    Return filtered copies, a per-decision audit table and a selection fingerprint.
    Unmatched exclusion entries are reported, never matched approximately by title.
    """
    if len(papers) != len(embeddings) or np.ndim(embeddings) != 2:
        raise ValueError('Paper rows and embedding rows must be aligned')
    if papers[key_column].isna().any() or not papers[key_column].is_unique:
        raise ValueError('Paper identities must be present and unique')
    config = read_json(exclusion_file)
    if config.get('schema_version') != 1:
        raise ValueError('Unsupported analysis exclusion schema')
    records = config['papers']
    hashes = [r['pdf_sha256'] for r in records]
    if len(set(hashes)) != len(hashes):
        raise ValueError('Duplicate PDF hashes in exclusion list')
    if any(not re.fullmatch(r'[0-9a-f]{64}', h) for h in hashes):
        raise ValueError('Exclusions must use full PDF SHA-256 hashes')
    if any(type(r.get('exclude')) is not bool or not r.get('reason') for r in records):
        raise ValueError('Each exclusion needs a boolean exclude field and a reason')
    active = {r['pdf_sha256'] for r in records if r['exclude']}
    present = set(papers[key_column])
    keep = ~papers[key_column].isin(active).to_numpy()
    selected = papers.iloc[np.flatnonzero(keep)].copy().reset_index(drop=True)
    vectors = np.asarray(embeddings)[keep].copy()
    audit = pd.DataFrame([{**r, 'status': (
        'kept_by_decision' if not r['exclude'] else
        'excluded' if r['pdf_sha256'] in present else 'not_in_loaded_corpus')}
        for r in records])
    details = {
        'exclusion_file': str(Path(exclusion_file)), 'config_id': fingerprint(config),
        'loaded_papers': len(papers), 'excluded_papers': int((~keep).sum()),
        'retained_papers': len(selected),
        'unmatched_exclusions': sorted(active - present),
        'selection_id': fingerprint(selected[key_column].tolist()),
    }
    return selected, vectors, audit, details


def deduplicate_pdf_corpus(papers, embeddings, exclusion_file, selection, *, key_column='Key'):
    """Keep reviewed canonical PDFs before fitting, preserving vectors and IDs.

    The sibling pdf_duplicates.json contains identity decisions only. The scope
    fingerprint remains available for locating the independently built citation graph.
    """
    path = Path(exclusion_file).with_name('pdf_duplicates.json')
    if not path.exists():
        return papers, embeddings, selection
    config = read_json(path)
    if config.get('schema_version') != 1:
        raise ValueError('Unsupported PDF duplicate schema')
    pairs = config['duplicate_pdfs']
    duplicates = [pair['duplicate_key'] for pair in pairs]
    canonicals = {pair['canonical_key'] for pair in pairs}
    if len(set(duplicates)) != len(duplicates) or set(duplicates) & canonicals:
        raise ValueError('Duplicate decisions must map directly to distinct canonical PDFs')
    if len(papers) != len(embeddings) or not papers[key_column].is_unique:
        raise ValueError('Paper identities and embedding rows must be aligned and unique')
    present = set(papers[key_column])
    for pair in pairs:
        if pair['duplicate_key'] in present and pair['canonical_key'] not in present:
            raise ValueError('Cannot remove a duplicate whose canonical PDF is absent')
    keep = ~papers[key_column].isin(duplicates).to_numpy()
    selected = papers.loc[keep].copy().reset_index(drop=True)
    details = {**selection,
        'scope_selection_id': selection['selection_id'],
        'scope_retained_papers': len(papers),
        'duplicate_pdfs_removed': int((~keep).sum()),
        'duplicate_config_id': fingerprint(config),
        'retained_papers': len(selected),
        'selection_id': fingerprint(selected[key_column].tolist())}
    return selected, np.asarray(embeddings)[keep].copy(), details
