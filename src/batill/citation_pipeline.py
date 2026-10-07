"""Finish the citation graph from existing reviewed identities; no repeated search."""
from pathlib import Path

import pandas as pd

from .citation_graph import citation_selection, retrieve_references, export_citation_graph
from .citation_review import apply_match_review
from .storage import write_json


def prepare_citation_graph(root):
    """Read and validate the current selection and saved review, without API calls."""
    root = Path(root).resolve()
    papers, exclusions, selection = citation_selection(
        root / 'outputs' / 'corpus_sections', root / 'configs' / 'analysis_exclusions.json',
        sorted((root / 'data' / 'bibliography').glob('*.csv')),
    )
    base = root / 'outputs' / 'citation_graph'
    folder = base / selection['selection_id'][:16]
    path = folder / 'matches.csv'
    if not path.exists():
        raise FileNotFoundError(f'No saved matching run for this selection: {path}')
    raw = pd.read_csv(path).fillna('')
    if raw.Key.tolist() != papers.Key.tolist():
        raise ValueError('Saved matches do not cover the current selection in order')
    # Ensure reviewed proposals still refer to the local query metadata.
    for column in ('query_title', 'doi'):
        if raw[column].tolist() != papers[column].tolist():
            raise ValueError(f'Local {column} metadata changed; review the saved matching run')
    matches, audit = apply_match_review(raw, root / 'configs' / 'citation_match_review.json')
    if not matches.approved.any():
        raise ValueError('No approved paper identities are available')
    return {'papers': papers, 'matches': matches, 'review_audit': audit,
            'exclusions': exclusions, 'selection': selection,
            'output_dir': folder, 'cache_dir': base / 'api_cache'}


def finish_citation_graph(job, client):
    """Resume reference retrieval and export the graph with its coverage report.

    Stops on exhausted rate-limit/authentication retries, preserving the API cache
    and reference checkpoints. Rerunning resumes from the cached requests.
    """
    folder = job['output_dir']
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'review').mkdir(exist_ok=True)
    job['papers'].to_csv(folder / 'selected_papers.csv', index=False)
    job['matches'].to_csv(folder / 'reviewed_matches.csv', index=False)
    job['review_audit'].to_csv(folder / 'review' / 'review_decisions.csv', index=False)
    job['exclusions'].to_csv(folder / 'exclusion_audit.csv', index=False)
    write_json(folder / 'selection.json', job['selection'])
    retrieval, references = retrieve_references(client, job['papers'], job['matches'], folder)
    nodes, edges, graph, summary = export_citation_graph(
        job['papers'], job['matches'], retrieval, references, folder, job['selection'], client,
    )
    return {'nodes': nodes, 'edges': edges, 'graph': graph, 'summary': summary,
            'retrieval': retrieval, 'references': references, 'output_dir': folder}
