"""Reviewed PDF examples tied to exact memberships and retained source passages."""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .storage import file_hash, fingerprint, write_json


def membership_id(papers, labels):
    return fingerprint(sorted(zip(papers.pdf_sha256.astype(str), map(int, labels))))


def validate_review(analysis, settings, review):
    """Bind content judgments to the reviewed partition, never to IDs alone."""
    papers, labels = analysis['papers'], np.asarray(analysis['labels'])
    if (settings.get('source') != 'saved' or settings.get('saved') != review['solution']
            or not np.isclose(settings.get('resolution', np.nan), review['resolution'])
            or membership_id(papers, labels) != review['membership_id']):
        raise ValueError(f"These reviewed examples require the exact saved {review['solution']} memberships.")
    choices = pd.DataFrame(review['papers'])
    expected_clusters = {int(cluster) for cluster in review['focus']}
    if (choices.paper_id.duplicated().any() or set(labels) != expected_clusters
            or choices.groupby('cluster').size().to_dict() != {c: 2 for c in expected_clusters}):
        raise ValueError('Choose exactly two distinct papers from each reviewed cluster.')


def representative_papers(analysis, settings, review):
    """Validate curated choices and compute diagnostics for every assigned paper.

    Content judgments are recorded explicitly in the review, not inferred from
    keyword counts or similarity. A changed partition requires a new review.
    """
    validate_review(analysis, settings, review)
    papers, labels = analysis['papers'], np.asarray(analysis['labels'])
    expected_clusters = {int(cluster) for cluster in review['focus']}
    choices = pd.DataFrame(review['papers'])
    vectors = np.asarray(analysis['embeddings'], dtype=float)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    similarity = vectors @ vectors.T
    diagnostics = papers[['paper_id', 'title', 'pdf_sha256', 'text_sha256', 'source_filename']].copy()
    diagnostics['cluster'] = labels
    # Membership diagnostics were computed on this exact analysis, in corpus order.
    membership = analysis['membership'].set_index('paper_id')
    diagnostics['cosine_silhouette'] = diagnostics.paper_id.map(membership.cosine_silhouette)
    for cluster in sorted(expected_clusters):
        indices = np.flatnonzero(labels == cluster)
        within = similarity[np.ix_(indices, indices)]
        centrality = (within.sum(axis=1) - np.diag(within)) / (len(indices) - 1)
        diagnostics.loc[diagnostics.index[indices], 'mean_within_cluster_cosine'] = centrality
        diagnostics.loc[diagnostics.index[indices], 'cluster_papers'] = len(indices)
    diagnostics['centrality_rank'] = diagnostics.groupby('cluster').mean_within_cluster_cosine.rank(
        ascending=False, method='min').astype(int)
    diagnostics['cluster_papers'] = diagnostics.cluster_papers.astype(int)
    diagnostics['selected'] = diagnostics.paper_id.isin(choices.paper_id)
    selected, evidence = reviewed_evidence(analysis, review, diagnostics)
    return selected, evidence, diagnostics


def reviewed_evidence(analysis, review, diagnostics):
    """Share exact source and membership checks across PDF and citation reviews."""
    papers = analysis['papers']
    selected, evidence = [], []
    for choice in review['papers']:
        matches = papers.loc[papers.paper_id.eq(choice['paper_id'])]
        if len(matches) != 1:
            raise ValueError(f"Missing or nonunique reviewed paper: {choice['paper_id']}")
        paper = matches.iloc[0]
        diagnostic = diagnostics.loc[diagnostics.paper_id.eq(paper.paper_id)].iloc[0].to_dict()
        if (diagnostic['cluster'] != choice['cluster'] or paper.pdf_sha256 != choice['pdf_sha256']
                or paper.text_sha256 != choice['text_sha256']):
            raise ValueError(f'Reviewed membership or source text changed: {paper.paper_id}')
        passages = analysis['passages'].loc[analysis['passages'].row.eq(paper.row)]
        if not choice['evidence']:
            raise ValueError(f'Missing content evidence: {paper.paper_id}')
        for number, quote in enumerate(choice['evidence'], 1):
            source = passages.loc[passages.page.eq(quote['page'])
                                  & passages.block_id.eq(quote['block_id'])
                                  & passages.segment.eq(quote['segment'])]
            if len(source) != 1 or quote['excerpt'] not in source.iloc[0].text:
                raise ValueError(f'Reviewed passage no longer matches source: {paper.paper_id}')
            text = source.iloc[0].text
            start = text.index(quote['excerpt'])
            evidence.append({'paper_id': paper.paper_id, 'cluster': choice['cluster'],
                             'evidence_number': number, **quote,
                             'excerpt_start': start, 'excerpt_end': start + len(quote['excerpt'])})
        selected.append({**diagnostic, **{key: value for key, value in choice.items()
                                         if key not in {'evidence', 'pdf_sha256', 'text_sha256'}},
                         'requested_focus': review['focus'][str(choice['cluster'])]})
    return pd.DataFrame(selected), pd.DataFrame(evidence)


def export_representative_papers(root, review_path, review, selected, evidence, diagnostics, settings,
                                 *, report_directory='reports/pdf_cluster_examples', provenance=None):
    """Save the qualitative judgments, source locations and quantitative context."""
    root, review_path = Path(root), Path(review_path)
    destination = root / report_directory / file_hash(review_path)[:16]
    destination.mkdir(parents=True, exist_ok=True)
    for name, table in [('representative_papers.csv', selected), ('source_passages.csv', evidence),
                        ('all_paper_diagnostics.csv', diagnostics)]:
        table.to_csv(destination / name, index=False)
    write_json(destination / 'run_manifest.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'review_config': str(review_path.relative_to(root)),
        'review_config_sha256': file_hash(review_path),
        'membership_id': review['membership_id'], 'partition': settings,
        'selection_method': review['selection_method'],
        'clustering_performed': False, 'papers_removed': 0,
        'code_sha256': file_hash(Path(__file__)),
        'provenance': provenance or {},
        'artifact_sha256': {path.name: file_hash(path) for path in destination.glob('*.csv')},
    })
    return destination
