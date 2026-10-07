"""Content-reviewed Scopus examples bound to the approved Final abstract corpus."""

from pathlib import Path

import numpy as np
import pandas as pd

from .pdf_cluster_examples import export_representative_papers as export_review
from .storage import file_hash, fingerprint


METADATA_FIELDS = ('title', 'authors', 'year', 'journal', 'doi', 'url',
                   'document_type', 'source_record')


def membership_id(papers, labels):
    """Use Scopus IDs, not PDF identities or corpus row numbers."""
    return fingerprint(sorted(zip(papers.paper_id.astype(str), map(int, labels))))


def representative_papers(context, analysis, settings, review):
    papers = analysis['papers'].reset_index(drop=True)
    labels = np.asarray(analysis['labels'])
    if (settings.get('source') != 'saved' or settings.get('saved') != review['solution']
            or settings.get('method') != 'K-means'
            or settings.get('clusters') != len(review['focus'])
            or membership_id(papers, labels) != review['membership_id']):
        raise ValueError('These examples require the exact reviewed saved K-means memberships.')
    if (context['input_id'] != review['embedding_input_id']
            or context['input_hashes'] != review['input_hashes']):
        raise ValueError('The reviewed Final abstract inputs or catalog changed.')
    root = Path(context['root'])
    if file_hash(root / review['source_csv']) != review['input_hashes']['abstracts_clean.csv']:
        raise ValueError('The reviewed abstract source CSV changed.')
    if (not papers.paper_id.is_unique
            or papers.paper_id.tolist() != context['papers'].paper_id.tolist()
            or not np.array_equal(labels, context['solutions'][review['solution']]['labels'])
            or not np.array_equal(analysis['embeddings'], context['embeddings'])):
        raise ValueError('Use the exact saved Final abstract IDs, order, labels and vectors.')
    choices = pd.DataFrame(review['papers'])
    clusters = {int(c) for c in review['focus']}
    if (choices.paper_id.duplicated().any() or set(labels) != clusters
            or choices.groupby('cluster').size().to_dict() != {c: 2 for c in clusters}):
        raise ValueError('Choose exactly two distinct papers from each reviewed cluster.')

    diagnostics = papers[['paper_id', *METADATA_FIELDS]].copy()
    diagnostics['cluster'] = labels
    diagnostics['embedding_text_sha256'] = papers.embedding_text.map(fingerprint)
    membership = analysis['membership'].set_index('paper_id')
    diagnostics['cosine_silhouette'] = diagnostics.paper_id.map(membership.cosine_silhouette)
    vectors = np.array(analysis['embeddings'], dtype=float, copy=True)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    for cluster in sorted(clusters):
        indices = np.flatnonzero(labels == cluster)
        within = vectors[indices]
        # Mean cosine to the other members; no quadratic similarity matrix needed.
        diagnostics.loc[indices, 'mean_within_cluster_cosine'] = (
            within @ within.sum(axis=0) - (within * within).sum(axis=1)) / (len(indices) - 1)
        diagnostics.loc[indices, 'cluster_papers'] = len(indices)
    diagnostics['cluster_papers'] = diagnostics.cluster_papers.astype(int)
    diagnostics['centrality_rank'] = diagnostics.groupby('cluster').mean_within_cluster_cosine.rank(
        ascending=False, method='min').astype(int)
    diagnostics['selected'] = diagnostics.paper_id.isin(choices.paper_id)

    selected, evidence = [], []
    for choice in review['papers']:
        rows = papers.loc[papers.paper_id.eq(choice['paper_id'])]
        if len(rows) != 1:
            raise ValueError(f'Missing or duplicated reviewed abstract: {choice["paper_id"]}')
        paper = rows.iloc[0]
        diagnostic = diagnostics.loc[diagnostics.paper_id.eq(paper.paper_id)].iloc[0].to_dict()
        if (diagnostic['cluster'] != choice['cluster']
                or fingerprint(paper.embedding_text) != choice['embedding_text_sha256']
                or fingerprint(paper.abstract) != choice['abstract_sha256']
                or any(str(paper[field]) != choice[field] for field in METADATA_FIELDS)):
            raise ValueError(f'Reviewed abstract text, metadata or membership changed: {paper.paper_id}')
        if len(choice['evidence']) != 2:
            raise ValueError('Each reviewed paper requires two supporting abstract passages.')
        for number, passage in enumerate(choice['evidence'], 1):
            start, end = passage['excerpt_start'], passage['excerpt_end']
            if (passage['source_field'] != 'abstract'
                    or not 0 <= start < end <= len(paper.abstract)
                    or paper.abstract[start:end] != passage['excerpt']):
                raise ValueError(f'Reviewed passage no longer matches source: {paper.paper_id}')
            evidence.append({'paper_id': paper.paper_id, 'cluster': choice['cluster'],
                'evidence_number': number, **passage, 'source_kind': 'cleaned_scopus_abstract',
                'source_path': review['source_csv'],
                'source_sha256': review['input_hashes']['abstracts_clean.csv'],
                'source_record': paper.source_record, 'source_url': paper.url,
                'page': None, 'block_id': None})
        selected.append({**diagnostic, **{k: v for k, v in choice.items() if k != 'evidence'},
                         'requested_focus': review['focus'][str(choice['cluster'])],
                         'abstract': paper.abstract})
    return pd.DataFrame(selected), pd.DataFrame(evidence), diagnostics


def export_representative_papers(context, review_path, review, selected, evidence, diagnostics, settings):
    return export_review(context['root'], review_path, review, selected, evidence, diagnostics, settings,
        report_directory='reports/abstract_final_cluster_examples',
        provenance={'embedding_input_id': review['embedding_input_id'],
                    'source_input_hashes': context['input_hashes'],
                    'source_csv': review['source_csv'],
                    'abstract_examples_code_sha256': file_hash(Path(__file__)),
                    'diagnostic_representation': 'saved Final broad title-and-abstract vectors',
                    'evidence_basis': 'included cleaned Scopus abstracts',
                    'cluster_review_notes': review.get('cluster_review_notes', {})})
