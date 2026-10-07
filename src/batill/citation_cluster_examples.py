"""Content-reviewed examples for the saved citation communities, using graph diagnostics."""

from pathlib import Path

import numpy as np
import pandas as pd

from .pdf_cluster_examples import (
    export_representative_papers as export_review,
    membership_id, reviewed_evidence, validate_review,
)
from .storage import file_hash


def representative_papers(context, analysis, settings, review):
    validate_review(analysis, settings, review)
    corpus = context['papers']
    members = analysis['membership'].set_index('paper_id').loc[corpus.paper_id]
    labels = members.cluster.to_numpy()
    if (membership_id(corpus, labels) != review['full_membership_id']
            or context['input_id'] != review['citation_input_id']):
        raise ValueError('The reviewed citation graph or full membership (including isolates) changed.')
    assigned = members.loc[members.cluster.gt(0)]
    if (assigned.index.tolist() != analysis['papers'].paper_id.tolist()
            or not np.array_equal(assigned.cluster.to_numpy(), analysis['labels'])):
        raise ValueError('Assigned citation papers and full membership are not aligned.')

    # Recompute diagnostics from the reviewed graph, retaining all three isolates.
    network = context['network']
    adjacency, directed = network['adjacency'], network['directed']
    diagnostics = corpus[['paper_id', 'title', 'pdf_sha256', 'text_sha256', 'source_filename']].copy()
    diagnostics['cluster'] = labels
    diagnostics['assigned'] = labels > 0
    diagnostics['citation_degree'] = np.asarray(adjacency.sum(axis=1)).ravel().astype(int)
    diagnostics['incoming_corpus_citations'] = np.asarray(directed.sum(axis=0)).ravel().astype(int)
    diagnostics['within_community_degree'] = 0
    diagnostics['cluster_papers'] = pd.Series(pd.NA, index=diagnostics.index, dtype='Int64')
    diagnostics['internal_degree_rank'] = pd.Series(pd.NA, index=diagnostics.index, dtype='Int64')
    for cluster in sorted(set(labels) - {0}):
        indices = np.flatnonzero(labels == cluster)
        internal = np.asarray(adjacency[indices][:, indices].sum(axis=1)).ravel().astype(int)
        diagnostics.loc[indices, 'within_community_degree'] = internal
        diagnostics.loc[indices, 'cluster_papers'] = len(indices)
        # Equal degrees share a rank: important for the three-work practice-management component.
        diagnostics.loc[indices, 'internal_degree_rank'] = pd.Series(internal).rank(
            method='min', ascending=False).astype(int).to_numpy()
    diagnostics['external_degree'] = diagnostics.citation_degree - diagnostics.within_community_degree
    diagnostics['external_link_fraction'] = np.divide(
        diagnostics.external_degree, diagnostics.citation_degree,
        out=np.zeros(len(diagnostics)), where=diagnostics.citation_degree.gt(0))
    diagnostics['selected'] = diagnostics.paper_id.isin([p['paper_id'] for p in review['papers']])
    for column in ['citation_degree', 'within_community_degree', 'external_link_fraction']:
        np.testing.assert_allclose(diagnostics[column], members[column])
    selected, evidence = reviewed_evidence(analysis, review, diagnostics)
    return selected, evidence, diagnostics


def export_representative_papers(context, review_path, review, selected, evidence, diagnostics, settings):
    return export_review(context['root'], review_path, review, selected, evidence, diagnostics, settings,
        report_directory='reports/citation_cluster_examples',
        provenance={'citation_input_id': context['input_id'],
                    'full_membership_id': review['full_membership_id'],
                    'source_input_hashes': context['input_hashes'],
                    'citation_code_sha256': file_hash(Path(__file__)),
                    'diagnostic_scope': 'reviewed corpus graph, not global citation counts',
                    'unassigned_isolates': int((~diagnostics.assigned).sum())})
