"""Reviewed examples for elite-paper abstract clusters, with explicit text provenance."""

from pathlib import Path

import numpy as np
import pandas as pd

from .pdf_cluster_examples import export_representative_papers as export_review, validate_review
from .storage import file_hash, fingerprint, read_json


def representative_papers(context, analysis, settings, review):
    normalized_settings = {**settings, 'source': 'saved', 'saved': settings.get('saved_partition')}
    validate_review(analysis, normalized_settings, review)
    if (context['source_manifest']['input_id'] != review['embedding_input_id']
            or context['input_hashes']['catalog_manifest.json'] != review['catalog_manifest_sha256']):
        raise ValueError('The reviewed elite abstract vectors or partition catalog changed.')
    papers = analysis['papers'].reset_index(drop=True)
    labels = np.asarray(analysis['labels'])
    if (papers.paper_id.tolist() != context['papers'].paper_id.tolist()
            or not np.array_equal(labels, context['solutions'][review['solution']]['labels'])):
        raise ValueError('Examples must use the exact saved elite abstract memberships and order.')

    diagnostics = papers[['paper_id', 'title', 'pdf_sha256', 'source_filename', 'abstract_source_kind']].copy()
    diagnostics['cluster'] = labels
    diagnostics['embedding_text_sha256'] = papers.embedding_text.map(fingerprint)
    membership = analysis['membership'].set_index('paper_id')
    diagnostics['cosine_silhouette'] = diagnostics.paper_id.map(membership.cosine_silhouette)
    vectors = np.array(analysis['embeddings'], dtype=float, copy=True)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    similarity = vectors @ vectors.T
    for cluster in sorted(set(labels)):
        indices = np.flatnonzero(labels == cluster)
        within = similarity[np.ix_(indices, indices)]
        diagnostics.loc[indices, 'mean_within_cluster_cosine'] = (
            within.sum(axis=1) - np.diag(within)) / (len(indices) - 1)
        diagnostics.loc[indices, 'cluster_papers'] = len(indices)
    diagnostics['cluster_papers'] = diagnostics.cluster_papers.astype(int)
    diagnostics['centrality_rank'] = diagnostics.groupby('cluster').mean_within_cluster_cosine.rank(
        ascending=False, method='min').astype(int)
    diagnostics['selected'] = diagnostics.paper_id.isin([p['paper_id'] for p in review['papers']])

    selected, evidence = [], []
    root = Path(context['root'])
    for choice in review['papers']:
        rows = papers.loc[papers.paper_id.eq(choice['paper_id'])]
        if len(rows) != 1:
            raise ValueError('A reviewed abstract is missing or duplicated.')
        paper = rows.iloc[0]
        diagnostic = diagnostics.loc[diagnostics.paper_id.eq(paper.paper_id)].iloc[0].to_dict()
        if (diagnostic['cluster'] != choice['cluster'] or paper.pdf_sha256 != choice['pdf_sha256']
                or fingerprint(paper.embedding_text) != choice['embedding_text_sha256']
                or fingerprint(paper.abstract) != choice['abstract_sha256']):
            raise ValueError(f'Reviewed abstract text or membership changed: {paper.paper_id}')
        for field in ['abstract_source_kind', 'abstract_source_path', 'abstract_source_sha256',
                      'abstract_source_url', 'abstract_source_record_id', 'abstract_spans']:
            if paper[field] != choice[field]:
                raise ValueError(f'Reviewed abstract provenance changed: {paper.paper_id}')
        if (paper.abstract_source_kind not in {'pdf_abstract', 'external_full_abstract'}
                or str(paper.is_short_description).strip().lower() not in {'false', '0'}):
            raise ValueError('Only included full abstracts can provide representative evidence.')
        if file_hash(root / paper.abstract_source_path) != paper.abstract_source_sha256:
            raise ValueError(f'Cached abstract source changed: {paper.paper_id}')
        source = read_json(root / paper.abstract_source_path)
        if source['pdf_sha256'] != paper.pdf_sha256:
            raise ValueError('Abstract source belongs to a different paper.')
        if paper.abstract_source_kind == 'external_full_abstract' and paper.abstract_spans != '[]':
            raise ValueError('External abstracts cannot claim PDF abstract spans.')
        if len(choice['evidence']) != 2:
            raise ValueError('Each reviewed paper requires two supporting passages.')
        for number, quote in enumerate(choice['evidence'], 1):
            if quote['source_field'] == 'abstract':
                text = paper.abstract
                page, block_id = None, None
                source_path, source_hash = paper.abstract_source_path, paper.abstract_source_sha256
                kind = paper.abstract_source_kind
                spans = paper.abstract_spans
            elif quote['source_field'] == 'pdf_context':
                source_path, source_hash = choice['document_path'], choice['document_sha256']
                if (paper.document_path != source_path or paper.document_sha256 != source_hash
                        or file_hash(root / source_path) != source_hash):
                    raise ValueError(f'Supplementary PDF source changed: {paper.paper_id}')
                document = read_json(root / source_path)
                if document['pdf_sha256'] != paper.pdf_sha256:
                    raise ValueError('Supplementary PDF belongs to a different paper.')
                blocks = [b for b in document['blocks']
                          if b['id'] == quote['block_id'] and b['page'] == quote['page']]
                if len(blocks) != 1:
                    raise ValueError('Missing or ambiguous supplementary PDF block.')
                text = blocks[0]['text']
                page, block_id = quote['page'], quote['block_id']
                kind, spans = 'supplementary_pdf_context', '[]'
            else:
                raise ValueError('Unknown evidence source field.')
            start, end = quote['excerpt_start'], quote['excerpt_end']
            if not 0 <= start < end <= len(text) or text[start:end] != quote['excerpt']:
                raise ValueError(f'Reviewed passage no longer matches source: {paper.paper_id}')
            evidence.append({'paper_id': paper.paper_id, 'cluster': choice['cluster'],
                'evidence_number': number, **quote, 'page': page, 'block_id': block_id,
                'source_kind': kind, 'source_path': source_path, 'source_sha256': source_hash,
                'abstract_spans': spans,
                'source_url': paper.abstract_source_url if quote['source_field'] == 'abstract' else ''})
        selected.append({**diagnostic, **{k: v for k, v in choice.items() if k != 'evidence'},
                         'requested_focus': review['focus'][str(choice['cluster'])],
                         'abstract': paper.abstract, 'doi': paper.doi})
    return pd.DataFrame(selected), pd.DataFrame(evidence), diagnostics


def export_representative_papers(context, review_path, review, selected, evidence, diagnostics, settings):
    return export_review(context['root'], review_path, review, selected, evidence, diagnostics, settings,
        report_directory='reports/elite_abstract_cluster_examples',
        provenance={'embedding_input_id': review['embedding_input_id'],
                    'source_input_hashes': context['input_hashes'],
                    'elite_code_sha256': file_hash(Path(__file__)),
                    'diagnostic_representation': 'saved elite-paper title-and-abstract vectors',
                    'supplementary_pdf_text_used_for_clustering': False})
