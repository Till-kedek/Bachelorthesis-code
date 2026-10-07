"""Transfer existing abstract-text partitions to topic interpretation without fitting.

Only Clustering Abstracts.ipynb exports fits. The topic notebook loads the snapshot,
validates its inputs and preserves the original K-means and embedding-Leiden IDs.
"""
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples
from sklearn.metrics.pairwise import cosine_distances

from .storage import file_hash, fingerprint, read_json, write_json
from .abstract_clustering import load_abstract_embeddings


SOURCE_NOTEBOOK = 'Clustering Abstracts.ipynb'
SOLUTIONS = {'kmeans_k6': 'K-means', 'embedding_leiden_k6': 'Embedding Leiden'}


def _solution_spec(solution):
    match = re.fullmatch(r'(kmeans|embedding_leiden)_k(\d+)', solution)
    if not match or int(match[2]) < 2:
        raise ValueError(f'Invalid abstract solution: {solution}')
    return ('K-means' if match[1] == 'kmeans' else 'Embedding Leiden', int(match[2]))


def abstract_partition_input_id(keys, embeddings):
    """Bind a fit to the ordered abstract identities AND the exact vector values."""
    vectors = np.ascontiguousarray(embeddings)
    keys = list(keys)
    if (vectors.ndim != 2 or len(keys) != len(vectors)
            or len(set(keys)) != len(keys) or not np.isfinite(vectors).all()):
        raise ValueError('Expected unique abstract IDs aligned with finite vectors')
    return fingerprint({'keys': keys, 'shape': list(vectors.shape),
                        'dtype': str(vectors.dtype),
                        'vectors_sha256': hashlib.sha256(vectors.tobytes()).hexdigest()})


def load_abstract_topic_corpus(run_dir, source_csv):
    """Load the exact saved title-and-abstract inputs, with Scopus EIDs as keys."""
    papers, embeddings, provenance = load_abstract_embeddings(run_dir, source_csv)
    papers['text'] = papers['embedding_text']
    return papers, embeddings, provenance


def _validate_labels(labels, n, solution):
    labels = np.asarray(labels)
    method, k = _solution_spec(solution)
    start = 0 if method == 'K-means' else 1
    if (labels.shape != (n,) or labels.dtype.kind not in 'iu'
            or not k < n or set(labels.tolist()) != set(range(start, start + k))):
        raise ValueError(f'{solution} must contain exactly {k} original cluster IDs '
                         f'({start}–{start + k - 1}) aligned with every selected abstract')
    return labels


def _distance(embeddings):
    distance = cosine_distances(embeddings)
    distance = np.maximum((distance + distance.T) / 2, 0)
    np.fill_diagonal(distance, 0)
    return distance


def export_abstract_topic_partitions(destination, papers, embeddings, *, kmeans_result,
                                leiden_result, embedding_graph, fit_input_ids,
                                run_dir, source_csv, source_notebook,
                                settings):
    """Export the two selected fits with their actual counts; never fit clusters.

    The source notebook passes its selected K-means and Leiden results.
    Their counts may differ. Citation result objects are not inputs.
    Fit-time fingerprints reject stale arrays even if the new corpus has equal size.
    """
    km_k = int(kmeans_result['model'].n_clusters)
    leiden_k = len(np.unique(leiden_result['labels']))
    km_name, leiden_name = f'kmeans_k{km_k}', f'embedding_leiden_k{leiden_k}'
    current_id = abstract_partition_input_id(papers.Key, embeddings)
    if (set(fit_input_ids) != {km_name, leiden_name}
            or any(value != current_id for value in fit_input_ids.values())):
        raise ValueError('Stale abstract-text fit: rerun the source clustering on the current inputs')
    source_notebook = Path(source_notebook)
    if source_notebook.name != SOURCE_NOTEBOOK:
        raise ValueError(f'Export must originate in {SOURCE_NOTEBOOK}')
    selected, vectors, provenance = load_abstract_topic_corpus(run_dir, source_csv)
    if (papers.Key.tolist() != selected.paper_id.tolist()
            or papers['Paper ID'].tolist() != selected.paper_id.tolist()
            or not np.array_equal(embeddings, vectors)):
        raise ValueError('Source fit does not use the current selected abstract corpus and vectors')

    km_labels = _validate_labels(kmeans_result['labels'], len(papers), km_name)
    model = kmeans_result['model']
    if (model.n_features_in_ != embeddings.shape[1]
            or not np.array_equal(model.labels_, km_labels)
            or not np.array_equal(model.predict(embeddings), km_labels)):
        raise ValueError('K-means labels do not match the fitted abstract embedding model')
    leiden_labels = _validate_labels(leiden_result['labels'], len(papers), leiden_name)
    # A citation adjacency cannot masquerade as the source embedding graph.
    from .graph_clustering import similarity_graph
    expected_graph = similarity_graph(embeddings, neighbours=embedding_graph['neighbours'],
        mutual=embedding_graph['mutual'], min_similarity=embedding_graph['min_similarity'])
    for field in ('edges', 'weights', 'similarity'):
        if not np.array_equal(embedding_graph[field], expected_graph[field]):
            raise ValueError('Leiden graph must be the abstract embedding similarity graph')

    distance = _distance(embeddings)
    km_sil = silhouette_samples(distance, km_labels, metric='precomputed')
    if not np.allclose(kmeans_result['silhouettes'], km_sil, atol=1e-7, rtol=1e-6):
        raise ValueError('K-means silhouettes do not belong to these abstract assignments')
    specs = {
        km_name: {'method': 'K-means', 'source_variable': f'kmeans_results[{km_k}]',
                     'seed': int(kmeans_result['seed']), 'inertia': float(model.inertia_),
                     'selection': 'minimum inertia across source-notebook seeds'},
        leiden_name: {'method': 'Embedding Leiden',
            'source_variable': f"leiden_results[{float(leiden_result['resolution'])!r}]",
            'seed': int(leiden_result['seed']), 'resolution': float(leiden_result['resolution']),
            'quality': float(leiden_result['quality']),
            'selection': 'explicit source resolution; highest objective seed at that resolution',
            'graph': {'kind': 'abstract_embedding_cosine_knn',
                      'neighbours': int(embedding_graph['neighbours']),
                      'mutual': bool(embedding_graph['mutual']),
                      'min_similarity': float(embedding_graph['min_similarity'])}},
    }
    rows = []
    for name, labels in [(km_name, km_labels), (leiden_name, leiden_labels)]:
        # K-means keeps its already computed scores; Leiden gets a diagnostic only.
        scores = (np.asarray(kmeans_result['silhouettes']) if name == km_name
                  else silhouette_samples(distance, labels, metric='precomputed'))
        frame = pd.DataFrame({'solution': name, 'paper_id': papers['Paper ID'].to_numpy(), 'cluster': labels,
            'cosine_silhouette': scores})
        rows.append(frame)
        specs[name].update({'k': len(np.unique(labels)), 'papers': len(frame),
            'assignment_id': fingerprint(list(zip(frame.paper_id, frame.cluster.astype(int))))})

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(destination / 'assignments.csv', index=False)
    notebook = json.loads(source_notebook.read_text())
    manifest = {'schema_version': 1, 'source_kind': 'abstract_text_embeddings',
        'source_notebook': SOURCE_NOTEBOOK,
        'source_code_id': fingerprint([c['source'] for c in notebook['cells'] if c['cell_type'] == 'code']),
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'input_id': current_id, 'corpus_size': len(papers),
        'source_sha256': provenance['source_sha256'],
        'input_hashes': provenance['input_hashes'],
        'settings': settings, 'solutions': specs,
        'assignments_sha256': file_hash(destination / 'assignments.csv'),
        'packages': {name: version(name) for name in ('numpy', 'pandas', 'scikit-learn', 'igraph', 'leidenalg')}}
    write_json(destination / 'partition_manifest.json', manifest)
    return pd.concat(rows, ignore_index=True), manifest


def load_abstract_topic_partitions(destination, papers, embeddings, *, provenance):
    """Load exact source memberships by Scopus EID; fail instead of fitting a fallback."""
    destination = Path(destination)
    if not all((destination / name).is_file() for name in ('partition_manifest.json', 'assignments.csv')):
        raise FileNotFoundError(f'No abstract-text partition export at {destination}. '
            f'Run the text clustering and partition export cells in {SOURCE_NOTEBOOK} first. '
            'Topic analysis never creates replacement clusters.')
    manifest = read_json(destination / 'partition_manifest.json')
    solutions_in_manifest = manifest.get('solutions', {})
    methods = [_solution_spec(name)[0] for name in solutions_in_manifest]
    if (manifest.get('schema_version') != 1
            or manifest.get('source_kind') != 'abstract_text_embeddings'
            or manifest.get('source_notebook') != SOURCE_NOTEBOOK
            or sorted(methods) != ['Embedding Leiden', 'K-means']):
        raise ValueError('Expected two abstract-text partition exports; citation partitions are not accepted')
    if (manifest['input_hashes'] != provenance['input_hashes']
            or manifest['source_sha256'] != provenance['source_sha256']
            or manifest['corpus_size'] != len(papers)
            or manifest['input_id'] != abstract_partition_input_id(papers.paper_id, embeddings)):
        raise ValueError('abstract embeddings or scope changed since export; rerun the source notebook export')
    if file_hash(destination / 'assignments.csv') != manifest['assignments_sha256']:
        raise ValueError('abstract assignment file checksum mismatch')
    assignments = pd.read_csv(destination / 'assignments.csv',
                              dtype={'paper_id': str, 'solution': str})
    required = {'solution', 'paper_id', 'cluster', 'cosine_silhouette'}
    if not required <= set(assignments) or assignments[list(required)].isna().any().any():
        raise ValueError('Incomplete abstract-text assignment table')
    if set(assignments.solution) != set(solutions_in_manifest):
        raise ValueError('Unexpected or missing abstract-text solution')
    distance = _distance(embeddings)
    solutions = {}
    for name in solutions_in_manifest:
        method, k = _solution_spec(name)
        frame = assignments.loc[assignments.solution.eq(name)]
        spec = manifest['solutions'][name]
        if (not frame.paper_id.is_unique or set(frame.paper_id) != set(papers.paper_id)
                or spec['k'] != k or spec['papers'] != len(papers) or spec['method'] != method):
            raise ValueError(f'{name} does not cover exactly the selected abstracts with {k} clusters')
        frame = frame.set_index('paper_id').loc[papers.paper_id].reset_index()
        labels = _validate_labels(frame.cluster.to_numpy(), len(papers), name)
        if fingerprint(list(zip(frame.paper_id, labels.tolist()))) != spec['assignment_id']:
            raise ValueError('abstract membership fingerprint mismatch')
        scores = frame.cosine_silhouette.to_numpy(dtype=float)
        expected = silhouette_samples(distance, labels, metric='precomputed')
        if not np.isfinite(scores).all() or not np.allclose(scores, expected, atol=1e-7, rtol=1e-6):
            raise ValueError('Saved silhouettes do not match the source assignments')
        if method == 'Embedding Leiden' and spec.get('graph', {}).get('kind') != 'abstract_embedding_cosine_knn':
            raise ValueError('Only embedding-based Leiden is accepted')
        solutions[name] = {**spec, 'labels': labels, 'silhouettes': scores}
    return solutions, manifest


def describe_abstract_partition(papers, embeddings, result, terms, random_seed=42):
    """Describe saved assignments using actual abstract metadata and source text.

    Mean within-group cosine similarity is computed from a vector sum, avoiding
    another full pairwise matrix. Excerpt offsets refer to the cleaned abstract,
    never to a PDF page. Full abstracts remain in the review table for checking.
    """
    labels, sil = np.asarray(result['labels']), np.asarray(result['silhouettes'])
    if labels.shape != (len(papers),) or sil.shape != labels.shape:
        raise ValueError('Assignments must align with every abstract')
    if not papers.paper_id.is_unique or not np.isfinite(sil).all():
        raise ValueError('Expected unique abstract IDs and finite silhouettes')
    metadata = ['paper_id', 'title', 'doi', 'year', 'journal', 'document_type',
                'document_language', 'abstract_word_count']
    membership = papers[metadata].copy().reset_index(drop=True)
    membership['cluster'] = labels
    membership['cosine_silhouette'] = sil
    normalized = np.asarray(embeddings, dtype=float)
    normalized = normalized / np.linalg.norm(normalized, axis=1, keepdims=True)
    summaries, examples = [], []
    for cluster in sorted(np.unique(labels)):
        members = np.flatnonzero(labels == cluster)
        vectors = normalized[members]
        centrality = (vectors @ vectors.sum(axis=0) - 1) / max(1, len(members) - 1)
        central = members[np.argsort(-centrality, kind='stable')[:3]]
        boundary = members[np.argsort(sil[members], kind='stable')[:2]]
        rest = np.setdiff1d(members, np.union1d(central, boundary))
        random = np.random.default_rng(random_seed + int(cluster)).choice(
            rest, size=min(2, len(rest)), replace=False)
        selected_terms = terms.loc[terms.cluster.eq(cluster)]
        contrast = selected_terms.loc[selected_terms.ranking.eq('tfidf_contrast')].sort_values('rank').term.tolist()
        ctfidf = selected_terms.loc[selected_terms.ranking.eq('c_tf_idf')].sort_values('rank').term.tolist()
        # Bind interpretations to source text as well as IDs, not just group size.
        identity = sorted((papers.iloc[i].paper_id, fingerprint(papers.iloc[i].embedding_text))
                          for i in members)
        summaries.append({'cluster': int(cluster), 'papers': len(members),
            'mean_silhouette': sil[members].mean(),
            'negative_fraction': float(np.mean(sil[members] < 0)),
            'membership_id': fingerprint(identity),
            'ctfidf_terms': '; '.join(ctfidf[:10]), 'contrast_terms': '; '.join(contrast[:10]),
            'representatives': '\n'.join(f'{papers.iloc[i].paper_id}: {papers.iloc[i].title}' for i in central)})
        roles = {}
        for role, indices in [('representative', central), ('boundary', boundary), ('random', random)]:
            for index in indices:
                roles.setdefault(int(index), []).append(role)
        for index, role in roles.items():
            abstract = papers.iloc[index].abstract
            matched_term, begin = '', 0
            for term in contrast[:10]:
                pattern = r'(?<!\w)' + r'\s+'.join(map(re.escape, term.split())) + r'(?!\w)'
                match = re.search(pattern, abstract, re.I)
                if match:
                    matched_term, begin = term, max(0, match.start() - 130)
                    break
            end = min(len(abstract), begin + 600)
            examples.append({**membership.iloc[index].to_dict(), 'selection': '; '.join(role),
                'source_field': 'abstract', 'term': matched_term,
                'excerpt_start': begin, 'excerpt_end': end,
                'excerpt': abstract[begin:end], 'abstract': abstract})
    return pd.DataFrame(summaries), membership, pd.DataFrame(examples)


def write_abstract_review_cards(profiles, term_scores, examples, proposals, destination):
    """Attach recorded proposals only to matching methods, IDs AND source texts."""
    required = ['solution', 'cluster', 'membership_id', 'proposed_label', 'status']
    if not set(required) <= set(proposals):
        raise ValueError('Incomplete abstract label proposals')
    if proposals.proposed_label.isna().any() or proposals.proposed_label.astype(str).str.strip().eq('').any():
        raise ValueError('Each abstract label proposal needs a nonempty name')
    columns = required + [c for c in ('rationale', 'review_caveat') if c in proposals]
    joined = profiles.merge(proposals[columns], on=['solution', 'cluster', 'membership_id'],
                            how='left', validate='one_to_one')
    joined['proposal_matched'] = joined.proposed_label.notna()
    joined.loc[~joined.proposal_matched, 'proposed_label'] = 'Name pending review'
    joined.loc[~joined.proposal_matched, 'status'] = 'Review these memberships and abstracts before naming'
    lines = ['# Abstract cluster review cards', '',
        'AI-assisted analyst proposals, not automatically generated or user-validated labels. '
        'Every name is bound to the saved memberships and source texts. Full sampled abstracts '
        'and excerpt offsets are in review_papers.csv.', '']
    for _, row in joined.iterrows():
        display_id = f'L{row.cluster}' if row.solution.startswith('embedding_leiden_') else str(row.cluster)
        lines += [f'## {row.solution}, cluster {display_id}: {row.proposed_label}', '',
            f'{row.papers} abstracts; mean silhouette {row.mean_silhouette:.3f}; '
            f'negative silhouettes {row.negative_fraction:.1%}. {row.status}.', '',
            f'c-TF-IDF: {row.ctfidf_terms}', '', f'TF-IDF contrast: {row.contrast_terms}', '']
        for column in ('rationale', 'review_caveat'):
            if pd.notna(row.get(column)) and str(row[column]).strip():
                lines += [str(row[column]), '']
        subset = term_scores.loc[(term_scores.solution == row.solution)
            & (term_scores.cluster == row.cluster) & (term_scores.ranking == 'tfidf_contrast')].head(8)
        lines += ['| Term | Abstracts inside | Abstracts outside |', '|---|---:|---:|']
        lines += [f'| {t.term} | {t.inside_count}/{t.inside_n} | {t.outside_count}/{t.outside_n} |'
                  for _, t in subset.iterrows()]
        lines += ['', 'Representative papers:', '']
        lines += [f'- {title}' for title in row.representatives.splitlines()]
        subset = examples.loc[(examples.solution == row.solution)
            & (examples.cluster == row.cluster) & examples.selection.str.contains('boundary')]
        lines += ['', 'Boundary papers:', '']
        lines += [f'- {p.paper_id}: {p.title} (silhouette {p.cosine_silhouette:.3f})'
                  for _, p in subset.iterrows()]
        lines += ['', '']
    Path(destination).write_text('\n'.join(lines), encoding='utf-8')
    return joined
