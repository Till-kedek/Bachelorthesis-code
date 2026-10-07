"""Transfer existing PDF-text partitions to topic interpretation without fitting.

Only Clustering PDFs.ipynb exports fits. The topic notebook loads the snapshot,
validates its inputs and preserves the original K-means and embedding-Leiden IDs.
"""
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples
from sklearn.metrics.pairwise import cosine_distances

from .selection import filter_analysis_corpus, deduplicate_pdf_corpus
from .storage import file_hash, fingerprint, read_json, write_json
from .topic_analysis import load_topic_corpus


SOURCE_NOTEBOOK = 'Clustering PDFs.ipynb'
SOLUTIONS = {'kmeans_k5': 'K-means', 'embedding_leiden_k5': 'Embedding Leiden'}


def pdf_partition_input_id(keys, embeddings):
    """Bind a fit to the ordered PDF identities AND the exact vector values."""
    vectors = np.ascontiguousarray(embeddings)
    keys = list(keys)
    if (vectors.ndim != 2 or len(keys) != len(vectors)
            or len(set(keys)) != len(keys) or not np.isfinite(vectors).all()):
        raise ValueError('Expected unique PDF keys aligned with finite vectors')
    return fingerprint({'keys': keys, 'shape': list(vectors.shape),
                        'dtype': str(vectors.dtype),
                        'vectors_sha256': hashlib.sha256(vectors.tobytes()).hexdigest()})


def load_pdf_topic_corpus(input_dir, exclusion_file, *, deduplicate=True):
    """Use the PDF notebook's scope and duplicate filters, preserving original IDs."""
    papers, embeddings, passages, validation = load_topic_corpus(input_dir)
    papers, embeddings, audit, selection = filter_analysis_corpus(
        papers, embeddings, exclusion_file, key_column='pdf_sha256')
    # Citation interpretation validates the original scope against its graph,
    # then selects canonical works using the reviewed graph mapping.
    if deduplicate:
        papers, embeddings, selection = deduplicate_pdf_corpus(
            papers, embeddings, exclusion_file, selection, key_column="pdf_sha256")
    old_to_new = dict(zip(papers.row, range(len(papers))))
    passages = passages.loc[passages.row.isin(old_to_new)].copy()
    passages['row'] = passages.row.map(old_to_new)
    papers['row'] = np.arange(len(papers))
    return papers, embeddings, passages, audit, selection, validation


def _input_hashes(input_dir, exclusion_file):
    root = Path(input_dir)
    duplicate_file = Path(exclusion_file).with_name('pdf_duplicates.json')
    duplicate_hash = ({'pdf_duplicates.json': file_hash(duplicate_file)}
                      if duplicate_file.exists() else {})
    return {**duplicate_hash, **{name: file_hash(root / name) for name in (
        'embeddings/embeddings.npz', 'embeddings/manifest.json', 'prepared/manifest.json')},
        'analysis_exclusions.json': file_hash(exclusion_file)}


def _validate_labels(labels, n, solution):
    labels = np.asarray(labels)
    start = 0 if solution == 'kmeans_k5' else 1
    if (labels.shape != (n,) or labels.dtype.kind not in 'iu'
            or set(labels.tolist()) != set(range(start, start + 5))):
        raise ValueError(f'{solution} must contain exactly five original cluster IDs '
                         f'({start}–{start + 4}) aligned with every selected PDF')
    return labels


def _distance(embeddings):
    distance = cosine_distances(embeddings)
    distance = np.maximum((distance + distance.T) / 2, 0)
    np.fill_diagonal(distance, 0)
    return distance


def export_pdf_topic_partitions(destination, papers, embeddings, *, kmeans_result,
                                leiden_result, embedding_graph, fit_input_ids,
                                input_dir, exclusion_file, source_notebook,
                                settings):
    """Export the two existing five-group fits; never call a clustering algorithm.

    The source notebook passes kmeans_results[5] and its explicitly selected
    five-community leiden_results entry. Citation result objects are not inputs.
    Fit-time fingerprints reject stale arrays even if the new corpus has equal size.
    """
    current_id = pdf_partition_input_id(papers.Key, embeddings)
    if (set(fit_input_ids) != set(SOLUTIONS)
            or any(value != current_id for value in fit_input_ids.values())):
        raise ValueError('Stale PDF-text fit: rerun the source clustering on the current inputs')
    source_notebook = Path(source_notebook)
    if source_notebook.name != SOURCE_NOTEBOOK:
        raise ValueError(f'Export must originate in {SOURCE_NOTEBOOK}')
    selected, vectors, _, _, selection, _ = load_pdf_topic_corpus(input_dir, exclusion_file)
    if (papers.Key.tolist() != selected.pdf_sha256.tolist()
            or papers['Paper ID'].tolist() != selected.paper_id.tolist()
            or not np.array_equal(embeddings, vectors)):
        raise ValueError('Source fit does not use the current selected PDF corpus and vectors')

    km_labels = _validate_labels(kmeans_result['labels'], len(papers), 'kmeans_k5')
    model = kmeans_result['model']
    if (model.n_clusters != 5 or model.n_features_in_ != embeddings.shape[1]
            or not np.array_equal(model.labels_, km_labels)
            or not np.array_equal(model.predict(embeddings), km_labels)):
        raise ValueError('K-means labels do not match the fitted PDF embedding model')
    leiden_labels = _validate_labels(leiden_result['labels'], len(papers), 'embedding_leiden_k5')
    # A citation adjacency cannot masquerade as the source embedding graph.
    from .graph_clustering import similarity_graph
    expected_graph = similarity_graph(embeddings, neighbours=embedding_graph['neighbours'],
        mutual=embedding_graph['mutual'], min_similarity=embedding_graph['min_similarity'])
    for field in ('edges', 'weights', 'similarity'):
        if not np.array_equal(embedding_graph[field], expected_graph[field]):
            raise ValueError('Leiden graph must be the PDF embedding similarity graph')

    distance = _distance(embeddings)
    km_sil = silhouette_samples(distance, km_labels, metric='precomputed')
    if not np.allclose(kmeans_result['silhouettes'], km_sil, atol=1e-7, rtol=1e-6):
        raise ValueError('K-means silhouettes do not belong to these PDF assignments')
    specs = {
        'kmeans_k5': {'method': 'K-means', 'source_variable': 'kmeans_results[5]',
                     'seed': int(kmeans_result['seed']), 'inertia': float(model.inertia_),
                     'selection': 'minimum inertia across source-notebook seeds'},
        'embedding_leiden_k5': {'method': 'Embedding Leiden',
            'source_variable': f"leiden_results[{float(leiden_result['resolution'])!r}]",
            'seed': int(leiden_result['seed']), 'resolution': float(leiden_result['resolution']),
            'quality': float(leiden_result['quality']),
            'selection': 'explicit source resolution; highest objective seed at that resolution',
            'graph': {'kind': 'pdf_embedding_cosine_knn',
                      'neighbours': int(embedding_graph['neighbours']),
                      'mutual': bool(embedding_graph['mutual']),
                      'min_similarity': float(embedding_graph['min_similarity'])}},
    }
    rows = []
    for name, labels in [('kmeans_k5', km_labels), ('embedding_leiden_k5', leiden_labels)]:
        # K-means keeps its already computed scores; Leiden gets a diagnostic only.
        scores = (np.asarray(kmeans_result['silhouettes']) if name == 'kmeans_k5'
                  else silhouette_samples(distance, labels, metric='precomputed'))
        frame = pd.DataFrame({'solution': name, 'pdf_sha256': papers.Key.to_numpy(),
            'paper_id': papers['Paper ID'].to_numpy(), 'cluster': labels,
            'cosine_silhouette': scores})
        rows.append(frame)
        specs[name].update({'k': 5, 'papers': len(frame),
            'assignment_id': fingerprint(list(zip(frame.pdf_sha256, frame.cluster.astype(int))))})

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(destination / 'assignments.csv', index=False)
    notebook = json.loads(source_notebook.read_text())
    manifest = {'schema_version': 1, 'source_kind': 'pdf_text_embeddings',
        'source_notebook': SOURCE_NOTEBOOK,
        'source_code_id': fingerprint([c['source'] for c in notebook['cells'] if c['cell_type'] == 'code']),
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'input_id': current_id, 'selection': selection,
        'input_hashes': _input_hashes(input_dir, exclusion_file),
        'settings': settings, 'solutions': specs,
        'assignments_sha256': file_hash(destination / 'assignments.csv'),
        'packages': {name: version(name) for name in ('numpy', 'pandas', 'scikit-learn', 'igraph', 'leidenalg')}}
    write_json(destination / 'partition_manifest.json', manifest)
    return pd.concat(rows, ignore_index=True), manifest


def load_pdf_topic_partitions(destination, papers, embeddings, *, input_dir,
                              exclusion_file, selection):
    """Load exact source memberships by PDF hash; fail instead of fitting a fallback."""
    destination = Path(destination)
    if not all((destination / name).is_file() for name in ('partition_manifest.json', 'assignments.csv')):
        raise FileNotFoundError(f'No PDF-text partition export at {destination}. '
            f'Run the text clustering and five-cluster export cells in {SOURCE_NOTEBOOK} first. '
            'Topic analysis never creates replacement clusters.')
    manifest = read_json(destination / 'partition_manifest.json')
    if (manifest.get('schema_version') != 1
            or manifest.get('source_kind') != 'pdf_text_embeddings'
            or manifest.get('source_notebook') != SOURCE_NOTEBOOK
            or set(manifest.get('solutions', {})) != set(SOLUTIONS)):
        raise ValueError('Expected the two PDF-text five-cluster exports; citation partitions are not accepted')
    if (manifest['input_hashes'] != _input_hashes(input_dir, exclusion_file)
            or manifest['selection']['selection_id'] != selection['selection_id']
            or manifest['selection']['config_id'] != selection['config_id']
            or manifest['input_id'] != pdf_partition_input_id(papers.pdf_sha256, embeddings)):
        raise ValueError('PDF embeddings or scope changed since export; rerun the source notebook export')
    if file_hash(destination / 'assignments.csv') != manifest['assignments_sha256']:
        raise ValueError('PDF assignment file checksum mismatch')
    assignments = pd.read_csv(destination / 'assignments.csv',
                              dtype={'pdf_sha256': str, 'paper_id': str, 'solution': str})
    required = {'solution', 'pdf_sha256', 'paper_id', 'cluster', 'cosine_silhouette'}
    if not required <= set(assignments) or assignments[list(required)].isna().any().any():
        raise ValueError('Incomplete PDF-text assignment table')
    if set(assignments.solution) != set(SOLUTIONS):
        raise ValueError('Unexpected or missing PDF-text solution')
    distance = _distance(embeddings)
    solutions = {}
    for name, method in SOLUTIONS.items():
        frame = assignments.loc[assignments.solution.eq(name)]
        spec = manifest['solutions'][name]
        if (not frame.pdf_sha256.is_unique or set(frame.pdf_sha256) != set(papers.pdf_sha256)
                or spec['k'] != 5 or spec['papers'] != len(papers) or spec['method'] != method):
            raise ValueError(f'{name} does not cover exactly the selected PDFs with five clusters')
        frame = frame.set_index('pdf_sha256').loc[papers.pdf_sha256].reset_index()
        if frame.paper_id.tolist() != papers.paper_id.tolist():
            raise ValueError('Paper IDs do not match PDF hashes')
        labels = _validate_labels(frame.cluster.to_numpy(), len(papers), name)
        if fingerprint(list(zip(frame.pdf_sha256, labels.tolist()))) != spec['assignment_id']:
            raise ValueError('PDF membership fingerprint mismatch')
        scores = frame.cosine_silhouette.to_numpy(dtype=float)
        expected = silhouette_samples(distance, labels, metric='precomputed')
        if not np.isfinite(scores).all() or not np.allclose(scores, expected, atol=1e-7, rtol=1e-6):
            raise ValueError('Saved silhouettes do not match the source assignments')
        if name == 'embedding_leiden_k5' and spec.get('graph', {}).get('kind') != 'pdf_embedding_cosine_knn':
            raise ValueError('Only embedding-based Leiden is accepted')
        solutions[name] = {**spec, 'labels': labels, 'silhouettes': scores}
    return solutions, manifest
