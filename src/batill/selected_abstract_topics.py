"""Exact clustering handoff and topic evidence for selected-PDF abstract vectors."""
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples
from sklearn.metrics.pairwise import cosine_distances

from .abstract_topic_analysis import abstract_partition_input_id
from .abstract_topic_explorer import (
    analyse_partition, evidence_table, export_analysis, ranked_terms,
    search_expression, supporting_papers, partition_catalog,
)
from .selected_abstracts import load_selected_embeddings, resolve_selected_run
from .storage import file_hash, fingerprint, read_json, write_json


CATALOG_CONFIG = 'configs/selected_abstracts_partitions.json'


def notebook_source_hash(path):
    """Execution counts and saved cell outputs must not change a fit's identity."""
    cells = read_json(path).get('cells', [])
    return fingerprint([{'cell_type': c['cell_type'], 'source': ''.join(c['source'])
                        if isinstance(c['source'], list) else c['source']} for c in cells])


def export_catalog(root, papers, embeddings, *, run_dir, source_csv,
                   kmeans_results, leiden_results, fit_input_ids, graph,
                   kmeans_diagnostics, leiden_diagnostics, settings,
                   selected_k=None, selected_resolution=None,
                   source_notebook='Clustering Elite Papers Abstracts.ipynb'):
    """Persist all existing sweep winners, never refit or transfer old topic labels."""
    root = Path(root).resolve()
    current_id = abstract_partition_input_id(papers.Key, embeddings)
    if fit_input_ids != {'kmeans': current_id, 'leiden': current_id}:
        raise ValueError('Stale clustering results: rerun both sweeps for the current vectors')
    if (set(kmeans_results) != set(settings['k_values'])
            or set(leiden_results) != set(settings['leiden_resolutions'])
            or set(kmeans_diagnostics.k) != set(kmeans_results)
            or set(leiden_diagnostics.resolution) != set(leiden_results)):
        raise ValueError('Saved results must cover both complete configured sweeps')
    loaded, vectors, provenance = load_selected_embeddings(run_dir, source_csv)
    if (papers.Key.tolist() != loaded.Key.tolist()
            or papers.paper_id.tolist() != loaded.paper_id.tolist()
            or not np.array_equal(vectors, embeddings)):
        raise ValueError('Clustering does not match the authenticated selected abstract inputs')
    from .graph_clustering import similarity_graph
    expected_graph = similarity_graph(vectors, neighbours=graph['neighbours'],
                                      min_similarity=graph['min_similarity'], mutual=graph['mutual'])
    if any(not np.array_equal(graph[field], expected_graph[field]) for field in ('edges', 'weights', 'similarity')):
        raise ValueError('Leiden graph does not belong to the current abstract vectors')
    # The notebook reuses the hierarchy's float64 cosine distances. Recomputing
    # in float32 can shift silhouettes enough to falsely reject the same fit.
    distance = cosine_distances(np.asarray(vectors, dtype=np.float64))
    distance = np.maximum((distance + distance.T) / 2, 0)
    np.fill_diagonal(distance, 0)
    solutions, arrays = {}, {'paper_ids': papers.paper_id.to_numpy(dtype=str),
                             'pdf_sha256': papers.Key.to_numpy(dtype=str)}

    def retain(name, labels, spec, expected_start):
        labels = np.asarray(labels)
        count = len(np.unique(labels))
        if (labels.shape != (len(papers),) or labels.dtype.kind not in 'iu'
                or set(labels) != set(range(expected_start, expected_start + count))):
            raise ValueError(f'Invalid aligned labels for {name}')
        scores = (silhouette_samples(distance, labels, metric='precomputed')
                  if 1 < count < len(papers) else np.full(len(papers), np.nan))
        arrays[name] = labels
        arrays[name + '__silhouettes'] = scores
        solutions[name] = {**spec, 'clusters': count,
                           'assignment_id': fingerprint(list(zip(papers.Key, map(int, labels))))}
        return scores

    for k, result in kmeans_results.items():
        model, labels = result['model'], result['labels']
        if (int(k) != model.n_clusters or model.n_features_in_ != embeddings.shape[1]
                or not np.array_equal(labels, model.labels_)
                or not np.array_equal(labels, model.predict(embeddings))):
            raise ValueError(f'K-means k={k} labels do not match the fitted model')
        scores = retain(f'kmeans_k{k}', labels, {
            'method': 'K-means', 'seed': int(result['seed']),
            'inertia': float(model.inertia_), 'seed_ARI': float(np.mean(result['seed_aris'])),
            'selection': f"lowest inertia across {len(settings['kmeans_seeds'])} seeds, {settings['n_init']} initializations per seed",
        }, 0)
        if not np.allclose(scores, result['silhouettes'], atol=1e-6):
            raise ValueError('K-means silhouettes differ from the exported inputs')
    for resolution, result in leiden_results.items():
        diagnostics = leiden_diagnostics.loc[np.isclose(leiden_diagnostics.resolution, resolution, rtol=0, atol=1e-12)]
        if len(diagnostics) != 1:
            raise ValueError('Leiden result has no unique diagnostic row')
        retain(f'leiden_r{resolution:g}', result['labels'], {
            'method': 'Embedding Leiden', 'seed': int(result['seed']),
            'resolution': float(resolution), 'quality': float(result['quality']),
            'seed_ARI': float(diagnostics.iloc[0].seed_stability_ARI),
            'selection': 'highest objective seed within this resolution',
        }, 1)
    if not kmeans_results or not leiden_results:
        raise ValueError('Both complete sweeps are required for the topic catalog')
    preferred = []
    if selected_k is not None:
        preferred.append(f'kmeans_k{selected_k}')
    if selected_resolution is not None:
        preferred.append(f'leiden_r{selected_resolution:g}')
    if not set(preferred) <= set(solutions):
        raise ValueError('Select only a partition already fitted in the sweeps')
    if any(not 2 <= solutions[name]['clusters'] < len(papers) for name in preferred):
        raise ValueError('Topic interpretation requires 2 to n-1 clusters')
    packages = {name: version(name) for name in ('numpy', 'pandas', 'scikit-learn', 'igraph', 'leidenalg')}
    spec = {'schema_version': 1, 'input_id': current_id, 'settings': settings,
            'solutions': solutions, 'packages': packages,
            'source_notebook': source_notebook,
            'source_notebook_source_sha256': notebook_source_hash(root / source_notebook),
            'reference_clustering_notebook_sha256': file_hash(root / 'Clustering Abstracts.ipynb'),
            'code_sha256': {name: file_hash(Path(__file__).with_name(name)) for name in
                           ('selected_abstract_topics.py', 'graph_clustering.py', 'clustering.py',
                            'abstract_topic_explorer.py', 'topic_explorer.py', 'topic_analysis.py')},
            'representation_id': provenance['representation_id'],
            'run_dir': str(Path(run_dir).resolve().relative_to(root)),
            'source_csv': str(Path(source_csv).resolve().relative_to(root)),
            'run_summary_sha256': file_hash(Path(run_dir) / 'run_summary.json'),
            'input_hashes': provenance['input_hashes']}
    identity = fingerprint(spec)
    folder = root / 'outputs/selected_abstracts/clustering' / current_id[:20] / identity[:20]
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / 'catalog_manifest.json'
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if any(manifest.get(key) != value for key, value in spec.items()):
            raise ValueError('Existing catalog has different settings or memberships')
        for name, checksum in manifest['artifact_sha256'].items():
            if file_hash(folder / name) != checksum:
                raise ValueError(f'Existing catalog artifact changed: {name}')
    else:
        np.savez_compressed(folder / 'partitions.npz', **arrays)
        kmeans_diagnostics.to_csv(folder / 'kmeans_diagnostics.csv', index=False)
        leiden_diagnostics.to_csv(folder / 'leiden_diagnostics.csv', index=False)
        papers[['paper_id', 'Key', 'title', 'doi', 'year']].to_csv(folder / 'papers.csv', index=False)
        manifest = {**spec, 'artifact_sha256': {name: file_hash(folder / name) for name in
                     ('partitions.npz', 'kmeans_diagnostics.csv', 'leiden_diagnostics.csv', 'papers.csv')}}
        write_json(manifest_path, manifest)
    write_json(root / CATALOG_CONFIG, {'schema_version': 1, 'input_id': current_id,
        'catalog_dir': str(folder.relative_to(root)), 'preferred_partitions': preferred,
        'catalog_manifest_sha256': file_hash(manifest_path)})
    return folder, manifest


def load_context(root):
    """Load authenticated saved memberships; never fit an implicit replacement."""
    root = Path(root).resolve()
    run, source = resolve_selected_run(root)
    papers, embeddings, provenance = load_selected_embeddings(run, source)
    path = root / CATALOG_CONFIG
    if not path.exists():
        raise FileNotFoundError('Run Clustering Elite Papers Abstracts.ipynb and its export cell first.')
    config = read_json(path)
    folder = root / config['catalog_dir']
    manifest_path = folder / 'catalog_manifest.json'
    manifest = read_json(manifest_path)
    input_id = abstract_partition_input_id(papers.Key, embeddings)
    if (config['input_id'] != input_id or manifest['input_id'] != input_id
            or file_hash(manifest_path) != config['catalog_manifest_sha256']
            or provenance['representation_id'] != manifest['representation_id']
            or file_hash(run / 'run_summary.json') != manifest['run_summary_sha256']
            or provenance['input_hashes'] != manifest['input_hashes']):
        raise ValueError('Topic catalog is stale or belongs to different abstract inputs')
    for name, checksum in manifest['artifact_sha256'].items():
        if file_hash(folder / name) != checksum:
            raise ValueError(f'Topic catalog checksum mismatch: {name}')
    solutions = {}
    with np.load(folder / 'partitions.npz', allow_pickle=False) as stored:
        if (stored['paper_ids'].tolist() != papers.paper_id.tolist()
                or stored['pdf_sha256'].tolist() != papers.Key.tolist()):
            raise ValueError('Catalog paper identity/order differs from the embedding rows')
        for name, spec in manifest['solutions'].items():
            labels = stored[name].copy()
            scores = stored[name + '__silhouettes'].copy()
            if (labels.shape != (len(papers),) or scores.shape != labels.shape
                    or labels.dtype.kind not in 'iu' or (labels < 0).any()
                    or len(np.unique(labels)) != spec['clusters']
                    or fingerprint(list(zip(papers.Key, map(int, labels)))) != spec['assignment_id']):
                raise ValueError(f'Invalid memberships in {name}')
            solutions[name] = {**spec, 'labels': labels, 'silhouettes': scores}
    passages = pd.DataFrame([{'row': i, 'page': None, 'block_id': field, 'text': getattr(paper, field)}
                             for i, paper in enumerate(papers.itertuples()) for field in ('title', 'abstract')])
    return {'root': root, 'evidence': folder, 'papers': papers, 'embeddings': embeddings,
            'passages': passages, 'solutions': solutions, 'source_manifest': manifest,
            'preferred_partitions': config['preferred_partitions'],
            'input_hashes': {**provenance['input_hashes'], 'catalog_manifest.json': file_hash(manifest_path)},
            'partition_file': folder / 'partitions.npz',
            'export_directory': 'outputs/selected_abstracts/topic_analysis'}


def choose_partition(context, saved):
    if saved not in context['solutions']:
        raise ValueError('Choose a partition from the saved catalog')
    result = context['solutions'][saved]
    if not 2 <= result['clusters'] < len(context['papers']):
        raise ValueError('This resolution has no interpretable multi-cluster partition; choose another')
    return result['labels'].copy(), {key: value for key, value in
                                   {'saved_partition': saved, **result}.items()
                                   if key not in ('labels', 'silhouettes')}
