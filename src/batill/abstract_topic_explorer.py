"""Interactive expression evidence for saved or explicitly fitted abstract clusters."""

from pathlib import Path
from itertools import combinations
from importlib.metadata import version

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_samples
from sklearn.metrics.pairwise import cosine_distances
from threadpoolctl import threadpool_limits

from .abstract_topic_analysis import (
    abstract_partition_input_id, describe_abstract_partition,
    load_abstract_topic_corpus,
)
from .graph_clustering import leiden_sweep, similarity_graph
from .abstract_scope import resolve_abstract_inputs
from .storage import file_hash, fingerprint, read_json, write_json
from .topic_analysis import build_vocabulary, topic_terms
from .topic_explorer import (
    choose_partition as choose_text_partition, evidence_table, export_analysis,
    ranked_terms, search_expression, supporting_papers as source_supporting_papers,
)


def _load_catalog(root, papers, embeddings, provenance, input_id, *, build_missing=True):
    """Cache actual memberships for every setting in the authenticated source sweeps."""
    report = root / 'reports/abstract_clustering' / input_id[:16]
    manifests = {}
    hashes = dict(provenance['input_hashes'])
    for method, manifest_name, csv_name in (
        ('kmeans', 'kmeans_sweep_manifest.json', 'kmeans_diagnostics.csv'),
        ('leiden', 'sweep_manifest.json', 'leiden_resolution_diagnostics.csv'),
    ):
        spec = read_json(report / manifest_name)
        if (spec['input_id'] != input_id or spec['papers'] != len(papers)
                or spec['diagnostics_sha256'] != file_hash(report / csv_name)):
            raise ValueError(f'{method} sweep does not match the current abstract inputs')
        manifests[method] = spec
        hashes[manifest_name] = file_hash(report / manifest_name)
        hashes[csv_name] = file_hash(report / csv_name)
    packages = {name: version(name) for name in
                ('numpy', 'scikit-learn', 'igraph', 'leidenalg')}
    identity = {'input_id': input_id, 'input_hashes': hashes,
                'sweeps': manifests, 'packages': packages, 'schema_version': 1}
    evidence = (root / 'outputs/abstract_text_clusters' / input_id[:16]
                / 'explorer_catalog' / fingerprint(identity)[:16])
    manifest_path = evidence / 'catalog_manifest.json'
    partition_file = evidence / 'partitions.npz'
    if not manifest_path.exists():
        if not build_missing:
            raise FileNotFoundError('Saved abstract catalog is missing. Run Topic_analysis_Abstracts_v2.ipynb first.')
        print('Building abstract partition catalog from the saved sweep settings; '
              'subsequent loads reuse these memberships.', flush=True)
        solutions = _fit_catalog(embeddings, manifests)
        evidence.mkdir(parents=True, exist_ok=True)
        arrays = {'paper_ids': papers.paper_id.to_numpy(dtype=str)}
        specs = {}
        for name, result in solutions.items():
            arrays[name] = result['labels']
            arrays[name + '__silhouettes'] = result['silhouettes']
            specs[name] = {key: value for key, value in result.items()
                           if key not in ('labels', 'silhouettes')}
        np.savez_compressed(partition_file, **arrays)
        write_json(manifest_path, {**identity, 'solutions': specs,
                                  'partitions_sha256': file_hash(partition_file)})
    manifest = read_json(manifest_path)
    if (any(manifest.get(key) != value for key, value in identity.items())
            or file_hash(partition_file) != manifest['partitions_sha256']):
        raise ValueError('Abstract partition catalog checksum or provenance mismatch')
    solutions = {}
    with np.load(partition_file, allow_pickle=False) as archive:
        if archive['paper_ids'].tolist() != papers.paper_id.tolist():
            raise ValueError('Abstract catalog paper order changed')
        for name, spec in manifest['solutions'].items():
            labels, scores = archive[name], archive[name + '__silhouettes']
            start = 0 if spec['method'] == 'K-means' else 1
            if (labels.shape != (len(papers),) or labels.dtype.kind not in 'iu'
                    or set(labels) != set(range(start, start + spec['clusters']))
                    or scores.shape != labels.shape or not np.isfinite(scores).all()):
                raise ValueError(f'Invalid abstract catalog partition: {name}')
            solutions[name] = {**spec, 'labels': labels.copy(), 'silhouettes': scores.copy()}
    return evidence, solutions, manifest, hashes


def _fit_catalog(embeddings, manifests):
    """Reproduce source selection rules without rerunning subsample diagnostics."""
    distance = cosine_distances(embeddings)
    distance = np.maximum((distance + distance.T) / 2, 0)
    np.fill_diagonal(distance, 0)
    solutions = {}
    spec = manifests['kmeans']
    with threadpool_limits(limits=1):
        for k in spec['k_values']:
            runs = []
            for seed in spec['seeds']:
                model = KMeans(n_clusters=k, random_state=seed, n_init=spec['n_init'])
                labels = model.fit_predict(embeddings)
                runs.append((float(model.inertia_), seed, labels))
            inertia, seed, labels = min(runs, key=lambda run: run[0])
            solutions[f'kmeans_k{k}'] = {
                'method': 'K-means', 'clusters': int(k), 'seed': int(seed),
                'seeds': spec['seeds'], 'n_init': spec['n_init'], 'inertia': inertia,
                'selection': 'minimum inertia across source-notebook seeds',
                'seed_ARI': float(np.mean([adjusted_rand_score(a[2], b[2])
                                          for a, b in combinations(runs, 2)])),
                'labels': labels, 'silhouettes': silhouette_samples(
                    distance, labels, metric='precomputed')}
            print(f'Catalog: K-means k={k}', flush=True)
        spec = manifests['leiden']
        graph = similarity_graph(embeddings, neighbours=spec['neighbours'],
                                 mutual=spec['mutual'], min_similarity=spec['min_similarity'])
        diagnostics, results = leiden_sweep(graph, spec['resolutions'], seeds=spec['seeds'])
        for row in diagnostics.itertuples():
            result = results[row.resolution]
            solutions[f'leiden_r{row.resolution:g}'] = {
                'method': 'Embedding Leiden', 'clusters': int(row.clusters),
                'resolution': float(row.resolution), 'seed': int(result['seed']),
                'seeds': spec['seeds'], 'neighbours': spec['neighbours'],
                'mutual': spec['mutual'], 'min_similarity': spec['min_similarity'],
                'quality': float(result['quality']),
                'selection': 'highest objective across seeds at this resolution',
                'seed_ARI': float(row.seed_stability_ARI), 'labels': result['labels'],
                'silhouettes': silhouette_samples(distance, result['labels'], metric='precomputed')}
        print(f"Catalog: {len(results)} Leiden resolutions, including 0.45", flush=True)
    return solutions


def load_context(root, *, build_missing=True, use_original=False, input_paths=None):
    """Authenticate abstracts and load the complete cached clustering catalog."""
    root = Path(root).resolve()
    if input_paths is not None and use_original:
        raise ValueError('Explicit abstract inputs cannot be combined with use_original')
    run, source = (input_paths if input_paths is not None
                   else resolve_abstract_inputs(root, use_original=use_original))
    papers, embeddings, provenance = load_abstract_topic_corpus(run, source)
    input_id = abstract_partition_input_id(papers.paper_id, embeddings)
    evidence, solutions, manifest, hashes = _load_catalog(
        root, papers, embeddings, provenance, input_id, build_missing=build_missing)
    # Separate title and abstract so manual phrases never span these fields.
    passages = pd.DataFrame([
        {'row': i, 'page': None, 'block_id': field, 'text': getattr(paper, field)}
        for i, paper in enumerate(papers.itertuples(index=False))
        for field in ('title', 'abstract')
    ])
    return {'root': root, 'evidence': evidence, 'papers': papers,
            'embeddings': embeddings, 'passages': passages, 'solutions': solutions,
            'input_hashes': {**hashes,
                             'catalog_manifest.json': file_hash(evidence / 'catalog_manifest.json')},
            'partition_file': evidence / 'partitions.npz',
            'export_directory': 'outputs/topic_analysis_abstracts_v2',
            'input_id': input_id, 'source_manifest': manifest}


def partition_catalog(context):
    """List exact saved memberships, including the valid K-means cluster zero."""
    return pd.DataFrame([
        {'solution': name, 'method': result['method'],
         'clusters': len(np.unique(result['labels'])), 'assigned': len(result['labels']),
         'unassigned': 0,
         'cosine_silhouette': float(np.mean(result['silhouettes'])),
         'seed_ARI': result.get('seed_ARI', np.nan),
         'resolution': result.get('resolution', np.nan), 'seed': result['seed'],
         }
        for name, result in context['solutions'].items()
    ])


def choose_partition(context, *, source='saved', saved='leiden_r0.45', method='leiden',
                     k=5, resolution=0.45, neighbours=15, seed=42, n_init=10,
                     leiden_seeds=(42, 43, 44)):
    """Preserve saved IDs or fit one setting; match the source Leiden seed rule."""
    if source == 'saved':
        if saved not in context['solutions']:
            raise ValueError(f'Unknown saved partition {saved!r}; inspect the catalog')
        result = context['solutions'][saved]
        settings = {key: value for key, value in result.items()
                    if key not in ('labels', 'silhouettes')}
        settings.update(source='saved', saved=saved, clusters=len(np.unique(result['labels'])))
        return result['labels'].copy(), settings
    if source != 'fit':
        raise ValueError("SOURCE must be 'saved' or 'fit'")
    if method == 'kmeans':
        labels, settings = choose_text_partition(
            context, source='fit', method=method, k=k, seed=seed, n_init=n_init)
        # Source abstract K-means uses 0-based IDs; zero is an assigned cluster.
        return labels - 1, settings
    if method != 'leiden':
        raise ValueError("METHOD must be 'kmeans' or 'leiden'")
    graph = similarity_graph(context['embeddings'], neighbours=neighbours,
                             min_similarity=0.0, mutual=False)
    diagnostics, results = leiden_sweep(graph, [resolution], seeds=leiden_seeds)
    resolution = round(float(resolution), 12)
    result = results[resolution]
    settings = {'source': 'fit', 'method': 'Embedding Leiden',
                'resolution': float(resolution), 'neighbours': int(graph['neighbours']),
                'mutual': False, 'min_similarity': 0.0,
                'seeds': list(map(int, leiden_seeds)), 'seed': int(result['seed']),
                'selection': 'highest objective across seeds at this resolution',
                'quality': float(result['quality']),
                'clusters': len(np.unique(result['labels'])),
                'seed_ARI': float(diagnostics.iloc[0].seed_stability_ARI)}
    return result['labels'].copy(), settings


def analyse_partition(context, labels, *, min_df=3, max_df=0.9,
                      min_cluster_fraction=0.0, min_cluster_papers=2):
    """Include all abstract clusters, preserving both zero- and one-based IDs."""
    labels = np.asarray(labels)
    papers, vectors = context['papers'], context['embeddings']
    if (labels.shape != (len(papers),) or labels.dtype.kind not in 'iu'
            or np.any(labels < 0) or not 2 <= len(np.unique(labels)) < len(papers)):
        raise ValueError('Need aligned integer labels with between 2 and n-1 clusters')
    vocabulary = build_vocabulary(papers, min_df=min_df, max_df=max_df)
    terms = topic_terms(vocabulary, labels, top_n=len(vocabulary['terms']),
                        min_cluster_fraction=min_cluster_fraction,
                        min_cluster_papers=min_cluster_papers)
    distance = cosine_distances(vectors)
    distance = np.maximum((distance + distance.T) / 2, 0)
    np.fill_diagonal(distance, 0)
    silhouettes = silhouette_samples(distance, labels, metric='precomputed')
    summary, membership, examples = describe_abstract_partition(
        papers, vectors, {'labels': labels, 'silhouettes': silhouettes}, terms)
    summary['minimum_support'] = np.maximum(
        min_cluster_papers, np.ceil(summary.papers * min_cluster_fraction).astype(int))
    return {'papers': papers.copy(), 'embeddings': vectors,
            'passages': context['passages'].copy(), 'labels': labels.copy(),
            'vocabulary': vocabulary, 'terms': terms, 'summary': summary,
            'membership': membership, 'examples': examples,
            'unassigned': papers.iloc[:0].drop(columns='text').copy(),
            'min_df': min_df, 'max_df': max_df,
            'minimum_cluster_support': (
                f'max({min_cluster_papers}, ceil({min_cluster_fraction:g} * cluster_size))'),
            'search_cache': {}}


def supporting_papers(analysis, expression, cluster):
    """Return the first matching title or abstract with explicit field provenance."""
    return source_supporting_papers(analysis, expression, cluster).drop(
        columns='page').rename(columns={'block_id': 'source_field'})
