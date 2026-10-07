"""Explore saved citation communities using their retained PDF text and graph evidence."""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from .citation_clustering import citation_leiden_sweep
from .citation_topic_analysis import (
    _labels, describe_citation_topics, load_citation_topic_corpus,
    load_citation_topic_partition, save_citation_topic_partition,
)
from .storage import file_hash, fingerprint, read_json, write_json
from .topic_analysis import build_vocabulary, topic_terms
from .topic_explorer import (
    evidence_table, export_analysis, ranked_terms, search_expression, supporting_papers,
)


def _saved_sweep(corpus, sweep_dir, provenance_file):
    """Align source memberships by work identity and check their graph diagnostics."""
    provenance = read_json(provenance_file)
    if (provenance['selection_id'] != corpus['selection']['selection_id']
            or provenance['include_manual_citations'] != corpus['network']['include_manual']
            or provenance['citation_graph_rule'] != 'binary undirected union; isolates unassigned'
            or provenance['embedding_manifest_sha256'] != corpus['input_hashes']['embeddings/manifest.json']
            or any(corpus['input_hashes'].get(f'graph/{name}') != sha
                   for name, sha in provenance['graph_files'].items())):
        raise ValueError('Saved citation sweep graph or input scope does not match this corpus')
    diagnostics = pd.read_csv(sweep_dir / 'resolution_sweep.csv')
    memberships = pd.read_csv(sweep_dir / 'all_seed_memberships.csv')
    resolutions = diagnostics.resolution.round(12).tolist()
    if (len(set(resolutions)) != len(resolutions)
            or resolutions != [round(r, 12) for r in provenance['citation_resolution_sweep']]
            or set(memberships.resolution.round(12)) != set(resolutions)):
        raise ValueError('Saved citation resolution grids disagree')
    seeds = provenance['citation_seeds']
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError('Expected distinct citation seeds')
    keys = corpus['papers'].pdf_sha256
    network = corpus['network']
    active = network['active']
    degrees = network['degree']
    edges = network['edges']
    solutions = {}
    for row in diagnostics.itertuples():
        resolution = round(row.resolution, 12)
        subset = memberships.loc[np.isclose(memberships.resolution, resolution, rtol=0, atol=1e-12)]
        if set(subset.seed) != set(seeds):
            raise ValueError('Saved citation seeds disagree')
        runs = []
        for seed in seeds:
            frame = subset.loc[subset.seed.eq(seed)]
            if not frame.Key.is_unique or set(frame.Key) != set(keys):
                raise ValueError('Saved citation memberships must cover every work exactly once')
            frame = frame.set_index('Key').loc[keys]
            if frame.paper_id.tolist() != corpus['papers'].paper_id.tolist():
                raise ValueError('Saved citation paper IDs disagree')
            labels = _labels(frame.citation_label.to_numpy(), corpus)
            internal = float(np.mean(labels[edges[:, 0]] == labels[edges[:, 1]]))
            degree_shares = np.bincount(labels, weights=degrees)[1:] / degrees.sum()
            penalty = float(np.sum(degree_shares ** 2))
            runs.append({'seed': int(seed), 'labels': labels, 'internal': internal,
                         'modularity': internal - penalty,
                         'objective': internal - float(row.resolution) * penalty})
        selected = next((run for run in runs if run['seed'] == row.selected_seed), None)
        stability = float(np.mean([adjusted_rand_score(a['labels'][active], b['labels'][active])
                                   for a, b in combinations(runs, 2)]))
        if (selected is None
                or not np.isclose(selected['objective'], max(run['objective'] for run in runs), atol=1e-12)
                or len(set(selected['labels']) - {0}) != row.communities
                or int(np.sum(selected['labels'] == 0)) != row.unassigned_isolates
                or not np.isclose(selected['modularity'], row.modularity_gamma_1, atol=1e-12)
                or not np.isclose(selected['internal'], row.within_community_link_fraction, atol=1e-12)
                or not np.isclose(stability, row.seed_stability_ARI_mean, atol=1e-12)):
            raise ValueError(f'Saved citation diagnostics or selected seed disagree at {resolution:g}')
        solutions[f'citation_r{resolution:g}'] = {
            'labels': selected['labels'], 'method': 'Citation Leiden',
            'resolution': resolution, 'seed': selected['seed'], 'seeds': seeds,
            'clusters': int(row.communities), 'seed_ARI': stability,
            'modularity_gamma_1': selected['modularity'],
            'within_community_link_fraction': selected['internal'],
            'selection': 'saved highest-objective source seed at this resolution'}
    return solutions


def load_context(root):
    """Load the reviewed graph and authenticate saved sweep memberships; never fit."""
    root = Path(root).resolve()
    graph = root / 'outputs/citation_graph_openalex/9ad47e071ee12c2f/reviewed'
    corpus = load_citation_topic_corpus(
        root / 'outputs/corpus_sections', root / 'configs/analysis_exclusions.json',
        graph, include_manual=True)
    scope = corpus['selection']['selection_id'][:16]
    sweep_dir = root / 'outputs/citation_clustering' / scope / 'with_manual'
    provenance_file = sweep_dir / 'resolution_1/leiden_0.95/run_manifest.json'
    solutions = _saved_sweep(corpus, sweep_dir, provenance_file)
    source_files = [sweep_dir / 'resolution_sweep.csv',
                    sweep_dir / 'all_seed_memberships.csv', provenance_file]
    source_hashes = {str(path.relative_to(root)): file_hash(path) for path in source_files}
    identity = {'schema_version': 1, 'input_id': corpus['input_id'], 'source_files': source_hashes}
    evidence = (root / 'outputs/citation_topic_partitions' / scope
                / 'explorer_catalog' / fingerprint(identity)[:16])
    manifest_path = evidence / 'catalog_manifest.json'
    if not manifest_path.exists():
        snapshots = {}
        for name, result in solutions.items():
            folder = f"resolution_{result['resolution']:g}"
            spec = save_citation_topic_partition(
                evidence / folder, corpus, result['labels'], resolution=result['resolution'],
                seed=result['seed'], fit_input_id=corpus['input_id'],
                source_evidence={'origin': 'existing citation sweep memberships; no refitting',
                                 'files': source_hashes, 'selection': result['selection'],
                                 'seeds': result['seeds']})
            snapshots[name] = {'folder': folder, 'assignments_sha256': spec['assignments_sha256']}
        write_json(manifest_path, {**identity, 'solutions': snapshots})
    manifest = read_json(manifest_path)
    if (any(manifest.get(key) != value for key, value in identity.items())
            or set(manifest['solutions']) != set(solutions)):
        raise ValueError('Citation catalog provenance changed')
    for name, snapshot in manifest['solutions'].items():
        labels, spec = load_citation_topic_partition(evidence / snapshot['folder'], corpus)
        result = solutions[name]
        if (spec['assignments_sha256'] != snapshot['assignments_sha256']
                or not np.array_equal(labels, result['labels'])
                or spec['resolution'] != result['resolution'] or spec['seed'] != result['seed']):
            raise ValueError(f'Saved citation snapshot differs from the source sweep: {name}')
        result['labels'] = labels
    return {**corpus, 'root': root, 'evidence': evidence, 'solutions': solutions,
            'partition_file': manifest_path,
            'input_hashes': {**corpus['input_hashes'], **source_hashes},
            'export_directory': 'outputs/topic_analysis_citations_v2'}


def partition_catalog(context):
    return pd.DataFrame([
        {'solution': name, 'method': result['method'], 'clusters': result['clusters'],
         'assigned': int(np.sum(result['labels'] > 0)),
         'unassigned': int(np.sum(result['labels'] == 0)),
         'resolution': result['resolution'], 'modularity_gamma_1': result['modularity_gamma_1'],
         'seed_ARI': result['seed_ARI'], 'seed': result['seed']}
        for name, result in context['solutions'].items()])


def choose_partition(context, *, source='saved', saved='citation_r0.9', resolution=0.9,
                     seeds=(42, 43, 44)):
    if source == 'saved':
        if saved not in context['solutions']:
            raise ValueError(f'Unknown saved citation partition {saved!r}; inspect the catalog')
        result = context['solutions'][saved]
        return result['labels'].copy(), {
            **{key: value for key, value in result.items() if key != 'labels'},
            'source': 'saved', 'saved': saved, 'include_manual': context['network']['include_manual']}
    if source != 'fit':
        raise ValueError("SOURCE must be 'saved' or 'fit'")
    diagnostics, results = citation_leiden_sweep(context['network'], [resolution], seeds=seeds)
    result = results[float(resolution)]
    return result['labels'].copy(), {
        'source': 'fit', 'method': 'Citation Leiden', 'resolution': float(resolution),
        'seeds': list(seeds), 'seed': result['seed'],
        'clusters': int(diagnostics.iloc[0].communities),
        'seed_ARI': float(diagnostics.iloc[0].seed_stability_ARI_mean),
        'modularity_gamma_1': result['modularity_gamma_1'],
        'selection': 'highest objective across seeds at this resolution',
        'include_manual': context['network']['include_manual']}


def analyse_partition(context, labels, *, min_df=3, max_df=0.9,
                      min_cluster_papers=2, min_cluster_fraction=0.0):
    """Exclude isolates from term contrasts, but preserve them in membership exports."""
    labels = _labels(labels, context)
    active = np.flatnonzero(labels > 0)
    papers = context['papers'].iloc[active].copy().reset_index(drop=True)
    row_map = dict(zip(active, range(len(active))))
    passages = context['passages'].loc[context['passages'].row.isin(row_map)].copy()
    passages['row'] = passages.row.map(row_map).astype(int)
    papers['row'] = np.arange(len(papers))
    vocabulary = build_vocabulary(papers, min_df=min_df, max_df=max_df)
    terms = topic_terms(vocabulary, labels[active], top_n=len(vocabulary['terms']),
                        min_cluster_papers=min_cluster_papers,
                        min_cluster_fraction=min_cluster_fraction)
    summary, membership, examples = describe_citation_topics(context, labels, terms)
    summary['minimum_support'] = np.maximum(
        min_cluster_papers, np.ceil(summary.papers * min_cluster_fraction).astype(int))
    unassigned = context['papers'].loc[labels == 0].drop(columns='text').copy()
    return {'papers': papers, 'passages': passages, 'labels': labels[active].copy(),
            'vocabulary': vocabulary, 'terms': terms, 'summary': summary,
            'membership': membership, 'examples': examples, 'unassigned': unassigned,
            'min_df': min_df, 'max_df': max_df, 'search_cache': {},
            'minimum_cluster_support': (
                f'max({min_cluster_papers}, ceil({min_cluster_fraction:g} * cluster_size))')}
