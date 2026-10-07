"""Inspect saved thesis partitions and source expressions without new PDF extraction."""

from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.metrics.pairwise import cosine_distances
from threadpoolctl import threadpool_limits

from .graph_clustering import leiden_partition, similarity_graph
from .storage import file_hash, fingerprint, read_json, write_json
from .thesis_analysis import canonical_corpus
from .topic_analysis import build_vocabulary, describe_partition, topic_terms


def load_context(root):
    """Load the exact saved 292-work thesis corpus and authenticate its partitions."""
    root = Path(root).resolve()
    evidence = root / 'reports/thesis/evidence'
    manifest = read_json(evidence / 'run_manifest.json')
    for relative, expected in manifest['inputs'].items():
        path = root / relative
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f'Saved thesis input changed or is missing: {relative}')

    papers, embeddings, passages, *_ = canonical_corpus(root)
    saved = pd.read_csv(evidence / 'corpus.csv')
    if (len(papers) != 292 or not saved.Key.tolist() == papers.Key.tolist()
            or not saved.paper_id.tolist() == papers.paper_id.tolist()
            or not saved.text_sha256.tolist() == papers.text_sha256.tolist()):
        raise ValueError('Saved thesis partitions do not match the current canonical works and text')

    partitions = {}
    with np.load(evidence / 'text_partitions.npz', allow_pickle=False) as archive:
        for name in archive.files:
            partitions[name] = _checked_labels(archive[name], len(papers), name)
    metrics = pd.read_csv(evidence / 'text_metrics.csv')
    if set(partitions) != set(metrics.solution) or metrics.solution.duplicated().any():
        raise ValueError('Saved text partitions and metrics disagree')
    for row in metrics.itertuples():
        if int(row.k) != len(np.unique(partitions[row.solution])):
            raise ValueError(f'Saved cluster count changed: {row.solution}')

    for name, filename, column in (
        ('citation_r1', 'citation_membership_r1.csv', 'citation_label'),
        ('hierarchy_15', 'hierarchy_membership_15.csv', 'cluster'),
    ):
        table = pd.read_csv(evidence / filename)
        if table.Key.tolist() != papers.Key.tolist() or table.paper_id.tolist() != papers.paper_id.tolist():
            raise ValueError(f'Saved {name} membership does not match canonical work order')
        partitions[name] = _checked_labels(table[column].to_numpy(), len(papers), name)
    if (set(partitions['citation_r1']) - {0} != set(range(1, 7))
            or int((partitions['citation_r1'] == 0).sum()) != 3):
        raise ValueError('Saved citation partition no longer has six communities and three isolates')

    return {'root': root, 'evidence': evidence, 'papers': papers, 'embeddings': embeddings,
            'passages': passages, 'partitions': partitions, 'metrics': metrics,
            'input_hashes': manifest['inputs']}


def _checked_labels(values, n, name):
    labels = np.asarray(values)
    if (labels.shape != (n,) or labels.dtype.kind not in 'iu'
            or len(np.unique(labels)) < 2 or len(np.unique(labels)) >= n
            or np.any(labels < 0) or (name != 'citation_r1' and np.any(labels == 0))):
        raise ValueError(f'Invalid saved partition: {name}')
    return labels.astype(int, copy=True)


def partition_catalog(context):
    """List saved partitions without selecting or refitting one."""
    rows = []
    metrics = context['metrics'].set_index('solution')
    for name, labels in context['partitions'].items():
        assigned = labels[labels > 0]
        row = {'solution': name, 'method': 'Citation Leiden' if name == 'citation_r1'
               else 'Hierarchy' if name == 'hierarchy_15' else str(metrics.loc[name, 'method']),
               'clusters': len(np.unique(assigned)), 'assigned': len(assigned),
               'unassigned': int(np.sum(labels == 0))}
        if name in metrics.index:
            row['cosine_silhouette'] = float(metrics.loc[name, 'cosine_silhouette'])
            row['seed_ARI'] = float(metrics.loc[name, 'seed_ARI'])
            row['resolution'] = metrics.loc[name, 'resolution']
        rows.append(row)
    return pd.DataFrame(rows)


def choose_partition(context, *, source='saved', saved='kmeans_k5', method='kmeans',
                     k=5, resolution=1.0, neighbours=15, seed=42, n_init=10):
    """Preserve a saved assignment or run one explicitly requested exploratory fit."""
    if source == 'saved':
        if saved not in context['partitions']:
            raise ValueError(f'Unknown saved partition {saved!r}; inspect the catalog')
        labels = context['partitions'][saved].copy()
        settings = {'source': 'saved', 'saved': saved,
                    'method': 'Citation Leiden' if saved == 'citation_r1' else
                    'Hierarchy' if saved == 'hierarchy_15' else 'K-means' if saved.startswith('kmeans_') else 'Text Leiden',
                    'clusters': len(set(labels) - {0})}
        metric = context['metrics'].loc[context['metrics'].solution.eq(saved)]
        if len(metric):
            row = metric.iloc[0]
            for key in ('seed', 'resolution', 'selection'):
                if key in row and pd.notna(row[key]):
                    settings[key] = int(row[key]) if key == 'seed' else row[key]
        return labels, settings
    if source != 'fit':
        raise ValueError("SOURCE must be 'saved' or 'fit'")
    vectors = context['embeddings']
    if method == 'kmeans':
        if not isinstance(k, (int, np.integer)) or not 2 <= k < len(vectors):
            raise ValueError('K must be an integer between 2 and the number of works minus one')
        if not isinstance(n_init, (int, np.integer)) or n_init < 1:
            raise ValueError('N_INIT must be a positive integer')
        with threadpool_limits(limits=1):
            labels = KMeans(n_clusters=int(k), n_init=int(n_init), random_state=int(seed)).fit_predict(vectors) + 1
        settings = {'source': 'fit', 'method': 'K-means', 'k': int(k), 'seed': int(seed), 'n_init': int(n_init)}
    elif method == 'leiden':
        graph = similarity_graph(vectors, neighbours=neighbours)
        fit = leiden_partition(graph, resolution=resolution, seed=seed)
        labels = fit['labels']
        settings = {'source': 'fit', 'method': 'Text Leiden', 'resolution': float(resolution),
                    'neighbours': int(neighbours), 'seed': int(seed), 'quality': fit['quality']}
    else:
        raise ValueError("METHOD must be 'kmeans' or 'leiden' when SOURCE='fit'")
    settings['clusters'] = len(np.unique(labels))
    return labels, settings


def analyse_partition(context, labels, *, min_df=3, max_df=0.9):
    """Compute complete descriptive term rankings and verifiable paper examples."""
    labels = np.asarray(labels)
    if (labels.shape != (len(context['papers']),) or labels.dtype.kind not in 'iu'
            or np.any(labels < 0) or len(set(labels) - {0}) < 2):
        raise ValueError('Need aligned integer labels with at least two assigned clusters')
    active = np.flatnonzero(labels > 0)
    papers = context['papers'].iloc[active].copy().reset_index(drop=True)
    vectors = context['embeddings'][active]
    selected_labels = labels[active].astype(int, copy=True)
    row_map = dict(zip(active, range(len(active))))
    passages = context['passages'].loc[context['passages'].row.isin(row_map)].copy()
    passages['row'] = passages.row.map(row_map).astype(int)
    papers['row'] = np.arange(len(papers))
    vocabulary = build_vocabulary(papers, min_df=min_df, max_df=max_df)
    terms = topic_terms(vocabulary, selected_labels, top_n=len(vocabulary['terms']))
    distance = cosine_distances(vectors)
    distance = np.maximum((distance + distance.T) / 2, 0)
    np.fill_diagonal(distance, 0)
    silhouettes = silhouette_samples(distance, selected_labels, metric='precomputed')
    summary, membership, examples = describe_partition(
        papers, vectors, passages, {'labels': selected_labels, 'silhouettes': silhouettes}, terms)
    summary['minimum_support'] = np.maximum(2, np.ceil(summary.papers * 0.1).astype(int))
    unassigned = context['papers'].iloc[np.flatnonzero(labels == 0)].drop(columns='text').copy()
    return {'papers': papers, 'embeddings': vectors, 'passages': passages, 'labels': selected_labels,
            'vocabulary': vocabulary, 'terms': terms, 'summary': summary, 'membership': membership,
            'examples': examples, 'unassigned': unassigned, 'min_df': min_df, 'max_df': max_df,
            'search_cache': {}}


def ranked_terms(analysis, cluster, ranking='tfidf_contrast', kind='expression', top_n=20):
    if cluster not in set(analysis['labels']):
        raise ValueError('Unknown cluster')
    if ranking not in ('tfidf_contrast', 'c_tf_idf') or kind not in ('expression', 'word'):
        raise ValueError('Choose a known ranking and term kind')
    if not isinstance(top_n, (int, np.integer)) or top_n < 1:
        raise ValueError('TOP_N must be positive')
    rows = analysis['terms']
    mask = rows.cluster.eq(cluster) & rows.ranking.eq(ranking)
    mask &= rows.term.str.contains(' ', regex=False) if kind == 'expression' else ~rows.term.str.contains(' ', regex=False)
    return rows.loc[mask].sort_values('score', ascending=False, kind='stable').head(top_n).reset_index(drop=True)


def evidence_table(rows):
    """Format ranked terms without rounding the stored scores."""
    result = rows[['term', 'score', 'inside_count', 'inside_n', 'outside_count', 'outside_n',
                   'inside_prevalence', 'outside_prevalence', 'prevalence_gap']].copy()
    result.columns = ['Term', 'Score', 'Inside papers', 'Inside total', 'Outside papers',
                      'Outside total', 'Inside %', 'Outside %', 'Gap (pp)']
    for column in ('Inside %', 'Outside %', 'Gap (pp)'):
        result[column] *= 100
    return result


def _search_matches(analysis, expression):
    normalized = unicodedata.normalize('NFKC', expression).strip().lower().replace('-', ' ')
    tokens = re.findall(r'[^\W_]+', normalized, flags=re.UNICODE)
    if not tokens or not all(token.isalnum() for token in tokens):
        raise ValueError('Enter at least one word or number')
    normalized = ' '.join(tokens)
    if normalized in analysis['search_cache']:
        return normalized, analysis['search_cache'][normalized]
    pattern = re.compile(r'(?<!\w)' + r'[\s-]+'.join(map(re.escape, tokens)) + r'(?!\w)', re.I)
    counts = np.zeros(len(analysis['papers']), dtype=int)
    first = {}
    for passage in analysis['passages'].itertuples(index=False):
        hits = list(pattern.finditer(passage.text))
        if hits:
            counts[passage.row] += len(hits)
            first.setdefault(passage.row, {'page': passage.page, 'block_id': passage.block_id,
                                           'passage': passage.text})
    analysis['search_cache'][normalized] = (counts, first)
    return normalized, (counts, first)


def search_expression(analysis, expression):
    normalized, (counts, _) = _search_matches(analysis, expression)
    rows = []
    for cluster in sorted(set(analysis['labels'])):
        within = counts[analysis['labels'] == cluster]
        papers = len(within)
        rows.append({'Cluster': int(cluster), 'Papers in cluster': papers,
                     'Papers containing term': int(np.count_nonzero(within)),
                     'Papers containing term (%)': 100 * np.count_nonzero(within) / papers,
                     'Total occurrences': int(within.sum()),
                     'Occurrences per paper': float(within.mean())})
    result = pd.DataFrame(rows)
    result.attrs['normalized_query'] = normalized
    return result


def supporting_papers(analysis, expression, cluster):
    """List every supporting work and the first source block containing the phrase."""
    if cluster not in set(analysis['labels']):
        raise ValueError('Unknown cluster')
    _, (counts, first) = _search_matches(analysis, expression)
    rows = []
    for i in np.flatnonzero(counts):
        paper = analysis['papers'].iloc[i]
        rows.append({'paper_id': paper.paper_id, 'title': paper.title,
                     'cluster': int(analysis['labels'][i]), 'inside': bool(analysis['labels'][i] == cluster),
                     'occurrences': int(counts[i]), **first[i]})
    return pd.DataFrame(rows, columns=['paper_id', 'title', 'cluster', 'inside', 'occurrences',
                                        'page', 'block_id', 'passage'])


def export_analysis(analysis, context, partition_settings, *, support=None, display_settings=None):
    """Create a new, never overwritten evidence folder for this exact analysis."""
    base = Path(context['root']) / context.get('export_directory', 'outputs/topic_analysis_2')
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination = base / f'{stamp}_{fingerprint(partition_settings)[:8]}'
    destination.mkdir(exist_ok=False)
    tables = {'cluster_summary.csv': analysis['summary'],
              'membership.csv': analysis['membership'],
              'term_rankings.csv': analysis['terms'],
              'review_papers.csv': analysis['examples'],
              'unassigned.csv': analysis['unassigned']}
    if support is not None:
        tables['supporting_papers.csv'] = support
    for name, frame in tables.items():
        frame.to_csv(destination / name, index=False)
    write_json(destination / 'run_manifest.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_report': str(context['evidence'].relative_to(context['root'])),
        'source_input_hashes': context['input_hashes'],
        'source_partitions_sha256': file_hash(context.get(
            'partition_file', context['evidence'] / 'text_partitions.npz')),
        'partition_settings': partition_settings,
        'membership_id': fingerprint(analysis['membership'][['paper_id', 'cluster']].to_dict('records')),
        'text_settings': {'min_df': analysis['min_df'], 'max_df': analysis['max_df'],
                          'minimum_cluster_support': analysis.get(
                              'minimum_cluster_support', 'max(2, ceil(0.1 * cluster_size))')},
        'display_settings': display_settings or {},
        'packages': {name: version(name) for name in
                     ('numpy', 'pandas', 'scikit-learn', 'igraph', 'leidenalg')},
        'artifact_sha256': {name: file_hash(destination / name) for name in tables},
    })
    return destination
