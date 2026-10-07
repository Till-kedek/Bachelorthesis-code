"""Leiden communities on a sparse, positive cosine-similarity graph."""
import numpy as np


def similarity_graph(embeddings, neighbours=15, min_similarity=0.0, mutual=False):
    """Undirected union (or mutual) kNN graph; edge weight is raw cosine similarity.

    No self-edges or negative/zero weights. All papers remain nodes, even isolated
    ones. Graph construction uses the original normalized embedding directions.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    x = np.array(embeddings, dtype=float, copy=True)
    if x.ndim != 2 or len(x) < 2 or x.shape[1] == 0 or not np.isfinite(x).all():
        raise ValueError('Expected at least two finite embedding vectors')
    norms = np.linalg.norm(x, axis=1)
    if np.any(norms == 0):
        raise ValueError('Zero vectors have undefined cosine similarity')
    if isinstance(neighbours, bool) or not isinstance(neighbours, (int, np.integer)) or neighbours < 1:
        raise ValueError('neighbours must be a positive integer')
    if not np.isfinite(min_similarity) or not 0 <= min_similarity < 1:
        raise ValueError('min_similarity must be in [0, 1)')
    x /= norms[:, None]
    similarity = np.clip(x @ x.T, -1, 1)
    np.fill_diagonal(similarity, 1)
    ranking = similarity.copy()
    np.fill_diagonal(ranking, -np.inf)
    k = min(neighbours, len(x) - 1)
    nearest = np.argsort(-ranking, axis=1, kind='stable')[:, :k]
    directed = np.zeros_like(similarity, dtype=bool)
    directed[np.arange(len(x))[:, None], nearest] = True
    selected = (directed & directed.T) if mutual else (directed | directed.T)
    selected &= similarity > min_similarity
    np.fill_diagonal(selected, False)
    a, b = np.where(np.triu(selected, 1))
    edges = np.column_stack([a, b])
    weights = similarity[a, b]
    count, components = connected_components(csr_matrix(selected), directed=False)
    return {'similarity': similarity, 'edges': edges, 'weights': weights,
            'components': components, 'neighbours': k, 'mutual': mutual,
            'min_similarity': min_similarity, 'diagnostics': {
                'papers': len(x), 'edges': len(edges), 'components': count,
                'isolated_papers': int(np.sum(selected.sum(axis=1) == 0)),
                'min_degree': int(selected.sum(axis=1).min()),
                'median_degree': float(np.median(selected.sum(axis=1))),
                'max_degree': int(selected.sum(axis=1).max())}}


def leiden_partition(graph, resolution=1.0, seed=42):
    """Weighted RB-configuration objective; resolution is not a cluster count."""
    try:
        import igraph as ig
        import leidenalg
    except ImportError as exc:
        raise ImportError('Install Leiden dependencies in your kernel: %pip install igraph leidenalg') from exc
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError('resolution must be positive and finite')
    if not len(graph['edges']):
        raise ValueError('The graph has no positive edges; inspect embeddings and graph settings')
    g = ig.Graph(n=len(graph['similarity']), edges=graph['edges'].tolist(), directed=False)
    partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
        weights=graph['weights'].tolist(), resolution_parameter=float(resolution),
        seed=int(seed), n_iterations=-1)
    # Deterministic display IDs by first paper row, not optimizer-internal IDs.
    raw = partition.membership
    mapping = {label: i + 1 for i, label in enumerate(dict.fromkeys(raw))}
    labels = np.array([mapping[label] for label in raw])
    return {'labels': labels, 'resolution': resolution, 'seed': seed,
            'quality': float(partition.quality())}


def leiden_sweep(graph, resolutions, seeds=(42, 43, 44), max_clusters=15):
    """Report every resolution; retain the highest-objective seed within each one.

    Objective values across different resolutions must not be compared as scores.
    Stability is pairwise adjusted Rand agreement across seeds on the same graph.
    Resolutions use 12 decimal places so arange grids and literal lookups agree.
    """
    import pandas as pd
    from itertools import combinations
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError('Supply at least two distinct seeds for stability evaluation')
    resolutions = [round(float(value), 12) for value in resolutions]
    if (not resolutions or len(set(resolutions)) != len(resolutions)
            or any(not np.isfinite(value) or value <= 0 for value in resolutions)):
        raise ValueError('Resolutions must be unique, positive and finite at 12 decimal places')
    distance = np.clip(1 - graph['similarity'], 0, 2)
    np.fill_diagonal(distance, 0)
    rows, results = [], {}
    for resolution in resolutions:
        runs = [leiden_partition(graph, resolution, seed) for seed in seeds]
        best = max(runs, key=lambda r: r['quality'])
        labels = best['labels']
        sizes = np.bincount(labels)[1:]
        k = len(sizes)
        scores = [adjusted_rand_score(a['labels'], b['labels']) for a, b in combinations(runs, 2)]
        rows.append({'resolution': float(resolution), 'clusters': k,
            'within_target': 2 <= k <= max_clusters, 'smallest': int(sizes.min()),
            'largest': int(sizes.max()), 'singletons': int(np.sum(sizes == 1)),
            'cosine_silhouette': silhouette_score(distance, labels, metric='precomputed') if 1 < k < len(labels) else np.nan,
            'seed_stability_ARI': float(np.mean(scores)), 'selected_seed': best['seed']})
        results[float(resolution)] = best
    return pd.DataFrame(rows), results


def leiden_tables(graph, partition, papers, representatives=3):
    """Representative titles and full membership, labelled L1, L2, … ."""
    from .clustering import hierarchy_tables

    labels = partition['labels']
    members = [np.flatnonzero(labels == i) for i in range(1, labels.max() + 1)]
    means = np.full((len(members), len(members)), np.nan)
    for i, group in enumerate(members):
        pairs = graph['similarity'][np.ix_(group, group)]
        if len(group) > 1:
            means[i, i] = pairs[~np.eye(len(group), dtype=bool)].mean()
    summary, membership = hierarchy_tables(
        {'similarity': graph['similarity'], 'order': np.argsort(labels, kind='stable')},
        {'labels': labels, 'members': members, 'similarity': means}, papers, representatives)
    for table in (summary, membership):
        table['Cluster'] = table['Cluster'].str.replace('C', 'L', regex=False)
    return summary, membership


def plot_leiden_resolution_selection(sweeps, selected_resolutions, *, title=None,
                                     show_selected_values=True):
    """Plot saved sweep diagnostics without fitting or choosing a resolution.

    Pass mappings such as {'PDFs': diagnostics, 'Abstracts': other_diagnostics}
    and {'PDFs': .95, 'Abstracts': .45} to overlay corpora on the same axes.
    Solid circles use the left silhouette axis; dashed diamonds use the right
    mean seed-ARI axis. Small numbers give cluster counts at tested resolutions.
    Set show_selected_values=False to omit the numeric summary boxes.
    """
    import matplotlib.pyplot as plt

    if not sweeps or set(sweeps) != set(selected_resolutions):
        raise ValueError('Supply one selected resolution for each named sweep')
    required = ['resolution', 'clusters', 'cosine_silhouette', 'seed_stability_ARI']
    prepared = []
    for name, table in sweeps.items():
        if not set(required) <= set(table):
            raise ValueError(f'{name}: missing sweep diagnostic columns')
        table = table.sort_values('resolution').copy()
        if table.empty or table.resolution.duplicated().any():
            raise ValueError(f'{name}: expected distinct tested resolutions')
        if not np.isfinite(table[required].to_numpy(dtype=float)).all():
            raise ValueError(f'{name}: diagnostic values must be finite')
        if (not table.cosine_silhouette.between(-1, 1).all()
                or not table.seed_stability_ARI.between(-1, 1).all()
                or not table.clusters.ge(1).all()
                or not table.clusters.mod(1).eq(0).all()):
            raise ValueError(f'{name}: invalid silhouette, ARI or cluster count')
        matches = np.isclose(table.resolution, selected_resolutions[name], rtol=0, atol=1e-12)
        if matches.sum() != 1:
            raise ValueError(f'{name}: selected resolution must match exactly one tested value')
        prepared.append((name, table, table.loc[matches].iloc[0]))

    fig, silhouette_ax = plt.subplots(figsize=(12.5, 6.8))
    stability_ax = silhouette_ax.twinx()
    fig.subplots_adjust(left=.09, right=.90, bottom=.27 if show_selected_values else .18, top=.80)
    colors = ['#17678a', '#c05a24', '#57823b', '#8b5aa5']
    handles, notes = [], []
    for i, (name, table, selected) in enumerate(prepared):
        color = colors[i % len(colors)]
        line, = silhouette_ax.plot(table.resolution, table.cosine_silhouette,
            color=color, marker='o', markersize=5, linewidth=2,
            label=f'{name}: cosine silhouette (left)')
        stability, = stability_ax.plot(table.resolution, table.seed_stability_ARI,
            color=color, marker='D', markersize=4, linewidth=1.5,
            linestyle='--', alpha=.65, label=f'{name}: seed stability ARI (right)')
        handles.extend([line, stability])
        for row in table.itertuples():
            silhouette_ax.annotate(str(int(row.clusters)),
                (row.resolution, row.cosine_silhouette), xytext=(0, 10 if len(prepared) > 1 and i == 0 else -17),
                textcoords='offset points', ha='center', fontsize=8, color=color,
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .8, 'pad': .1})
        silhouette_ax.axvline(selected.resolution, color=color, linewidth=1, linestyle=':', alpha=.6)
        if not show_selected_values:
            silhouette_ax.text(selected.resolution, 1.015,
                f'{name}: {int(selected.clusters)} clusters at {selected.resolution:g}',
                transform=silhouette_ax.get_xaxis_transform(), ha='center',
                color=color, fontsize=9)
        for ax, metric in [(silhouette_ax, 'cosine_silhouette'), (stability_ax, 'seed_stability_ARI')]:
            ax.scatter([selected.resolution], [selected[metric]], s=220,
                marker='o', facecolors='none', edgecolors='#c49720', linewidths=2.5, zorder=6)
        notes.append(f'{name}: resolution {selected.resolution:g} → {int(selected.clusters)} clusters\n'
                     f'Cosine silhouette {selected.cosine_silhouette:.3f} · '
                     f'Mean seed ARI {selected.seed_stability_ARI:.3f}')
        if len(prepared) == 1:
            # Shade only the contiguous run of tested settings containing the choice.
            k = table.clusters.to_numpy()
            position = int(np.flatnonzero(np.isclose(table.resolution, selected.resolution,
                                                    rtol=0, atol=1e-12))[0])
            left = right = position
            while left > 0 and k[left - 1] == k[position]:
                left -= 1
            while right + 1 < len(k) and k[right + 1] == k[position]:
                right += 1
            silhouette_ax.axvspan(table.resolution.iloc[left], table.resolution.iloc[right],
                color='#d9b954', alpha=.16, zorder=0)
            silhouette_ax.text(selected.resolution, .04,
                f'{int(selected.clusters)}-cluster settings', transform=silhouette_ax.get_xaxis_transform(),
                ha='center', fontsize=9, color='#715710')

    values = np.concatenate([table.cosine_silhouette.to_numpy() for _, table, _ in prepared])
    span = max(float(np.ptp(values)), .05)
    silhouette_ax.set_ylim(min(0, float(values.min()) - .15 * span),
                           min(1, float(values.max()) + .3 * span))
    min_ari = min(float(table.seed_stability_ARI.min()) for _, table, _ in prepared)
    stability_ax.set_ylim(min(0, min_ari - .05) if min_ari < 0 else 0, 1.05)
    silhouette_ax.set(xlabel='Leiden resolution', ylabel='Mean cosine silhouette')
    stability_ax.set_ylabel('Seed stability: mean pairwise adjusted Rand index')
    silhouette_ax.grid(axis='y', color='#d9dfe3', linewidth=.7, alpha=.8)
    silhouette_ax.spines['top'].set_visible(False)
    stability_ax.spines['top'].set_visible(False)
    silhouette_ax.set_axisbelow(True)
    ticks = sorted(set(round(float(row.resolution), 12)
                       for _, table, _ in prepared for row in table.itertuples()))
    if len(ticks) <= 30:
        silhouette_ax.set_xticks(ticks, [f'{r:.2f}' for r in ticks], rotation=45, ha='right')
    fig.suptitle(title or 'Leiden resolution: separation and seed stability',
                 fontsize=15, fontweight='bold', y=.98)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .93),
               ncol=2, frameon=False, fontsize=10)
    if show_selected_values:
        for i, note in enumerate(notes):
            fig.text(.09 + i * .8 / len(notes), .025, note, fontsize=10,
                     bbox={'boxstyle': 'round,pad=.5', 'facecolor': '#fbf6e5', 'edgecolor': '#d9b954'})
    fig.text(.09, .115 if show_selected_values else .025, 'Numbers beside solid points = clusters; rings = selected resolution. '
             'Each metric uses its labelled axis.', fontsize=9, color='#555555')
    return fig, {'silhouette': silhouette_ax, 'stability': stability_ax}
