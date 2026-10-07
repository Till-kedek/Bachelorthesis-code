"""Hierarchy diagnostics in the original paper-embedding space."""
import numpy as np


def cluster_hierarchy(hierarchy, max_clusters=15):
    """Cut into at most k groups and collapse the *same* tree for display.

    Tied merges at the cut height stay together, so fewer than k groups may result.
    Cluster IDs follow the original hierarchy's left-to-right leaf order.
    Similarities average paper pairs, never cluster centroids; the diagonal
    excludes self-pairs and is undefined (NaN) for singleton groups.
    """
    from scipy.cluster.hierarchy import fcluster

    if isinstance(max_clusters, bool) or not isinstance(max_clusters, (int, np.integer)) or max_clusters < 1:
        raise ValueError('max_clusters must be a positive integer')
    n = len(hierarchy['order'])
    tree = hierarchy['linkage']
    k = min(max_clusters, n)
    cut = None if k == n else float(tree[n - k - 1, 2])
    raw = np.arange(n) if cut is None else fcluster(tree, t=cut, criterion='distance')
    ordered_ids = list(dict.fromkeys(raw[hierarchy['order']].tolist()))
    mapping = {old: i for i, old in enumerate(ordered_ids)}
    labels = np.array([mapping[x] for x in raw], dtype=int)
    k = len(ordered_ids)
    members = [np.flatnonzero(labels == i) for i in range(k)]
    means = np.full((k, k), np.nan)
    for i, a in enumerate(members):
        for j in range(i, k):
            b = members[j]
            pairs = hierarchy['similarity'][np.ix_(a, b)]
            if i == j:
                pairs = pairs[~np.eye(len(a), dtype=bool)]
            if pairs.size:
                means[i, j] = means[j, i] = pairs.mean()

    # Contract each cut subtree into one leaf, preserving original merge heights.
    node_map = dict(enumerate(labels.tolist()))
    counts = {i: 1 for i in range(k)}
    collapsed = []
    for index, (left, right, height, _) in enumerate(tree):
        a, b = node_map[int(left)], node_map[int(right)]
        if a == b:
            node_map[n + index] = a
        else:
            new = k + len(collapsed)
            counts[new] = counts[a] + counts[b]
            collapsed.append([a, b, height, counts[new]])
            node_map[n + index] = new
    return {'labels': labels + 1, 'members': members, 'similarity': means,
            'linkage': np.asarray(collapsed, dtype=float).reshape(-1, 4),
            'sizes': np.array([len(m) for m in members]), 'cut_distance': cut,
            'requested_max': max_clusters}


def hierarchy_tables(hierarchy, clusters, papers, representatives=3):
    """Cluster overview plus full membership; representatives are central members.

    Rank by mean cosine similarity to other papers in the same cluster. A title
    describes a member, not an automatically inferred thematic cluster label.
    """
    import pandas as pd

    if len(papers) != len(clusters['labels']):
        raise ValueError('Paper rows must match the hierarchy')
    if representatives < 1:
        raise ValueError('representatives must be positive')
    rows = []
    for i, members in enumerate(clusters['members']):
        similarities = hierarchy['similarity'][np.ix_(members, members)]
        scores = ((similarities.sum(axis=1) - np.diag(similarities)) / (len(members) - 1)
                  if len(members) > 1 else np.zeros(1))
        chosen = members[np.argsort(-scores, kind='stable')[:representatives]]
        examples = [f"{papers.iloc[j]['Paper ID']}: {papers.iloc[j]['Title']}" for j in chosen]
        rows.append({'Cluster': f'C{i + 1}', 'Papers': len(members),
                     'Within-cluster similarity': clusters['similarity'][i, i],
                     'Representative papers': '\n'.join(examples)})
    metadata_columns = [name for name in (
        'source_filename', 'segments', 'doi', 'year', 'journal',
        'document_type', 'document_language', 'token_count', 'abstract_word_count',
    ) if name in papers]
    membership = papers[['Paper ID', 'Key', 'Title', *metadata_columns]].copy()
    membership.insert(0, 'Cluster', [f'C{i}' for i in clusters['labels']])
    membership = membership.iloc[hierarchy['order']].reset_index(drop=True)
    return pd.DataFrame(rows), membership


def cluster_review_report(summary, membership):
    """Plain text for manual review; small clusters are candidates, not exclusions.

    Use a summary generated with three representatives. Clusters with fewer than
    three papers list every member, independent of the representative ranking.
    """
    lines = ['HIERARCHICAL CLUSTER REVIEW',
             'Small clusters are review candidates, not confirmed outliers.',
             'Titles are extracted metadata and may need correction.', '']
    for _, row in summary.iterrows():
        name, count = row['Cluster'], int(row['Papers'])
        members = membership.loc[membership['Cluster'] == name]
        if len(members) != count:
            raise ValueError(f'Membership count differs from summary for {name}')
        label = 'all papers; small-cluster review' if count < 3 else '3 representative papers'
        lines.append(f'{name} — {count} paper(s) — {label}')
        if count < 3:
            titles = [f"{r['Paper ID']}: {r['Title']}" for _, r in members.iterrows()]
        else:
            titles = str(row['Representative papers']).splitlines()
            if len(titles) < 3:
                raise ValueError('Generate the summary with at least three representatives')
            titles = titles[:3]
        lines.extend(f'  - {title}' for title in titles)
        lines.append('')
    return '\n'.join(lines)


def plot_cluster_hierarchy(clusters, *, color_limits=None):
    """Aligned cluster dendrogram and annotated mean paper-pair similarities.

    Auto colour limits use the finite displayed values; explicit limits allow
    comparisons across runs. The numerical labels and colourbar show the scale.
    """
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    matrix = clusters['similarity']
    k = len(matrix)
    values = matrix[np.isfinite(matrix)]
    if color_limits is None:
        low, high = (float(values.min()), float(values.max())) if values.size else (-1, 1)
        if high - low < 1e-8:
            low, high = max(-1, low - .05), min(1, high + .05)
    else:
        low, high = color_limits
    if not -1 <= low < high <= 1:
        raise ValueError('Color limits must be increasing and within [-1, 1]')
    labels = [f'C{i + 1} (n={size})' for i, size in enumerate(clusters['sizes'])]
    fig = plt.figure(figsize=(max(10, .7 * k + 3), max(9, .55 * k + 4)), layout='constrained')
    grid = fig.add_gridspec(2, 2, height_ratios=[3, 8], width_ratios=[1, .035])
    tree_ax = fig.add_subplot(grid[0, 0])
    heat_ax = fig.add_subplot(grid[1, 0], sharex=tree_ax)
    color_ax = fig.add_subplot(grid[1, 1])
    if k > 1:
        drawn = dendrogram(clusters['linkage'], labels=labels, ax=tree_ax,
                           color_threshold=0, above_threshold_color='#285779', leaf_font_size=9)
        if drawn['leaves'] != list(range(k)):
            raise ValueError('Collapsed hierarchy order differs from cluster order')
    else:
        tree_ax.text(5, .5, labels[0], ha='center')
        tree_ax.set_ylim(0, 1)
    tree_ax.set_title(f'Average-linkage hierarchy: {k} clusters (requested maximum {clusters["requested_max"]})')
    tree_ax.set_ylabel('Merge cosine distance')
    tree_ax.tick_params(axis='x', bottom=True, labelbottom=True, labelrotation=45, labelsize=9)
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('#eeeeee')
    im = heat_ax.imshow(np.ma.masked_invalid(matrix), vmin=low, vmax=high, cmap=cmap,
                        extent=(0, 10 * k, 10 * k, 0), aspect='auto', interpolation='nearest')
    positions = 10 * np.arange(k) + 5
    heat_ax.set_xticks(positions, labels, rotation=45, ha='right', fontsize=9)
    heat_ax.set_yticks(positions, labels, fontsize=9)
    heat_ax.set_xlim(0, 10 * k)
    heat_ax.set_title('Mean cosine similarity between paper pairs\nDiagonal: within-cluster similarity; self-pairs excluded')
    for i in range(k):
        for j in range(k):
            value = matrix[i, j]
            label = f'{value:.2f}' if np.isfinite(value) else '—'
            color = 'black' if not np.isfinite(value) or (value - low) / (high - low) > .55 else 'white'
            heat_ax.text(positions[j], positions[i], label, ha='center', va='center', color=color, fontsize=8)
    fig.colorbar(im, cax=color_ax, label=f'Mean cosine similarity (scale {low:.3f} to {high:.3f})',
                 extend='both' if color_limits is not None else 'neither')
    return fig, {'dendrogram': tree_ax, 'heatmap': heat_ax, 'colorbar': color_ax}


def cosine_hierarchy(embeddings, *, optimal_ordering=True):
    """Average-linkage agglomeration with optional optimal leaf ordering.

    Normalization is applied to a copy. Cosine distance is 1 - cosine similarity;
    the condensed distance vector, not rows of a distance matrix, enters linkage.
    Optimal leaf ordering changes the display order, not the cluster merges.
    Disable it for larger corpora to avoid the expensive ordering step.
    """
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    values = np.array(embeddings, dtype=np.float64, copy=True)
    if values.ndim != 2 or len(values) < 2 or values.shape[1] == 0:
        raise ValueError('Hierarchy requires at least two nonempty embedding vectors')
    if not np.isfinite(values).all():
        raise ValueError('Embedding values must be finite')
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms == 0):
        raise ValueError('Cosine distance is undefined for zero vectors')
    values /= norms[:, None]
    similarity = np.clip(values @ values.T, -1, 1)
    similarity = (similarity + similarity.T) / 2
    np.fill_diagonal(similarity, 1)
    distance = np.clip(1 - similarity, 0, 2)
    np.fill_diagonal(distance, 0)
    tree = linkage(squareform(distance, checks=True), method='average',
                   optimal_ordering=optimal_ordering)
    return {'similarity': similarity, 'distance': distance, 'linkage': tree,
            'order': leaves_list(tree).astype(int)}


def plot_hierarchy(hierarchy, paper_labels, *, cut_distance=None, color_limits=(-1, 1), max_tick_labels=35):
    """Return a dendrogram and heatmap figure; never display or save automatically.

    Both panels use the same leaf coordinates. The heatmap diagonal is masked:
    self-similarity is uninformative. Optional boundaries denote a user-selected
    cut, not an inferred optimal number of topics.
    """
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, fcluster

    n = len(hierarchy['order'])
    if len(paper_labels) != n:
        raise ValueError('One paper label is required per embedding row')
    if max_tick_labels < 1:
        raise ValueError('max_tick_labels must be positive')
    low, high = color_limits
    if not -1 <= low < high <= 1:
        raise ValueError('Color limits must be increasing and within [-1, 1]')
    if cut_distance is not None and (not np.isfinite(cut_distance) or cut_distance < 0):
        raise ValueError('Cut distance must be non-negative and finite')
    fig = plt.figure(figsize=(15, 14), layout='constrained')
    grid = fig.add_gridspec(2, 2, height_ratios=[3, 10], width_ratios=[1, 0.035])
    tree_ax = fig.add_subplot(grid[0, 0])
    heat_ax = fig.add_subplot(grid[1, 0], sharex=tree_ax)
    color_ax = fig.add_subplot(grid[1, 1])
    drawn = dendrogram(hierarchy['linkage'], ax=tree_ax, no_labels=True,
                       color_threshold=0, above_threshold_color='#285779')
    order = np.array(drawn['leaves'], dtype=int)
    if not np.array_equal(order, hierarchy['order']):
        raise ValueError('Dendrogram order differs from heatmap order')
    tree_ax.set_title('Average-linkage hierarchy of paper embeddings')
    tree_ax.set_ylabel('Merge distance\n(1 − cosine similarity)')
    tree_ax.tick_params(axis='x', bottom=False, labelbottom=False)
    tree_ax.spines[['top', 'right']].set_visible(False)
    ordered = hierarchy['similarity'][np.ix_(order, order)]
    masked = np.ma.array(ordered, mask=np.eye(n, dtype=bool))
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('#eeeeee')
    image = heat_ax.imshow(masked, cmap=cmap, vmin=low, vmax=high, interpolation='nearest',
                           aspect='auto', extent=(0, 10 * n, 10 * n, 0))
    ticks = np.unique(np.linspace(0, n - 1, min(n, max_tick_labels), dtype=int))
    positions = 10 * ticks + 5
    labels = [str(paper_labels[order[i]]) for i in ticks]
    heat_ax.set_xticks(positions, labels, rotation=90, fontsize=8)
    heat_ax.set_yticks(positions, labels, fontsize=8)
    heat_ax.set_xlim(0, 10 * n)
    heat_ax.set_xlabel('Papers in dendrogram order (labels thinned for readability)')
    heat_ax.set_ylabel('Same paper order; self-similarity masked')
    heat_ax.set_title('Cosine similarity in the original embedding space')
    fig.colorbar(image, cax=color_ax, label='Cosine similarity',
                 extend='both' if (low, high) != (-1, 1) else 'neither')
    if cut_distance is not None:
        tree_ax.axhline(cut_distance, color='black', linestyle='--', linewidth=1)
        groups = fcluster(hierarchy['linkage'], t=cut_distance, criterion='distance')[order]
        for boundary in np.flatnonzero(np.diff(groups)) + 1:
            heat_ax.axvline(10 * boundary, color='black', linewidth=0.6)
            heat_ax.axhline(10 * boundary, color='black', linewidth=0.6)
    return fig, {'dendrogram': tree_ax, 'heatmap': heat_ax, 'colorbar': color_ax}


def plot_kmeans_selection(sweeps, *, selected_k=5, title=None):
    """Compare best-inertia-fit and seed-mean silhouettes on identical scales.

    Each named DataFrame must provide k, silhouette_best_fit and silhouette_mean.
    Plot diagnostics as recorded; do not select a fit by maximizing silhouette.
    """
    import matplotlib.pyplot as plt

    if not sweeps or isinstance(selected_k, bool) or not isinstance(selected_k, (int, np.integer)) or selected_k < 2:
        raise ValueError('Supply named sweeps and an integer selected_k of at least two')
    required = ['k', 'silhouette_best_fit', 'silhouette_mean']
    prepared = []
    for name, table in sweeps.items():
        if not set(required) <= set(table):
            raise ValueError(f'{name}: missing K-means diagnostic columns')
        table = table.sort_values('k').copy()
        if table.empty or table.k.duplicated().any():
            raise ValueError(f'{name}: expected distinct cluster counts')
        if (not np.isfinite(table[required].to_numpy(dtype=float)).all()
                or not table.k.ge(2).all() or not table.k.mod(1).eq(0).all()
                or not table.silhouette_best_fit.between(-1, 1).all()
                or not table.silhouette_mean.between(-1, 1).all()):
            raise ValueError(f'{name}: invalid cluster counts or silhouettes')
        selected = table.loc[table.k.eq(selected_k)]
        if len(selected) != 1:
            raise ValueError(f'{name}: selected k={selected_k} was not tested')
        prepared.append((name, table, selected.iloc[0]))

    fig, best_ax = plt.subplots(figsize=(12.5, 6.8))
    mean_ax = best_ax.twinx()
    fig.subplots_adjust(left=.09, right=.90, bottom=.18, top=.80)
    colors = ['#17678a', '#c05a24', '#57823b', '#8b5aa5']
    handles = []
    for i, (name, table, selected) in enumerate(prepared):
        color = colors[i % len(colors)]
        best, = best_ax.plot(table.k, table.silhouette_best_fit,
            color=color, marker='o', markersize=5, linewidth=2,
            label=f'{name}: best-fit silhouette (left)')
        mean, = mean_ax.plot(table.k, table.silhouette_mean,
            color=color, marker='D', markerfacecolor='white', markersize=4,
            linewidth=1.5, linestyle='--', alpha=.85,
            label=f'{name}: mean silhouette (right)')
        handles.extend([best, mean])
        for ax, metric in [(best_ax, 'silhouette_best_fit'), (mean_ax, 'silhouette_mean')]:
            ax.scatter([selected_k], [selected[metric]], s=200, marker='o',
                facecolors='none', edgecolors='#c49720', linewidths=2, zorder=6)

    values = np.concatenate([table[['silhouette_best_fit', 'silhouette_mean']].to_numpy().ravel()
                             for _, table, _ in prepared])
    span = max(float(np.ptp(values)), .05)
    limits = (min(0, float(values.min()) - .12 * span), min(1, float(values.max()) + .2 * span))
    best_ax.set_ylim(*limits)
    mean_ax.set_ylim(*limits)
    # Keep both axes identical when zooming or otherwise changing either limit.
    def sync_limits(source, target):
        if source.get_ylim() != target.get_ylim():
            target.set_ylim(source.get_ylim(), emit=False)
    best_ax.callbacks.connect('ylim_changed', lambda ax: sync_limits(ax, mean_ax))
    mean_ax.callbacks.connect('ylim_changed', lambda ax: sync_limits(ax, best_ax))
    best_ax.set(xlabel='Number of clusters (k)', ylabel='Cosine silhouette: best-inertia fit')
    mean_ax.set_ylabel('Cosine silhouette: mean across seeds')
    counts = sorted(set(int(k) for _, table, _ in prepared for k in table.k))
    best_ax.set_xticks(counts)
    best_ax.axvspan(selected_k - .18, selected_k + .18, color='#d9b954', alpha=.16, zorder=0)
    best_ax.axvline(selected_k, color='#997b20', linewidth=1, linestyle=':', alpha=.7)
    best_ax.text(selected_k, 1.015, f'Selected: {selected_k} clusters',
                 transform=best_ax.get_xaxis_transform(), ha='center', fontsize=10, color='#715710')
    best_ax.grid(axis='y', color='#d9dfe3', linewidth=.7, alpha=.8)
    best_ax.set_axisbelow(True)
    best_ax.spines['top'].set_visible(False)
    mean_ax.spines['top'].set_visible(False)
    fig.suptitle(title or 'PDFs and abstracts: K-means silhouette across cluster counts',
                 fontsize=15, fontweight='bold', y=.98)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .93),
               ncol=2, frameon=False, fontsize=10)
    fig.text(.09, .025, 'Both axes share the same scale. Best fit = lowest-inertia seed; '
             'mean = average silhouette across seeds.', fontsize=9, color='#555555')
    return fig, {'best_fit': best_ax, 'mean': mean_ax}
