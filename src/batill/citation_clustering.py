"""Leiden on direct citation links, aligned to the existing text analysis by hash.

The directed source graph remains untouched. Community detection uses a binary
undirected edge if either work cites the other. Isolates receive label 0, meaning
unassigned, and are excluded from partition agreement calculations.
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from .storage import read_json


def load_citation_corpus(graph_dir, papers, selection_id):
    """Validate the selection and resolve canonical work rows without positional joins."""
    graph_dir = Path(graph_dir)
    manifest = read_json(graph_dir / 'run_manifest.json')
    if manifest['selection_id'] != selection_id:
        raise ValueError('Citation graph and embedding corpus selections differ; rebuild the reviewed graph')
    nodes = pd.read_csv(graph_dir / 'nodes.csv').fillna('')
    edges = pd.read_csv(graph_dir / 'edges.csv').fillna('')
    mapping = pd.read_csv(graph_dir / 'pdf_to_work.csv').fillna('')
    if not papers.Key.is_unique or not nodes.Key.is_unique or not mapping.Key.is_unique:
        raise ValueError('Paper, work and mapping keys must be unique')
    if set(mapping.Key) != set(papers.Key) or set(mapping.canonical_key) != set(nodes.Key):
        raise ValueError('Citation mapping does not cover exactly the selected PDFs and graph works')
    if not nodes.approved.eq(True).all():
        raise ValueError('Resolve unapproved work identities before this citation analysis')
    canon = mapping.set_index('Key').canonical_key
    if not all(key in canon.index and canon[key] == key for key in nodes.Key):
        raise ValueError('Each graph node must be its own canonical PDF representative')
    edge_keys = set(edges.source_key) | set(edges.target_key)
    if not edge_keys <= set(nodes.Key):
        raise ValueError('Citation edge references a work absent from nodes.csv')
    if edges.duplicated(['source_key', 'target_key']).any() or edges.source_key.eq(edges.target_key).any():
        raise ValueError('Expected unique directed citations with no self-links')
    paper_rows = pd.Series(np.arange(len(papers)), index=papers.Key)
    nodes['embedding_row'] = nodes.Key.map(paper_rows).astype(int)
    nodes['pdf_count'] = nodes.Key.map(mapping.groupby('canonical_key').size())
    expected_ids = nodes.Key.map(papers.set_index('Key')['Paper ID'])
    if not expected_ids.eq(nodes.paper_id).all():
        raise ValueError('Canonical Paper IDs disagree with the embedding manifest')
    return nodes, edges, mapping


def citation_network(nodes, edges, *, include_manual=True):
    """Binary undirected union of directed citations, with all works retained."""
    if not nodes.Key.is_unique:
        raise ValueError('Work keys must be unique')
    selected = edges.copy()
    if not include_manual:
        if 'citation_source' not in selected:
            raise ValueError('Edge provenance is required for the API-only sensitivity check')
        selected = selected.loc[selected.citation_source.eq('openalex')]
    index = pd.Series(np.arange(len(nodes)), index=nodes.Key)
    source, target = selected.source_key.map(index), selected.target_key.map(index)
    if source.isna().any() or target.isna().any():
        raise ValueError('Unknown citation endpoint')
    if source.eq(target).any() or selected.duplicated(['source_key', 'target_key']).any():
        raise ValueError('Self-links and duplicate directed edges are not permitted')
    directed = sparse.csr_matrix((np.ones(len(selected)), (source.astype(int), target.astype(int))),
                                 shape=(len(nodes), len(nodes)))
    adjacency = directed.maximum(directed.T).tocsr()
    adjacency.data[:] = 1
    upper = sparse.triu(adjacency, k=1).tocoo()
    count, components = connected_components(adjacency, directed=False)
    degree = np.asarray(adjacency.sum(axis=1)).ravel().astype(int)
    component_sizes = np.bincount(components)
    active = np.flatnonzero(degree > 0)
    return {'adjacency': adjacency, 'directed': directed,
            'edges': np.column_stack([upper.row, upper.col]),
            'active': active, 'degree': degree, 'components': components,
            'component_sizes': component_sizes, 'include_manual': bool(include_manual),
            'diagnostics': {'works': len(nodes), 'directed_citations': len(selected),
                'undirected_links': len(upper.data), 'clustered_works': len(active),
                'isolated_works': int(np.sum(degree == 0)), 'components': int(count),
                'nonisolated_components': int(np.sum(component_sizes > 1)),
                'largest_component_works': int(component_sizes.max()) if count else 0}}


def citation_leiden_sweep(network, resolutions, *, seeds=(42, 43, 44), max_clusters=15):
    """Run RB-configuration Leiden on non-isolates; labels retain full work order.

    Choose the best objective only among seeds at the SAME resolution. Neither
    objective nor fixed-resolution modularity determines a unique correct k.
    Save every seed's membership for reproducibility and stability assessment.
    """
    import igraph as ig
    import leidenalg

    resolutions = [float(r) for r in resolutions]
    seeds = list(seeds)
    if not resolutions or len(set(resolutions)) != len(resolutions) or any(not np.isfinite(r) or r <= 0 for r in resolutions):
        raise ValueError('Resolutions must be unique, positive and finite')
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError('Supply at least two distinct seeds')
    active = network['active']
    if len(active) < 2 or not len(network['edges']):
        raise ValueError('No citation links to cluster; inspect graph coverage')
    indices = {old: new for new, old in enumerate(active)}
    edge_list = [(indices[a], indices[b]) for a, b in network['edges']]
    g = ig.Graph(n=len(active), edges=edge_list, directed=False)
    largest = int(np.argmax(network['component_sizes']))
    rows, results = [], {}
    for resolution in resolutions:
        runs = []
        for seed in seeds:
            partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=resolution, seed=int(seed), n_iterations=-1)
            raw = np.asarray(partition.membership)
            ids = {label: i + 1 for i, label in enumerate(dict.fromkeys(raw))}
            labels = np.zeros(len(network['degree']), dtype=int)
            labels[active] = [ids[label] for label in raw]
            runs.append({'labels': labels, 'seed': int(seed), 'quality': float(partition.quality()),
                         'modularity_gamma_1': float(g.modularity(raw.tolist()))})
        best = max(runs, key=lambda r: r['quality'])
        labels = best['labels']
        counts = pd.Series(labels[active]).value_counts()
        stability = [adjusted_rand_score(a['labels'][active], b['labels'][active]) for a, b in combinations(runs, 2)]
        internal = labels[network['edges'][:, 0]] == labels[network['edges'][:, 1]]
        rows.append({'resolution': resolution, 'communities': len(counts),
            'within_target': 2 <= len(counts) <= max_clusters,
            'largest_component_communities': len(set(labels[network['components'] == largest]) - {0}),
            'smallest': int(counts.min()), 'largest': int(counts.max()),
            'singletons': int(counts.eq(1).sum()), 'unassigned_isolates': len(labels) - len(active),
            'within_community_link_fraction': float(internal.mean()),
            'modularity_gamma_1': best['modularity_gamma_1'],
            'seed_stability_ARI_mean': float(np.mean(stability)),
            'seed_stability_ARI_min': float(np.min(stability)), 'selected_seed': best['seed']})
        results[resolution] = {**best, 'resolution': resolution, 'runs': runs}
    return pd.DataFrame(rows), results


def plot_citation_resolution_selection(diagnostics, selected_resolution=.9):
    """Display citation modularity and seed agreement without refitting a partition.

    Modularity is evaluated at gamma=1 for every tested partition, irrespective
    of its fitting resolution. Each metric uses its own labelled vertical axis.
    """
    import matplotlib.pyplot as plt

    required = ['resolution', 'communities', 'modularity_gamma_1',
                'seed_stability_ARI_mean']
    if not set(required) <= set(diagnostics):
        raise ValueError('Missing citation sweep diagnostic columns')
    table = diagnostics.sort_values('resolution').copy()
    if table.empty or table.resolution.round(12).duplicated().any():
        raise ValueError('Expected distinct tested citation resolutions')
    if not np.isfinite(table[required].to_numpy(dtype=float)).all():
        raise ValueError('Citation diagnostic values must be finite')
    matches = np.isclose(table.resolution, selected_resolution, rtol=0, atol=1e-12)
    if matches.sum() != 1:
        raise ValueError('Highlighted resolution must match exactly one tested value')
    selected = table.loc[matches].iloc[0]

    fig, modularity_ax = plt.subplots(figsize=(12.5, 6.8))
    stability_ax = modularity_ax.twinx()
    fig.subplots_adjust(left=.09, right=.90, bottom=.18, top=.80)
    color = '#17678a'
    modularity_line, = modularity_ax.plot(
        table.resolution, table.modularity_gamma_1,
        color=color, marker='o', markersize=5, linewidth=2,
        label='Citations: modularity at gamma = 1 (left)')
    stability_line, = stability_ax.plot(
        table.resolution, table.seed_stability_ARI_mean,
        color=color, marker='D', markersize=4, linewidth=1.5,
        linestyle='--', alpha=.65, label='Citations: mean seed stability ARI (right)')
    for row in table.itertuples():
        modularity_ax.annotate(str(int(row.communities)),
            (row.resolution, row.modularity_gamma_1), xytext=(0, 10),
            textcoords='offset points', ha='center', fontsize=8, color=color,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .8, 'pad': .1})
    modularity_ax.axvline(selected.resolution, color=color, linewidth=1,
                         linestyle=':', alpha=.6)
    modularity_ax.text(selected.resolution, 1.015,
        f'Citations: {int(selected.communities)} clusters at {selected.resolution:g}',
        transform=modularity_ax.get_xaxis_transform(), ha='center', color=color, fontsize=9)
    for ax, metric in [(modularity_ax, 'modularity_gamma_1'),
                       (stability_ax, 'seed_stability_ARI_mean')]:
        ax.scatter([selected.resolution], [selected[metric]], s=220,
            marker='o', facecolors='none', edgecolors='#c49720', linewidths=2.5, zorder=6)

    values = table.modularity_gamma_1.to_numpy()
    span = max(float(np.ptp(values)), .05)
    modularity_ax.set_ylim(min(0, float(values.min()) - .08 * span),
                           min(1, float(values.max()) + .3 * span))
    min_ari = float(table.seed_stability_ARI_mean.min())
    stability_ax.set_ylim(min(0, min_ari - .05) if min_ari < 0 else 0, 1.05)
    modularity_ax.set(xlabel='Leiden resolution', ylabel='Modularity (gamma = 1)')
    stability_ax.set_ylabel('Seed stability: mean pairwise adjusted Rand index')
    modularity_ax.set_xticks(table.resolution, [f'{r:g}' for r in table.resolution],
                             rotation=45, ha='right')
    modularity_ax.grid(axis='y', color='#d9dfe3', linewidth=.7, alpha=.8)
    modularity_ax.set_axisbelow(True)
    for ax in (modularity_ax, stability_ax):
        ax.spines['top'].set_visible(False)
    fig.suptitle('Citation clustering: modularity and stability across Leiden resolutions',
                 fontsize=15, fontweight='bold', y=.98)
    fig.legend(handles=[modularity_line, stability_line], loc='upper center',
               bbox_to_anchor=(.5, .93), ncol=2, frameon=False, fontsize=10)
    fig.text(.09, .025, 'Numbers beside solid points = clusters; rings = highlighted resolution. '
             'Each metric uses its labelled axis.', fontsize=9, color='#555555')
    return fig, {'modularity': modularity_ax, 'stability': stability_ax}


def citation_cluster_tables(nodes, network, partition, representatives=3):
    """Representatives have highest within-community degree, then internal in-degree."""
    labels = np.asarray(partition['labels'])
    if len(labels) != len(nodes) or representatives < 1:
        raise ValueError('Labels must align with work rows and representatives must be positive')
    membership = nodes.copy()
    membership['citation_label'] = labels
    membership['Citation community'] = ['Unassigned (no links)' if i == 0 else f'C{i}' for i in labels]
    membership['citation_degree'] = network['degree']
    membership['component_size'] = network['component_sizes'][network['components']]
    membership['within_community_degree'] = 0
    incoming = np.asarray(network['directed'].sum(axis=0)).ravel()
    summary = []
    for label in sorted(set(labels) - {0}):
        group = np.flatnonzero(labels == label)
        degree = np.asarray(network['adjacency'][group][:, group].sum(axis=1)).ravel().astype(int)
        membership.loc[membership.index[group], 'within_community_degree'] = degree
        order = np.lexsort((group, -incoming[group], -degree))
        chosen = group[order[:representatives]]
        summary.append({'Community': f'C{label}', 'Works': len(group),
                        'Within-community links': int(degree.sum() // 2),
                        'Representative papers': '\n'.join(f"{nodes.iloc[i].paper_id}: {nodes.iloc[i].query_title}" for i in chosen)})
    return pd.DataFrame(summary), membership


def partition_comparison(citation_labels, text_labels):
    """Chance-adjusted agreement and overlap counts on assigned canonical works.

    Citation label 0 is unassigned. Text label 0 is valid (e.g. K-means).
    Cluster numbers have no shared semantic meaning across the two methods.
    """
    citation_labels, text_labels = np.asarray(citation_labels), np.asarray(text_labels)
    if citation_labels.ndim != 1 or text_labels.shape != citation_labels.shape:
        raise ValueError('Partitions must have the same one-dimensional row order')
    if pd.isna(text_labels).any():
        raise ValueError('Text labels are missing for some canonical works')
    keep = citation_labels > 0
    if keep.sum() < 2:
        raise ValueError('At least two assigned works are needed for comparison')
    left, right = citation_labels[keep], text_labels[keep].astype(str)
    overlap = pd.crosstab(pd.Series([f'C{i}' for i in left], name='Citation community'),
                          pd.Series(right, name='Text cluster'))
    overlap = overlap.reindex(sorted(overlap.index, key=lambda s: int(s[1:])))
    metrics = {'compared_works': int(keep.sum()), 'unassigned_works_excluded': int((~keep).sum()),
               'citation_communities': len(np.unique(left)), 'text_clusters': len(np.unique(right)),
               'ARI': float(adjusted_rand_score(left, right)),
               'AMI': float(adjusted_mutual_info_score(left, right))}
    return metrics, overlap


def citation_umap_comparison(nodes, papers, citation_labels, text_labels, *, text_name, resolution):
    """Two panels on identical EXISTING UMAP coordinates; never fit a projection.

    Colours are matched one-to-one by largest overlap only to aid visual comparison.
    This permutes colours, never memberships. One point per canonical work is shown.
    """
    import html
    import textwrap
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.optimize import linear_sum_assignment

    required = ['UMAP 1', 'UMAP 2']
    if not set(required) <= set(papers.columns):
        raise ValueError('Run the earlier UMAP cell first; its coordinates are reused')
    if not papers.Key.is_unique or not set(nodes.Key) <= set(papers.Key):
        raise ValueError('UMAP paper keys do not cover canonical works uniquely')
    frame = papers.set_index('Key').loc[nodes.Key, required].reset_index()
    if not np.isfinite(frame[required].to_numpy(dtype=float)).all():
        raise ValueError('UMAP coordinates must be finite')
    metrics, overlap = partition_comparison(citation_labels, text_labels)
    labels = np.asarray(citation_labels)
    frame['Citation community'] = ['Unassigned (no links)' if i == 0 else f'C{i}' for i in labels]
    frame['Text cluster'] = np.asarray(text_labels).astype(str)
    frame['paper_id'] = nodes.paper_id.to_numpy()
    frame['Title'] = nodes.query_title.to_numpy()
    frame['Paper'] = frame.Title.map(lambda title: '<br>'.join(html.escape(line) for line in textwrap.wrap(title, 75)))
    text_order = sorted(frame['Text cluster'].unique())
    palette = px.colors.qualitative.Alphabet
    text_colors = {name: palette[i % len(palette)] for i, name in enumerate(text_order)}
    row, col = linear_sum_assignment(-overlap.to_numpy())
    paired = {overlap.index[i]: overlap.columns[j] for i, j in zip(row, col) if overlap.iloc[i, j] > 0}
    citation_order = [f'C{i}' for i in sorted(set(labels) - {0})]
    citation_colors, extra = {}, len(text_colors)
    for name in citation_order:
        if name in paired:
            citation_colors[name] = text_colors[paired[name]]
        else:
            citation_colors[name] = palette[extra % len(palette)]
            extra += 1
    citation_colors['Unassigned (no links)'] = '#a6a6a6'
    if (labels == 0).any():
        citation_order.append('Unassigned (no links)')
    fig = make_subplots(rows=1, cols=2, shared_xaxes=True, shared_yaxes=True,
                        subplot_titles=[text_name, f'Direct-citation Leiden · resolution {resolution}'])
    for column, field, order, colors in [(1, 'Text cluster', text_order, text_colors),
                                         (2, 'Citation community', citation_order, citation_colors)]:
        for category in order:
            subset = frame.loc[frame[field].eq(category)]
            fig.add_trace(go.Scatter(x=subset['UMAP 1'], y=subset['UMAP 2'], mode='markers',
                name=f'{"Text" if column == 1 else "Citation"}: {category}',
                legendgroup=f'{column}-{category}',
                marker={'color': colors[category], 'size': 8, 'opacity': .85,
                        'symbol': 'x' if category == 'Unassigned (no links)' else 'circle'},
                customdata=subset[['paper_id', 'Paper', 'Citation community', 'Text cluster']].to_numpy(),
                hovertemplate='%{customdata[0]}<br>%{customdata[1]}<br>Citation: %{customdata[2]}<br>Text: %{customdata[3]}<extra></extra>'), row=1, col=column)
    for axis in ['UMAP 1', 'UMAP 2']:
        values = frame[axis].to_numpy(); padding = max(float(np.ptp(values)) * .05, .1)
        bounds = [float(values.min()-padding), float(values.max()+padding)]
        if axis == 'UMAP 1':
            fig.update_xaxes(range=bounds, title_text=axis)
        else:
            fig.update_yaxes(range=bounds, title_text=axis)
    fig.update_layout(template='plotly_white', height=660, width=1250,
                      title='Same text-embedding UMAP · one point per unique work · colours paired by overlap')
    return fig, frame, pd.DataFrame([{'Citation community': key, 'Text cluster': value} for key, value in paired.items()])
