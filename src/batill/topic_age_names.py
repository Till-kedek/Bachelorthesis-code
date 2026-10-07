"""Eight-expression cluster labels for the Ages v3 test notebook's exact memberships."""

import ast
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd

from .storage import file_hash, fingerprint, read_json
from .topic_analysis import build_vocabulary, topic_terms
from .topic_age_v2 import bin_matrix, load_memberships
from .topic_explorer import ranked_terms


WORKBOOKS = {
    'pdfs': 'Topic_analysis_PDFs_v2.ipynb',
    'abstracts': 'Topic_analysis_Abstracts_v2.ipynb',
    'citations': 'Topic_analysis_Citations_v2.ipynb',
}


def workbook_term_settings(path):
    """Read literal v2 ranking/filter controls without executing a notebook."""
    controls = {'RANKING': 'tfidf_contrast', 'MIN_CLUSTER_PAPERS': 2,
                'MIN_CLUSTER_FRACTION': .1}
    settings = {'min_df': 3, 'max_df': .9}
    for cell in read_json(path)['cells']:
        if cell['cell_type'] != 'code':
            continue
        tree = ast.parse(''.join(cell['source']))
        for statement in tree.body:
            if isinstance(statement, ast.Assign):
                for target in statement.targets:
                    if isinstance(target, ast.Name) and target.id in controls:
                        controls[target.id] = ast.literal_eval(statement.value)
            for node in ast.walk(statement):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == 'analyse_partition'):
                    for keyword in node.keywords:
                        if keyword.arg in ('min_df', 'max_df', 'min_cluster_papers', 'min_cluster_fraction'):
                            settings[keyword.arg] = (controls[keyword.value.id]
                                if isinstance(keyword.value, ast.Name) else ast.literal_eval(keyword.value))
    return {'ranking': controls['RANKING'], 'min_cluster_papers': controls['MIN_CLUSTER_PAPERS'],
            'min_cluster_fraction': controls['MIN_CLUSTER_FRACTION'], **settings}


def expression_names(papers, labels, *, analysis, solution, settings, top_n=8):
    """Compute labels once across the whole partition, never separately by year bin."""
    labels = np.asarray(labels)
    if labels.shape != (len(papers),) or labels.dtype.kind not in 'iu' or (labels < 0).any():
        raise ValueError('Cluster IDs must align with source papers')
    active = labels > 0 if analysis == 'citations' else np.ones(len(labels), dtype=bool)
    vocabulary = build_vocabulary(papers.loc[active], min_df=settings['min_df'], max_df=settings['max_df'])
    terms = topic_terms(vocabulary, labels[active], top_n=len(vocabulary['terms']),
                        min_cluster_fraction=settings['min_cluster_fraction'],
                        min_cluster_papers=settings['min_cluster_papers'])
    view = {'labels': labels[active], 'terms': terms}
    stable_ids = papers.paper_id if analysis == 'abstracts' else papers.pdf_sha256
    records, evidence = [], []
    for cluster in sorted(set(labels[active])):
        rows = ranked_terms(view, cluster, settings['ranking'], 'expression', top_n)
        phrases = rows.term.tolist()
        if not phrases:
            raise ValueError(f'No qualifying two-word expressions for {analysis}/{solution}/{cluster}')
        if any(len(phrase.split()) != 2 for phrase in phrases):
            raise ValueError('Cluster labels require two-word expressions')
        membership_id = fingerprint(sorted(stable_ids.loc[labels == cluster].astype(str).tolist()))
        record = {'analysis': analysis, 'solution': solution, 'cluster': int(cluster),
                  'topic_name': ' · '.join(phrases),
                  'display_cluster': f"Cluster {cluster} — " + ' · '.join(phrases),
                  'phrase_count': len(phrases), 'ranking': settings['ranking'],
                  'membership_id': membership_id,
                  'text_id': fingerprint(sorted(zip(stable_ids.loc[labels == cluster].astype(str),
                                                   papers.loc[labels == cluster, 'text'].map(fingerprint))))}
        record.update({f'expression_{i}': phrase for i, phrase in enumerate(phrases, 1)})
        records.append(record)
        rows = rows.copy()
        rows['analysis'], rows['solution'] = analysis, solution
        rows['expression_rank'] = np.arange(1, len(rows) + 1)
        evidence.append(rows)
    return pd.DataFrame(records), pd.concat(evidence, ignore_index=True)


def attach_names(members, names):
    """Require exact membership identities before attaching labels to an age table."""
    keys = ['analysis', 'solution', 'cluster']
    if names.duplicated(keys).any():
        raise ValueError('Duplicate cluster labels')
    expected = members.loc[members.assigned].groupby(keys).stable_id.apply(
        lambda values: fingerprint(sorted(values.astype(str).tolist())))
    recorded = names.set_index(keys).membership_id
    if set(expected.index) != set(recorded.index) or not expected.eq(recorded.reindex(expected.index)).all():
        raise ValueError('Expression labels do not match the exact age-analysis memberships')
    result = members.drop(columns=['topic_name', 'display_cluster']).merge(
        names[keys + ['topic_name', 'display_cluster']], on=keys, how='left', validate='many_to_one')
    result.loc[~result.assigned, ['topic_name', 'display_cluster']] = 'Unassigned'
    pd.testing.assert_frame_equal(result[members.columns.difference(['topic_name', 'display_cluster'])],
                                  members[members.columns.difference(['topic_name', 'display_cluster'])])
    return result


def load_named_memberships(root, config, *, top_n=8):
    root = Path(root).resolve()
    members, catalog, provenance, sources = load_memberships(root, config, with_sources=True)
    all_names, all_evidence = [], []
    workbooks = {**WORKBOOKS, **config.get('topic_workbooks', {})}
    settings_by_corpus = {kind: workbook_term_settings(root / workbook)
                          for kind, workbook in workbooks.items()}
    for spec in config['partitions']:
        kind, solution = spec['analysis'], spec['solution']
        source = sources[kind]
        names, evidence = expression_names(
            source['papers'], source['solutions'][solution]['labels'],
            analysis=kind, solution=solution, settings=settings_by_corpus[kind], top_n=top_n)
        names['source_workbook'] = workbooks[kind]
        names['derivation'] = 'v2 ranking rules applied to exact saved age-analysis memberships'
        all_names.append(names)
        all_evidence.append(evidence)
    names = pd.concat(all_names, ignore_index=True)
    evidence = pd.concat(all_evidence, ignore_index=True)
    members = attach_names(members, names)
    provenance['expression_labels'] = {
        'top_n': top_n, 'kind': 'two-word expression', 'settings': settings_by_corpus,
        'workbooks': {name: file_hash(root / name) for name in workbooks.values()},
        'code_sha256': file_hash(Path(__file__)),
        'scope': 'Full assigned partition; labels fixed across year bins',
        'pdf_note': 'PDF v2 historical memberships differ; its ranking rules are applied to the saved PDF age partitions.'}
    return members, catalog, provenance, names, evidence


def plot_named_bins(counts, spec, periods, names, *, share_basis='period'):
    """Full expression names beside composition bars by period or by cluster."""
    import matplotlib.pyplot as plt

    if share_basis not in {'period', 'cluster'}:
        raise ValueError('share_basis must be period or cluster')
    numbers = bin_matrix(counts, spec['analysis'], spec['solution'], periods)
    shares = bin_matrix(counts, spec['analysis'], spec['solution'], periods, 'share_pct')
    label_rows = names.loc[names.analysis.eq(spec['analysis']) & names.solution.eq(spec['solution'])].sort_values('cluster')
    if label_rows.display_cluster.tolist() != numbers.columns.tolist():
        raise ValueError('Figure labels do not match count-table columns')
    if share_basis == 'cluster':
        expected = numbers.div(numbers.sum(axis=0), axis=1) * 100
        np.testing.assert_allclose(shares, expected, equal_nan=True)
    colors = ['#236b8e', '#e39c35', '#459b7d', '#ab5c78', '#8674b1']
    if len(label_rows) > len(colors):
        # Keep expression labels readable on white when a partition has six+ clusters.
        palette = 'tab10' if len(label_rows) <= 10 else 'tab20'
        colors = [plt.get_cmap(palette)(i % 20) for i in range(len(label_rows))]
    fig, axes = plt.subplots(1, 2, figsize=(18, 9), layout='constrained',
                             gridspec_kw={'width_ratios': [1.05, 1]})
    values = numbers.to_numpy(dtype=int).T
    axes[0].imshow(values, cmap='Blues', aspect='auto', vmin=0, vmax=max(1, int(values.max())))
    for (row, col), value in np.ndenumerate(values):
        axes[0].text(col, row, str(value), ha='center', va='center', fontsize=11,
                     color='white' if value > values.max() * .55 else '#182b3a')
    tick_labels = [f'Cluster {row.cluster}\n' + textwrap.fill(row.topic_name, width=46,
                                                            break_long_words=False, break_on_hyphens=False)
                   for row in label_rows.itertuples()]
    axes[0].set(xticks=range(len(numbers)), xticklabels=numbers.index,
                yticks=range(len(label_rows)), yticklabels=tick_labels,
                xlabel='Publication year', title='Papers per cluster and year bin')
    for tick, color in zip(axes[0].get_yticklabels(), colors):
        tick.set_color('#182b3a' if share_basis == 'cluster' else color)
        tick.set_fontsize(10)
    if share_basis == 'cluster':
        # Both panels use the same cluster order; colours identify publication bins.
        totals = numbers.sum(axis=0).to_numpy()
        left = np.zeros(len(label_rows))
        for index, period in enumerate(numbers.index):
            widths = shares.loc[period].fillna(0).to_numpy()
            color = ['#236b8e', '#e39c35', '#459b7d', '#8674b1'][index % 4]
            axes[1].barh(range(len(label_rows)), widths, left=left, color=color,
                         height=.65, label=period)
            for row, width in enumerate(widths):
                if width >= 7:
                    axes[1].text(left[row] + width / 2, row, f'{width:.1f}%',
                                 ha='center', va='center', fontsize=10)
            left += widths
        axes[1].set(yticks=range(len(label_rows)),
                    yticklabels=[f'Cluster {cluster} (n={n:,})'
                                 for cluster, n in zip(label_rows.cluster, totals)],
                    ylim=(len(label_rows) - .5, -.5), xlim=(0, 100),
                    xlabel='Papers in this cluster (%)', title='Year distribution within each cluster')
        axes[1].legend(loc='upper center', bbox_to_anchor=(.5, -.08), ncols=2,
                        frameon=False, title='Publication year')
        fig.suptitle(spec['label'] + ' · top 8 two-word expressions', fontsize=16)
        return fig
    totals = numbers.sum(axis=1).to_numpy()
    left = np.zeros(len(numbers))
    for index, row in enumerate(label_rows.itertuples()):
        widths = shares[row.display_cluster].fillna(0).to_numpy()
        axes[1].barh(range(len(numbers)), widths, left=left, color=colors[index], height=.65,
                      label=f'Cluster {row.cluster}')
        for period, width in enumerate(widths):
            if width >= 7:
                axes[1].text(left[period] + width / 2, period, f'{width:.1f}%',
                             ha='center', va='center', fontsize=10)
        left += widths
    for index, total in enumerate(totals):
        if total == 0:
            axes[1].text(50, index, 'No assigned papers', ha='center', va='center')
    axes[1].set(yticks=range(len(numbers)),
                yticklabels=[f'{period} (n={n:,})' for period, n in zip(numbers.index, totals)],
                xlim=(0, 100), xlabel='Assigned papers in this bin (%)', title='Cluster shares within each bin')
    axes[1].invert_yaxis()
    axes[1].legend(loc='upper center', bbox_to_anchor=(.5, -.08), ncols=3, frameon=False,
                    title='Colours match the expression labels on the left')
    fig.suptitle(spec['label'] + ' · top 8 two-word expressions', fontsize=16)
    return fig
