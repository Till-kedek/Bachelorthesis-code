"""Offline publication-age comparisons of immutable, named topic partitions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd


GROUP = ['analysis', 'solution', 'cluster', 'display_cluster', 'topic_name', 'assigned']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_csv(root, spec):
    path = Path(root) / spec['path']
    if not path.is_file() or sha256(path) != spec['sha256']:
        raise ValueError(f"Frozen input missing or changed: {spec['path']}. Restore the shared snapshot.")
    return pd.read_csv(path, keep_default_na=False)


def validate_years(values, paper_ids, reference_year, minimum_year=1800):
    years = pd.to_numeric(values, errors='coerce')
    valid = (years.notna() & np.isfinite(years) & years.mod(1).eq(0)
             & years.between(minimum_year, reference_year))
    if not valid.all():
        raise ValueError(f'Missing, invalid or future publication years: {list(paper_ids[~valid])}')
    return years.astype(int)


def period_labels(years, periods):
    labels = pd.Series('', index=years.index, dtype='str')
    hits = pd.Series(0, index=years.index)
    for period in periods:
        mask = years.between(period['start'], period['end'])
        labels.loc[mask] = period['label']
        hits += mask.astype(int)
    if not hits.eq(1).all():
        raise ValueError('Every publication year must belong to exactly one configured period.')
    return labels


def _named_memberships(root, analysis, spec):
    members = checked_csv(root, spec['memberships'])
    profiles = checked_csv(root, spec['profiles'])
    keys = ['solution', 'cluster']
    if members.duplicated(['solution', 'paper_id']).any() or profiles.duplicated(keys).any():
        raise ValueError(f'Duplicate membership or profile identities: {analysis}')
    if set(members.solution) != set(spec['solutions']):
        raise ValueError(f'Unexpected solutions: {analysis}')
    members['assigned'] = ~((analysis == 'citations') & members.cluster.eq(0))
    expected_ids = None
    for solution, count in spec['solutions'].items():
        selected = members.loc[members.solution.eq(solution)]
        ids = set(selected.paper_id)
        if len(selected) != spec['expected_papers_per_solution'] or (expected_ids is not None and ids != expected_ids):
            raise ValueError(f'Incomplete or different corpus: {analysis}/{solution}')
        expected_ids = ids
        if selected.loc[selected.assigned, 'cluster'].nunique() != count:
            raise ValueError(f'Wrong fixed cluster count: {analysis}/{solution}')
    assigned = members.loc[members.assigned]
    if not assigned.proposal_matched.eq(True).all() or not profiles.proposal_matched.eq(True).all():
        raise ValueError(f'Unmatched topic names: {analysis}')
    actual = assigned.groupby(keys).agg(papers=('paper_id', 'size'), names=('proposed_label', 'nunique'))
    expected = profiles.set_index(keys)
    if set(actual.index) != set(expected.index) or not actual.names.eq(1).all():
        raise ValueError(f'Profile/membership mismatch: {analysis}')
    if not actual.papers.eq(expected.papers.reindex(actual.index)).all():
        raise ValueError(f'Profile counts disagree: {analysis}')
    joined = assigned.merge(profiles[keys + ['proposed_label']], on=keys, validate='many_to_one', suffixes=('', '_profile'))
    if not joined.proposed_label.eq(joined.proposed_label_profile).all():
        raise ValueError(f'Profile names disagree: {analysis}')
    return members, profiles


def load_paper_ages(root, config):
    """Join by stable identities, preserving PDF aliases and citation isolates."""
    nodes = checked_csv(root, config['reviewed_nodes'])
    mapping = checked_csv(root, config['pdf_to_work'])
    if nodes.Key.duplicated().any() or nodes.paper_id.duplicated().any() or mapping.Key.duplicated().any():
        raise ValueError('Duplicate reviewed metadata identities')
    metadata = nodes[['Key', 'paper_id', 'openalex_id', 'openalex_year', 'query_year', 'query_title', 'doi']].rename(
        columns={'Key': 'canonical_key', 'paper_id': 'canonical_paper_id', 'query_title': 'metadata_title', 'doi': 'metadata_doi'})
    rows, all_profiles = [], []
    for analysis, spec in config['sources'].items():
        members, profiles = _named_memberships(root, analysis, spec)
        frame = members[['solution', 'paper_id', 'cluster', 'proposed_label', 'assigned']].copy()
        frame['analysis'] = analysis
        frame['unit'] = spec['unit']
        frame['source_notebook'] = spec['source_notebook']
        frame['source_memberships'] = spec['memberships']['path']
        frame['topic_name'] = frame.pop('proposed_label')
        frame['display_cluster'] = [
            ('Unassigned' if not assigned else f'C{cluster}' if analysis == 'citations'
             else f'L{cluster}' if 'leiden' in solution else str(cluster))
            for solution, cluster, assigned in zip(frame.solution, frame.cluster, frame.assigned)]
        if analysis == 'abstracts':
            frame['title'] = members.title
            frame['doi'] = members.doi
            frame['document_type'] = members.document_type
            frame['source_year'] = members.year
            frame['alternative_year'] = members.year
            frame['year_source'] = 'Scopus: saved topic membership year'
            frame['alternative_year_source'] = 'Same Scopus year (no alternative source)'
            frame['canonical_paper_id'] = members.paper_id
            frame['canonical_key'] = members.paper_id
            frame['is_duplicate_pdf'] = False
        else:
            frame['pdf_sha256'] = members.pdf_sha256 if analysis == 'pdfs' else members.Key
            frame['title'] = members.title if analysis == 'pdfs' else members.query_title
            frame = frame.merge(mapping.rename(columns={'Key': 'pdf_sha256'})[
                ['pdf_sha256', 'paper_id', 'canonical_key', 'canonical_paper_id', 'is_duplicate_pdf']],
                on=['pdf_sha256', 'paper_id'], how='left', validate='many_to_one', indicator=True)
            if not frame.pop('_merge').eq('both').all():
                raise ValueError(f'Missing reviewed PDF/work identity: {analysis}')
            frame = frame.merge(metadata, on=['canonical_key', 'canonical_paper_id'],
                                how='left', validate='many_to_one', indicator=True)
            if not frame.pop('_merge').eq('both').all():
                raise ValueError(f'Missing publication metadata: {analysis}')
            if analysis == 'citations' and frame.is_duplicate_pdf.any():
                raise ValueError('Citation units must already be canonical works')
            frame['doi'] = frame.pop('metadata_doi')
            frame['source_year'] = frame.openalex_year
            frame['alternative_year'] = frame.query_year
            frame['year_source'] = 'OpenAlex: reviewed nodes.openalex_year'
            frame['alternative_year_source'] = 'Reviewed nodes.query_year (bibliographic/PDF query metadata)'
        frame['publication_year'] = validate_years(frame.source_year, frame.paper_id,
            config['reference_year'], config['minimum_year'])
        frame['alternative_year'] = validate_years(frame.alternative_year, frame.paper_id,
            config['reference_year'], config['minimum_year'])
        frame['age_years'] = config['reference_year'] - frame.publication_year
        frame['reference_year'] = config['reference_year']
        frame['year_difference'] = frame.publication_year - frame.alternative_year
        frame['period'] = period_labels(frame.publication_year, config['periods'])
        # Prove that joins neither drop/duplicate papers nor change their cluster/name.
        before = members[['solution', 'paper_id', 'cluster', 'proposed_label']].rename(columns={'proposed_label': 'topic_name'})
        after = frame[before.columns]
        order = ['solution', 'paper_id']
        pd.testing.assert_frame_equal(before.sort_values(order).reset_index(drop=True),
                                      after.sort_values(order).reset_index(drop=True))
        rows.append(frame)
        profiles.insert(0, 'analysis', analysis)
        all_profiles.append(profiles)
    papers = pd.concat(rows, ignore_index=True).sort_values(['analysis', 'solution', 'cluster', 'paper_id']).reset_index(drop=True)
    return papers, pd.concat(all_profiles, ignore_index=True)


def summarize_clusters(papers):
    summary = papers.groupby(GROUP, sort=True).agg(
        papers=('paper_id', 'size'), dated_papers=('publication_year', 'count'),
        median_year=('publication_year', 'median'), mean_year=('publication_year', 'mean'),
        first_year=('publication_year', 'min'), last_year=('publication_year', 'max'),
        median_age=('age_years', 'median'), mean_age=('age_years', 'mean'),
        q25_age=('age_years', lambda x: x.quantile(.25)),
        q75_age=('age_years', lambda x: x.quantile(.75)),
        youngest_age=('age_years', 'min'), oldest_age=('age_years', 'max'),
    ).reset_index()
    summary['small_cluster'] = summary.papers.lt(10)
    return summary


def temporal_counts(papers, periods=None):
    """All assigned clusters × every time bin; shares use the bin's corpus total."""
    records = []
    for (analysis, solution), frame in papers.groupby(['analysis', 'solution'], sort=True):
        assigned = frame.loc[frame.assigned]
        labels = assigned[GROUP].drop_duplicates().sort_values('cluster')
        if periods is None:
            bins = [{'label': str(year), 'start': year, 'end': year}
                    for year in range(int(frame.publication_year.min()), int(frame.reference_year.iloc[0]) + 1)]
        else:
            bins = periods
        for period in bins:
            selected = frame.loc[frame.publication_year.between(period['start'], period['end'])]
            n_assigned = int(selected.assigned.sum())
            counts = selected.loc[selected.assigned].groupby('cluster').size()
            for row in labels.to_dict('records'):
                n = int(counts.get(row['cluster'], 0))
                records.append({**row, 'period': period['label'], 'start_year': period['start'],
                    'end_year': period['end'], 'papers': n, 'period_assigned_papers': n_assigned,
                    'period_all_papers': len(selected), 'period_unassigned_papers': len(selected) - n_assigned,
                    'share_pct': 100 * n / n_assigned if n_assigned else np.nan})
    return pd.DataFrame(records)


def period_changes(periods, early, late):
    keys = [x for x in GROUP if x != 'assigned']
    columns = keys + ['papers', 'period_assigned_papers', 'share_pct']
    before = periods.loc[periods.period.eq(early), columns]
    after = periods.loc[periods.period.eq(late), columns]
    change = before.merge(after, on=keys, suffixes=('_early', '_late'), validate='one_to_one')
    change['early_period'] = early
    change['late_period'] = late
    change['change_percentage_points'] = change.share_pct_late - change.share_pct_early
    return change.sort_values(['analysis', 'solution', 'change_percentage_points'], ascending=[True, True, False])


def analyze(papers, config):
    summary = summarize_clusters(papers)
    periods = temporal_counts(papers, config['periods'])
    years = temporal_counts(papers)
    changes = period_changes(periods, *config['comparison_periods'])
    alternative = papers.copy()
    alternative['publication_year'] = alternative.alternative_year
    alternative['age_years'] = config['reference_year'] - alternative.publication_year
    alt_summary = summarize_clusters(alternative)
    alt_periods = temporal_counts(alternative, config['periods'])
    alt_changes = period_changes(alt_periods, *config['comparison_periods'])
    sensitivity = summary[GROUP + ['median_year', 'mean_year', 'median_age']].merge(
        alt_summary[GROUP + ['median_year', 'mean_year', 'median_age']], on=GROUP,
        suffixes=('_primary', '_alternative'), validate='one_to_one')
    sensitivity['median_year_difference'] = sensitivity.median_year_primary - sensitivity.median_year_alternative
    change_keys = [x for x in GROUP if x != 'assigned']
    sensitivity = sensitivity.merge(changes[change_keys + ['change_percentage_points']], on=change_keys,
                                   how='left', validate='one_to_one').merge(
        alt_changes[change_keys + ['change_percentage_points']], on=change_keys,
        how='left', validate='one_to_one', suffixes=('_primary', '_alternative'))
    sensitivity['share_change_sign_agrees'] = [
        '' if pd.isna(a) or pd.isna(b) else bool(np.sign(a) == np.sign(b))
        for a, b in zip(sensitivity.change_percentage_points_primary, sensitivity.change_percentage_points_alternative)]
    coverage = papers.groupby(['analysis', 'solution', 'unit'], sort=True).agg(
        papers=('paper_id', 'size'), dated_papers=('publication_year', 'count'),
        assigned_papers=('assigned', 'sum'), canonical_identities=('canonical_key', 'nunique'),
        duplicate_pdf_units=('is_duplicate_pdf', 'sum'), earliest_year=('publication_year', 'min'),
        latest_year=('publication_year', 'max'),
        differing_years=('year_difference', lambda x: x.ne(0).sum()),
    ).reset_index()
    coverage['unassigned_papers'] = coverage.papers - coverage.assigned_papers
    return {'paper_ages': papers, 'cluster_age_summary': summary, 'period_shares': periods,
            'annual_shares': years, 'period_changes': changes, 'year_sensitivity': sensitivity,
            'alternative_period_shares': alt_periods, 'coverage': coverage}


def write_tables(tables, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(output / f'{name}.csv', index=False, float_format='%.10f', lineterminator='\n')


def make_figures(tables, config, output):
    """Return figures for inline inspection; also save standalone PNG and PDF."""
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    figures = []
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42}):
        for (analysis, solution), frame in tables['paper_ages'].groupby(['analysis', 'solution'], sort=True):
            summary = tables['cluster_age_summary']
            summary = summary.loc[summary.analysis.eq(analysis) & summary.solution.eq(solution) & summary.assigned]
            summary = summary.sort_values(['median_age', 'mean_age', 'cluster'], ascending=[False, False, True])
            labels = ['\n'.join(textwrap.wrap(f'{r.display_cluster}: {r.topic_name}', 46)) + f' (n={r.papers})'
                      for r in summary.itertuples()]
            title = f'{analysis.title()} · {solution}'
            fig, ax = plt.subplots(figsize=(13, max(5.2, len(summary) * 1.0)))
            values = [frame.loc[frame.cluster.eq(c), 'age_years'].to_numpy() for c in summary.cluster]
            ax.boxplot(values, orientation='horizontal', tick_labels=labels, patch_artist=True,
                       boxprops={'facecolor': '#c9deed'}, medianprops={'color': '#173c5b', 'linewidth': 2},
                       flierprops={'marker': '.', 'markersize': 4, 'alpha': .65})
            ax.invert_yaxis()
            ax.set_xlim(left=-.5)
            ax.set_xlabel(f"Paper age in calendar years ({config['reference_year']} − publication year)")
            ax.set_title(title + '\nOlder median at top · box: middle 50% · whiskers: 1.5 × IQR')
            ax.grid(axis='x', alpha=.2)
            fig.tight_layout()
            figures.append((f'{analysis}_{solution}_ages', fig))

            shares = tables['period_shares']
            shares = shares.loc[shares.analysis.eq(analysis) & shares.solution.eq(solution)]
            order = [p['label'] for p in config['periods']]
            matrix = shares.pivot(index='cluster', columns='period', values='share_pct').reindex(index=summary.cluster, columns=order)
            counts = shares.pivot(index='cluster', columns='period', values='papers').reindex(index=summary.cluster, columns=order)
            denominators = shares.groupby('period').period_assigned_papers.first()
            fig, ax = plt.subplots(figsize=(13, max(5.2, len(summary) * 1.0)))
            im = ax.imshow(matrix.to_numpy(), cmap='Blues', vmin=0, vmax=100, aspect='auto')
            ax.set_yticks(range(len(labels)), labels)
            ax.set_xticks(range(len(order)), [f'{p}\nN={denominators[p]}' for p in order])
            ax.set_title(title + '\nCluster share of assigned publications in each period')
            for i in range(len(matrix)):
                for j in range(len(order)):
                    value = matrix.iloc[i, j]
                    ax.text(j, i, 'No data' if pd.isna(value) else f'{value:.1f}%\n(n={counts.iloc[i, j]})',
                            ha='center', va='center', color='white' if value > 55 else '#142b3b')
            ax.set_xlabel('Unequal-length periods; 2026 coverage is incomplete. N = assigned papers in this corpus/period.')
            fig.colorbar(im, ax=ax, label='Within-period share (%)', shrink=.75)
            fig.tight_layout()
            figures.append((f'{analysis}_{solution}_shares', fig))
        for name, fig in figures:
            fig.savefig(output / f'{name}.png', dpi=160, metadata={'Software': 'batill topic age analysis'})
            fig.savefig(output / f'{name}.pdf', metadata={'CreationDate': None, 'ModDate': None})
    return figures


def input_specs(config):
    return [item for spec in config['sources'].values() for item in (spec['memberships'], spec['profiles'])] + [
        config['reviewed_nodes'], config['pdf_to_work']]


def write_manifest(root, config, output):
    import platform
    from importlib.metadata import version
    root, output = Path(root), Path(output)
    manifest = {
        'schema_version': 1, 'workflow': 'publication_ages_of_fixed_named_clusters',
        'clustering_performed': False, 'network_required': False,
        'reference_year': config['reference_year'], 'year_policy': config['year_policy'],
        'inputs_sha256': {spec['path']: spec['sha256'] for spec in input_specs(config)},
        'code_sha256': {p: sha256(root / p) for p in [
            'src/batill/topic_age_analysis.py', 'configs/topic-age-inputs.json']},
        'python': platform.python_version(),
        'packages': {name: version(name) for name in ['numpy', 'pandas', 'matplotlib', 'nbformat', 'ipython']},
        'artifacts_sha256': {p.name: sha256(p) for p in sorted(output.iterdir()) if p.suffix == '.csv'},
        'comparison': 'CSV tables are checked byte for byte; rendered figures and notebook displays are not byte-compared across platforms.',
    }
    (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    return manifest
