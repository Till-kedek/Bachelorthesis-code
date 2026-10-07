"""Publication-bin composition of selected saved partitions, with explicit counts."""

from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from .abstract_topic_explorer import load_context as load_abstract_context
from .citation_topic_explorer import load_context as load_citation_context
from .pdf_topic_analysis import load_pdf_topic_corpus, load_pdf_topic_partitions
from .storage import file_hash, fingerprint, write_json
from .topic_age_analysis import period_labels, temporal_counts, validate_years


def _dated_memberships(papers, labels, spec, config, *, nodes=None):
    """Join years by stable identity, preserving original IDs and isolate status."""
    labels = np.asarray(labels)
    if (labels.shape != (len(papers),) or labels.dtype.kind not in 'iu'
            or np.any(labels < 0) or not papers.paper_id.is_unique):
        raise ValueError('Expected unique paper IDs aligned with nonnegative integer labels')
    frame = papers[['paper_id', 'title']].copy().reset_index(drop=True)
    frame['doi'] = papers['doi'].to_numpy() if 'doi' in papers else ''
    frame['cluster'] = labels
    frame['assigned'] = (labels > 0) if spec['analysis'] == 'citations' else True
    actual_clusters = frame.loc[frame.assigned, 'cluster'].nunique()
    expected_clusters = spec['clusters']
    if expected_clusters != 'actual' and actual_clusters != expected_clusters:
        expected_text = 'five' if expected_clusters == 5 else str(expected_clusters)
        raise ValueError(f"{spec['analysis']}/{spec['solution']} must have exactly {expected_text} assigned clusters")
    if actual_clusters < 2:
        raise ValueError('Age analysis requires at least two assigned clusters')
    if spec['analysis'] == 'abstracts':
        frame['stable_id'] = papers.paper_id.to_numpy()
        frame['publication_year'] = papers.year.to_numpy()
        frame['year_source'] = 'Scopus: saved title-and-abstract metadata'
    else:
        if nodes is None or not nodes.Key.is_unique or not papers.pdf_sha256.is_unique:
            raise ValueError('Unique reviewed work metadata are required for publication-year matching')
        frame['stable_id'] = papers.pdf_sha256.to_numpy()
        metadata = nodes[['Key', 'paper_id', 'openalex_year', 'doi']].rename(columns={
            'Key': 'stable_id', 'paper_id': 'reviewed_paper_id',
            'openalex_year': 'publication_year', 'doi': 'reviewed_doi'})
        frame = frame.merge(metadata, on='stable_id', how='left', validate='one_to_one', indicator=True)
        if (not frame.pop('_merge').eq('both').all()
                or not frame.paper_id.eq(frame.reviewed_paper_id).all()):
            raise ValueError('Missing or inconsistent reviewed publication metadata')
        frame['doi'] = frame.pop('reviewed_doi')
        frame = frame.drop(columns='reviewed_paper_id')
        frame['year_source'] = 'OpenAlex: reviewed nodes.openalex_year (same as verbal complexity)'
    frame['publication_year'] = validate_years(
        frame.publication_year, frame.paper_id, config['reference_year'], config['minimum_year'])
    frame['period'] = period_labels(frame.publication_year, config['periods'])
    frame['reference_year'] = config['reference_year']
    frame['analysis'], frame['solution'] = spec['analysis'], spec['solution']
    frame['partition'] = spec['label']
    frame['resolution'] = spec.get('resolution', np.nan)
    frame['display_cluster'] = [f'Cluster {label}' if assigned else 'Unassigned'
                                for label, assigned in zip(frame.cluster, frame.assigned)]
    # Neutral labels only; names from different memberships must not carry over.
    frame['topic_name'] = frame.display_cluster
    np.testing.assert_array_equal(frame.cluster.to_numpy(), labels)
    if frame.paper_id.tolist() != papers.paper_id.tolist():
        raise ValueError('Publication-year join changed paper order')
    return frame


def load_memberships(root, config, *, with_sources=False):
    """Load exact saved PDF, abstract and citation fits; no clustering fallback."""
    root = Path(root).resolve()
    citations = load_citation_context(root)
    # Historical configurations keep their exact five-cluster requirements.
    # The Final configuration explicitly selects its separately authenticated corpus.
    abstract_input = config.get('abstract_input', 'original')
    if abstract_input == 'experiment':
        from .abstract_experiment import load_saved_experiment
        abstracts = load_saved_experiment(root, config['abstract_experiment'],
            [spec['solution'] for spec in config['partitions'] if spec['analysis'] == 'abstracts'])
    elif abstract_input == 'final':
        from .abstract_final import load_final_context
        abstracts = load_final_context(root, build_missing=False)
    elif abstract_input in {'original', 'active'}:
        abstracts = load_abstract_context(root, build_missing=False,
                                          use_original=abstract_input == 'original')
    else:
        raise ValueError('abstract_input must be original, active, final or experiment')
    input_dir, exclusions = root / 'outputs/corpus_sections', root / 'configs/analysis_exclusions.json'
    pdfs, vectors, _, _, selection, _ = load_pdf_topic_corpus(input_dir, exclusions)
    pdf_dir = root / 'outputs/pdf_text_clusters' / selection['selection_id'][:16] / 'five_clusters'
    pdf_solutions, pdf_manifest = load_pdf_topic_partitions(
        pdf_dir, pdfs, vectors, input_dir=input_dir, exclusion_file=exclusions, selection=selection)
    sources = {
        'pdfs': {'papers': pdfs, 'solutions': pdf_solutions,
                 'evidence': pdf_dir, 'partition_file': pdf_dir / 'assignments.csv',
                 'input_hashes': pdf_manifest['input_hashes']},
        'abstracts': abstracts, 'citations': citations,
    }
    frames, catalog, provenance = [], [], {}
    for spec in config['partitions']:
        source = sources[spec['analysis']]
        if spec['solution'] not in source['solutions']:
            raise ValueError(f"Missing saved partition: {spec['analysis']}/{spec['solution']}")
        result = source['solutions'][spec['solution']]
        if 'resolution' in spec and not np.isclose(
                result.get('resolution', np.nan), spec['resolution'], rtol=0, atol=1e-12):
            raise ValueError(f"Saved Leiden resolution differs from the selected graph: {spec['label']}")
        frame = _dated_memberships(source['papers'], result['labels'], spec, config,
                                   nodes=citations['nodes'])
        frames.append(frame)
        catalog.append({'analysis': spec['analysis'], 'solution': spec['solution'],
                        'partition': spec['label'], 'clusters': int(frame.loc[frame.assigned, 'cluster'].nunique()),
                        'resolution': result.get('resolution', np.nan), 'seed': result['seed'],
                        'papers': len(frame), 'assigned': int(frame.assigned.sum()),
                        'unassigned': int((~frame.assigned).sum()),
                        'year_source': frame.year_source.iloc[0],
                        'membership_id': fingerprint(frame[['stable_id', 'cluster']].to_dict('records'))})
        if spec['analysis'] not in provenance:
            files = [source['partition_file'], *source['evidence'].glob('*manifest.json')]
            provenance[spec['analysis']] = {
                'input_hashes': source['input_hashes'],
                'saved_files': {str(path.relative_to(root)): file_hash(path) for path in files}}
    members = pd.concat(frames, ignore_index=True)
    if members.duplicated(['analysis', 'solution', 'paper_id']).any():
        raise ValueError('Duplicate memberships in the configured partitions')
    # Both PDF/citation views must cover the same canonical works with the same years.
    pdf_years = members.loc[members.analysis.eq('pdfs'), ['stable_id', 'publication_year']].drop_duplicates()
    citation_years = members.loc[members.analysis.eq('citations'), ['stable_id', 'publication_year']].drop_duplicates()
    pd.testing.assert_frame_equal(pdf_years.sort_values('stable_id').reset_index(drop=True),
                                  citation_years.sort_values('stable_id').reset_index(drop=True))
    result = (members, pd.DataFrame(catalog), provenance)
    return (*result, sources) if with_sources else result


def bin_tables(members, periods, *, share_basis='period'):
    """Complete cluster × bin grid, with counts and explicit denominators."""
    if share_basis not in {'period', 'cluster'}:
        raise ValueError('share_basis must be period or cluster')
    counts = temporal_counts(members, periods)
    if share_basis == 'cluster':
        counts['cluster_papers'] = counts.groupby(['analysis', 'solution', 'cluster']).papers.transform('sum')
        counts['share_pct'] = 100 * counts.papers / counts.cluster_papers.replace(0, np.nan)
    coverage = counts[['analysis', 'solution', 'period', 'start_year', 'end_year',
                       'period_all_papers', 'period_assigned_papers',
                       'period_unassigned_papers']].drop_duplicates().reset_index(drop=True)
    for (analysis, solution), frame in members.groupby(['analysis', 'solution']):
        selected = counts.loc[counts.analysis.eq(analysis) & counts.solution.eq(solution)]
        clusters = frame.loc[frame.assigned, 'cluster'].nunique()
        if len(selected) != clusters * len(periods) or int(selected.papers.sum()) != int(frame.assigned.sum()):
            raise ValueError('Bin counts do not preserve all clusters and assigned papers')
    return counts, coverage


def bin_matrix(counts, analysis, solution, periods, metric='papers'):
    selected = counts.loc[counts.analysis.eq(analysis) & counts.solution.eq(solution)]
    if selected.empty or metric not in ('papers', 'share_pct'):
        raise ValueError('Choose an available partition and metric (papers or share_pct)')
    columns = selected[['cluster', 'display_cluster']].drop_duplicates().sort_values('cluster').display_cluster
    return selected.pivot(index='period', columns='display_cluster', values=metric).reindex(
        index=[period['label'] for period in periods], columns=columns)


def plot_partition_bins(counts, spec, periods):
    """Exact count heatmap beside composition bars; recent bin appears first."""
    import matplotlib.pyplot as plt

    numbers = bin_matrix(counts, spec['analysis'], spec['solution'], periods)
    shares = bin_matrix(counts, spec['analysis'], spec['solution'], periods, 'share_pct')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={'width_ratios': [1.2, 1]},
                             layout='constrained')
    values = numbers.to_numpy(dtype=int)
    axes[0].imshow(values, cmap='Blues', aspect='auto', vmin=0, vmax=max(1, int(values.max())))
    for (row, col), value in np.ndenumerate(values):
        axes[0].text(col, row, str(value), ha='center', va='center',
                     color='white' if value > values.max() * .55 else '#182b3a', fontsize=11)
    totals = values.sum(axis=1)
    axes[0].set(xticks=range(len(numbers.columns)), xticklabels=numbers.columns,
                yticks=range(len(numbers)), yticklabels=[f'{p}  (n={n:,})' for p, n in zip(numbers.index, totals)],
                title='Papers in each cluster', xlabel='Original cluster IDs', ylabel='Publication year')
    colors = ['#236b8e', '#e39c35', '#459b7d', '#ab5c78', '#8674b1']
    if len(numbers.columns) > len(colors):
        colors = [plt.get_cmap('tab20')(i % 20) for i in range(len(numbers.columns))]
    left = np.zeros(len(numbers))
    for index, cluster in enumerate(numbers.columns):
        widths = shares[cluster].fillna(0).to_numpy()
        axes[1].barh(range(len(numbers)), widths, left=left, color=colors[index], label=cluster, height=.65)
        for row, width in enumerate(widths):
            if width >= 7:
                axes[1].text(left[row] + width / 2, row, f'{width:.1f}%', ha='center', va='center', fontsize=9)
        left += widths
    for row, total in enumerate(totals):
        if total == 0:
            axes[1].text(50, row, 'No assigned papers', ha='center', va='center', color='#666666')
    axes[1].set(yticks=range(len(numbers)), yticklabels=numbers.index, xlim=(0, 100),
                title='Cluster shares within each bin', xlabel='Assigned papers in this bin (%)')
    axes[1].invert_yaxis()
    axes[1].legend(loc='upper center', bbox_to_anchor=(.5, -.17), ncols=3, frameon=False)
    fig.suptitle(spec['label'], fontsize=15)
    return fig


def export_results(root, destination, config, members, catalog, counts, coverage, provenance):
    root, destination = Path(root), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    tables = {'paper_memberships_and_years.csv': members, 'selected_partitions.csv': catalog,
              'cluster_counts_and_shares.csv': counts, 'bin_coverage.csv': coverage}
    for name, table in tables.items():
        table.to_csv(destination / name, index=False)
    write_json(destination / 'run_manifest.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'config': config,
        'clustering_performed': False, 'sources': provenance,
        'share_denominator': ('all papers in the same assigned cluster across the configured year bins'
                              if config.get('share_basis', 'period') == 'cluster'
                              else 'all assigned papers in the same partition and year bin'),
        'selected_partitions': catalog.astype(object).where(pd.notna(catalog), None).to_dict('records'),
        'input_config_sha256': file_hash(root / config.get('input_config', 'configs/topic-age-v2-inputs.json')),
        'code_sha256': file_hash(Path(__file__)),
        'packages': {name: version(name) for name in ('numpy', 'pandas', 'matplotlib')},
        'artifact_sha256': {str(path.relative_to(destination)): file_hash(path)
                           for path in destination.rglob('*') if path.is_file() and path.name != 'run_manifest.json'},
    })
    return destination
