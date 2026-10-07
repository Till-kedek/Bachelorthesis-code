"""Validate the current three-corpus age report and the saved 1,531-paper k=6 fit."""
from pathlib import Path
import sys

import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from batill.abstract_experiment import load_saved_experiment
from batill.storage import file_hash, read_json, write_json


def main():
    config = read_json(ROOT / 'configs/topic-age-final-inputs.json')
    out = ROOT / 'reports/topic_age_analysis_v3_test_final'
    previous = ROOT / config['previous_report_archive'] / 'reports/topic_age_analysis_v3_test_final'
    expected = {('pdfs', 'embedding_leiden_k5'), ('abstracts', 'kmeans_k6'), ('citations', 'citation_r0.9')}
    selected = pd.read_csv(out / 'selected_partitions.csv')
    assert set(zip(selected.analysis, selected.solution)) == expected
    assert selected.analysis.is_unique
    assert selected.set_index('analysis').clusters.to_dict() == {'pdfs': 5, 'abstracts': 6, 'citations': 5}
    experiment = load_saved_experiment(ROOT, config['abstract_experiment'], ['kmeans_k6'])
    members = pd.read_csv(out / 'paper_memberships_and_years.csv')
    abstracts = members.loc[members.analysis.eq('abstracts')]
    assert len(abstracts) == 1531 and abstracts.paper_id.is_unique
    assert abstracts.paper_id.tolist() == experiment['papers'].paper_id.tolist()
    np.testing.assert_array_equal(abstracts.cluster, experiment['solutions']['kmeans_k6']['labels'])
    assert set(abstracts.cluster) == set(range(6)) and abstracts.assigned.all()
    counts = pd.read_csv(out / 'cluster_counts_and_shares.csv')
    assert config['share_basis'] == 'cluster'
    assert len(counts) == (5 + 6 + 5) * len(config['periods'])
    for (analysis, solution), group in members.groupby(['analysis', 'solution']):
        rows = counts.loc[counts.analysis.eq(analysis) & counts.solution.eq(solution)]
        assert rows.papers.sum() == group.assigned.sum()
        totals = rows.groupby('cluster').share_pct.sum(min_count=1)
        np.testing.assert_allclose(totals, 100)
        cluster_sizes = group.loc[group.assigned].groupby('cluster').size()
        np.testing.assert_array_equal(rows.cluster_papers, rows.cluster.map(cluster_sizes))
        np.testing.assert_allclose(rows.share_pct, 100 * rows.papers / rows.cluster_papers)
    names = pd.read_csv(out / 'cluster_expression_names.csv')
    assert names.groupby('analysis').size().to_dict() == {'abstracts': 6, 'citations': 5, 'pdfs': 5}
    assert names.phrase_count.eq(8).all()
    for name in ('paper_memberships_and_years.csv', 'cluster_expression_names.csv',
                 'cluster_label_evidence.csv', 'cluster_counts_and_shares.csv', 'bin_coverage.csv'):
        new, old = pd.read_csv(out / name), pd.read_csv(previous / name)
        for analysis, solution in expected - {('abstracts', 'kmeans_k6')}:
            left = new.loc[new.analysis.eq(analysis) & new.solution.eq(solution)].reset_index(drop=True)
            right = old.loc[old.analysis.eq(analysis) & old.solution.eq(solution)].reset_index(drop=True)
            if name == 'cluster_counts_and_shares.csv':
                left = left.drop(columns=['share_pct', 'cluster_papers'])
                right = right.drop(columns=['share_pct'])
            pd.testing.assert_frame_equal(left, right, check_dtype=False)
    expected_figures = {f'{analysis}_{solution}_by_year_bin.{extension}'
                        for analysis, solution in expected for extension in ('png', 'pdf', 'svg')}
    assert {p.name for p in (out / 'figures').iterdir() if p.is_file()} == expected_figures
    manifest = read_json(out / 'run_manifest.json')
    assert manifest['share_denominator'] == 'all papers in the same assigned cluster across the configured year bins'
    for name, digest in manifest['artifact_sha256'].items():
        assert file_hash(out / name) == digest
    assert manifest['input_config_sha256'] == file_hash(ROOT / config['input_config'])
    notebook = ROOT / 'Topic_analysis_Ages_v3 test Final.ipynb'
    book = nbformat.read(notebook, as_version=4)
    nbformat.validate(book)
    assert all(c.execution_count is not None for c in book.cells if c.cell_type == 'code')
    assert not any(o.output_type == 'error' for c in book.cells if c.cell_type == 'code' for o in c.outputs)
    result = {'status': 'passed', 'abstract_papers': len(abstracts), 'abstract_clusters': 6,
              'one_method_per_corpus': True, 'exact_saved_abstract_memberships': True,
              'pdf_and_citation_memberships_and_counts_unchanged': True,
              'year_shares_sum_to_100_per_cluster': True,
              'figures': len(expected_figures), 'table_rows': len(counts),
              'notebook_sha256': file_hash(notebook),
              'input_config_sha256': manifest['input_config_sha256']}
    write_json(ROOT / 'reports/abstract_final/age_selection_validation.json', result)
    print(result)


if __name__ == '__main__':
    main()
