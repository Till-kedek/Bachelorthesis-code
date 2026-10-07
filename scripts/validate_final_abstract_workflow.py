"""Validate the Final notebook outputs against the approved funnel and protected inputs."""
from pathlib import Path
import sys

import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from batill.abstract_final import load_final_context, resolve_final_inputs
from batill.abstract_clustering import load_abstract_embeddings
from batill.storage import file_hash, read_json, write_json


def main():
    config = read_json(ROOT / 'configs/abstract_final_input.json')
    run, source = resolve_final_inputs(ROOT)
    papers, vectors, _ = load_abstract_embeddings(run, source)
    audit = pd.read_csv(ROOT / config['screening_decisions'])
    expected = set(audit.loc[audit.retained, 'paper_id'])
    excluded = set(audit.loc[~audit.retained, 'paper_id'])
    assert set(papers.paper_id) == expected and expected.isdisjoint(excluded)
    row_map = pd.read_csv(source.parent / 'embedding_row_map.csv')
    original = ROOT / 'outputs/abstract_embeddings/bge_m3/20260930T174845_508054Z_full_3c498a9d'
    original_vectors = np.load(original / 'embeddings.npy')
    assert papers.paper_id.tolist() == row_map.paper_id.tolist()
    assert np.array_equal(vectors, original_vectors[row_map.parent_embedding_row.to_numpy()])

    context = load_final_context(ROOT, build_missing=False)
    assert set(context['papers'].paper_id) == expected
    for result in context['solutions'].values():
        assert len(result['labels']) == len(expected)
    selected_dir = ROOT / 'outputs/abstract_final' / context['input_id'][:16]
    selected = pd.read_csv(selected_dir / 'selected_assignments.csv')
    for solution, rows in selected.groupby('solution'):
        assert set(rows.paper_id) == expected and rows.paper_id.is_unique
        np.testing.assert_array_equal(rows.cluster, context['solutions'][solution]['labels'])

    topic_exports = []
    for folder in sorted((ROOT / 'outputs/topic_analysis_abstracts_final').iterdir()):
        if not (folder / 'run_manifest.json').exists():
            continue
        manifest = read_json(folder / 'run_manifest.json')
        memberships = pd.read_csv(folder / 'membership.csv')
        assert set(memberships.paper_id) == expected and memberships.paper_id.is_unique
        if manifest['partition_settings'].get('source') == 'saved':
            solution = manifest['partition_settings']['saved']
            actual = memberships.set_index('paper_id').cluster.reindex(context['papers'].paper_id)
            np.testing.assert_array_equal(actual, context['solutions'][solution]['labels'])
        for name, digest in manifest['artifact_sha256'].items():
            assert file_hash(folder / name) == digest
        for path in folder.glob('*.csv'):
            frame = pd.read_csv(path)
            if 'paper_id' in frame:
                assert set(frame.paper_id).issubset(expected), path
        topic_exports.append(str(folder.relative_to(ROOT)))
    assert len(topic_exports) >= 2

    age_dir = ROOT / 'reports/topic_age_analysis_v3_test_final'
    age = pd.read_csv(age_dir / 'paper_memberships_and_years.csv')
    old_age = pd.read_csv(ROOT / 'reports/topic_age_analysis_v3_test/paper_memberships_and_years.csv')
    for solution, rows in age.loc[age.analysis.eq('abstracts')].groupby('solution'):
        assert set(rows.paper_id) == expected and rows.paper_id.is_unique
        actual = rows.set_index('paper_id').cluster.reindex(context['papers'].paper_id)
        np.testing.assert_array_equal(actual, context['solutions'][solution]['labels'])
    for name in ('paper_memberships_and_years.csv', 'cluster_expression_names.csv',
                 'cluster_label_evidence.csv', 'cluster_counts_and_shares.csv', 'bin_coverage.csv'):
        new = pd.read_csv(age_dir / name)
        old = pd.read_csv(ROOT / 'reports/topic_age_analysis_v3_test' / name)
        new = new.loc[~new.analysis.eq('abstracts')].reset_index(drop=True)
        old = old.loc[~old.analysis.eq('abstracts')].reset_index(drop=True)
        pd.testing.assert_frame_equal(new, old, check_dtype=False)
    for name, digest in read_json(age_dir / 'run_manifest.json')['artifact_sha256'].items():
        assert file_hash(age_dir / name) == digest

    notebooks = ['Clustering Abstracts Final.ipynb', 'Topic_analysis_Abstracts_v2 Final.ipynb',
                 'Topic_analysis_Ages_v3 test Final.ipynb']
    for name in notebooks:
        book = nbformat.read(ROOT / name, as_version=4)
        nbformat.validate(book)
        assert all(c.execution_count is not None for c in book.cells if c.cell_type == 'code')
        assert not any(o.output_type == 'error' for c in book.cells if c.cell_type == 'code' for o in c.outputs)

    protected = read_json(ROOT / 'reports/abstract_final/protected_before.json')
    diagnostic_extension = False
    for relative, digest in protected.items():
        if relative == 'Abstract Screening.ipynb' and file_hash(ROOT / relative) != digest:
            # A later, authorized read-only diagnostic extends the screening
            # notebook. Its original approved pipeline must remain intact.
            current = nbformat.read(ROOT / relative, as_version=4)
            snapshot = nbformat.read((ROOT / config['screening_decisions']).parent / 'notebook_source.ipynb', as_version=4)
            assert len(current.cells) == len(snapshot.cells) + 2
            assert all(a.cell_type == b.cell_type and a.source == b.source
                       for a, b in zip(current.cells, snapshot.cells))
            assert current.cells[-1].source.startswith('# PROTECTED_TITLE_DIAGNOSTIC')
            diagnostic_extension = True
            continue
        assert file_hash(ROOT / relative) == digest, relative
    result = {'status': 'passed', 'screening_run': config['screening_run'],
              'retained_papers': len(expected), 'excluded_papers': len(excluded),
              'vector_shape': list(vectors.shape), 'vectors_equal_original_rows': True,
              'catalog_partitions': len(context['solutions']),
              'selected_counts': {s: len(np.unique(context['solutions'][s]['labels']))
                                  for s in ('kmeans_k5', 'leiden_r0.45')},
              'topic_exports': topic_exports,
              'pdf_and_citation_age_tables_unchanged': True,
              'protected_original_files_unchanged': len(protected) - int(diagnostic_extension),
              'screening_pipeline_preserved_with_diagnostic_extension': diagnostic_extension,
              'executed_notebooks': {name: file_hash(ROOT / name) for name in notebooks}}
    write_json(ROOT / 'reports/abstract_final/validation.json', result)
    print(result)


if __name__ == '__main__':
    main()
