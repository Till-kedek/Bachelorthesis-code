"""Prepare the approved corpus and independent Final clones. Never rewrite originals."""
from pathlib import Path
import json
import shutil
import sys
import textwrap

import nbformat
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from batill.abstract_clustering import load_abstract_embeddings
from batill.storage import file_hash, read_json, write_json

SCREEN = ROOT / 'reports/abstract_screening/5779f6345a8f5d18'
CLUSTER = 'Clustering Abstracts Final.ipynb'
TOPIC = 'Topic_analysis_Abstracts_v2 Final.ipynb'
AGES = 'Topic_analysis_Ages_v3 test Final.ipynb'


def clone(name):
    notebook = nbformat.read(ROOT / name, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    notebook.metadata.pop('widgets', None)
    return notebook


def save(notebook, name):
    path = ROOT / name
    if path.exists():
        raise FileExistsError(f'Final notebook already exists; preserve its results: {path}')
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            compile(cell.source, name, 'exec')
    nbformat.validate(notebook)
    nbformat.write(notebook, path)


def main():
    for name in (CLUSTER, TOPIC, AGES, 'configs/abstract_final_input.json'):
        if (ROOT / name).exists():
            raise FileExistsError(f'Final workflow already prepared; run its notebooks: {name}')
    # Pin the approved run rather than selecting whichever folder is newest.
    manifest = read_json(SCREEN / 'manifest.json')
    assert manifest['retained_papers'] == 2056
    for name, digest in manifest['artifacts_sha256'].items():
        assert file_hash(SCREEN / name) == digest, name
    protected_files = list(ROOT.glob('*.ipynb'))
    protected_dirs = [ROOT / relative for relative in (
        'outputs/selected_abstracts', 'outputs/pdf_text_clusters',
        'reports/topic_age_analysis_v3_test', 'reports/topic_age_analysis_v2',
        'data/abstracts', 'configs',
        'outputs/abstract_embeddings/bge_m3/20260930T174845_508054Z_full_3c498a9d')]
    protected_files += [p for folder in protected_dirs for p in folder.rglob('*') if p.is_file()]
    write_json(ROOT / 'reports/abstract_final/protected_before.json', {
        str(p.relative_to(ROOT)): file_hash(p) for p in protected_files
        if 'final' not in str(p.relative_to(ROOT)).lower()})

    destination = ROOT / 'data/abstracts/final' / manifest['run_id']
    run = ROOT / 'outputs/abstract_embeddings/final' / manifest['run_id']
    destination.mkdir(parents=True, exist_ok=True)
    if run.exists():
        raise FileExistsError(f'Final embedding run already exists: {run}')
    shutil.copy2(SCREEN / 'abstracts_clean.csv', destination / 'abstracts_clean.csv')
    shutil.copy2(SCREEN / 'embedding_row_map.csv', destination / 'embedding_row_map.csv')
    shutil.copytree(SCREEN / 'embedding_run', run)
    summary = read_json(run / 'run_summary.json')
    summary['source'] = str(destination / 'abstracts_clean.csv')
    summary['approved_screening_run'] = manifest['run_id']
    write_json(run / 'run_summary.json', summary)
    papers, vectors, _ = load_abstract_embeddings(run, destination / 'abstracts_clean.csv')
    assert len(papers) == 2056
    assert np.array_equal(vectors, np.load(SCREEN / 'embedding_run/embeddings.npy'))
    checked_files = [*destination.iterdir(), *run.iterdir(), SCREEN / 'manifest.json', SCREEN / 'paper_decisions.csv']
    write_json(ROOT / 'configs/abstract_final_input.json', {
        'schema_version': 1, 'papers': len(papers), 'dimensions': vectors.shape[1],
        'screening_run': manifest['run_id'],
        'screening_decisions': str((SCREEN / 'paper_decisions.csv').relative_to(ROOT)),
        'source_csv': str((destination / 'abstracts_clean.csv').relative_to(ROOT)),
        'run_dir': str(run.relative_to(ROOT)),
        'input_hashes': {str(p.relative_to(ROOT)): file_hash(p) for p in checked_files},
    })

    notebook = clone('Clustering Abstracts.ipynb')
    notebook.cells[0].source = textwrap.dedent('''
        # Clustering Abstracts — Final

        Approved broad Scopus corpus: **2,056 papers × 1,024 embedding dimensions**.
        Inputs are pinned by `configs/abstract_final_input.json` to screening run
        `5779f6345a8f5d18`. All 1,763 excluded papers are absent from this CSV and
        embedding matrix. Vectors are exact original rows; no embedding inference.

        Run top to bottom with **Python (batill)**. Clustering is refitted on this
        corpus: previous memberships are not reused. The full K-means/Leiden sweeps
        remain available. The shared selected settings for topic/age analysis are
        **K-means k=5** and **Leiden resolution 0.45**, with its actual cluster count.
        PDF and Elite Papers Abstracts inputs are separate and unchanged.
    ''').strip()
    for cell in notebook.cells:
        cell.source = cell.source.replace('Clustering Abstracts.ipynb', CLUSTER)
        cell.source = cell.source.replace(
            'from batill.abstract_scope import resolve_abstract_inputs, abstract_partition_folder',
            'from batill.abstract_final import resolve_final_inputs')
        cell.source = cell.source.replace('resolve_abstract_inputs(ROOT, use_original=True)', 'resolve_final_inputs(ROOT)')
        cell.source = cell.source.replace('Original abstract corpus:', 'Approved final abstract corpus:')
        cell.source = cell.source.replace('Original embedding run:', 'Final aligned embedding run:')
        cell.source = cell.source.replace('SELECTED_LEIDEN_RESOLUTION = 0.5', 'SELECTED_LEIDEN_RESOLUTION = 0.45')
        cell.source = cell.source.replace('INSPECT_K = 6', 'INSPECT_K = 5')
        cell.source = cell.source.replace('| {6}', '| {5}')
        if cell.cell_type == 'markdown':
            cell.source = cell.source.replace('k=6', 'k=5').replace('resolution 0.5', 'resolution 0.45')
    # Match the float32 source-distance convention used by catalog validation.
    cell = notebook.cells[38]
    start = cell.source.index('# Reuse exact full-corpus distances;')
    end = cell.source.index('\nsubsample_indices =', start)
    cell.source = (cell.source[:start] +
        '# Compute distances from the same saved vectors used by topic/catalog validation.\n'
        'distance_matrix = cosine_distances(embeddings)\n'
        'distance_matrix = np.maximum((distance_matrix + distance_matrix.T) / 2, 0)\n'
        'np.fill_diagonal(distance_matrix, 0)\n' + cell.source[end:])
    notebook.cells[45].source = '## Save the Final catalog and selected memberships\n\nSave authenticated partitions for the full sweep, then export K-means k=5 and Leiden 0.45. All outputs are isolated by the new corpus fingerprint.'
    notebook.cells[46].source = textwrap.dedent('''
        from batill.abstract_final import load_final_context, export_final_selected
        from batill.abstract_topic_explorer import partition_catalog
        final_context = load_final_context(ROOT)
        assert final_context['papers'].paper_id.tolist() == papers.Key.tolist()
        for k, fitted in kmeans_results.items():
            assert np.array_equal(final_context['solutions'][f'kmeans_k{k}']['labels'], fitted['labels'])
        for resolution, fitted in leiden_results.items():
            assert np.array_equal(final_context['solutions'][f'leiden_r{resolution:g}']['labels'], fitted['labels'])
        selected_export = export_final_selected(ROOT, final_context)
        display(partition_catalog(final_context))
        print('Saved Final selected assignments:', selected_export)
        print('Next: Topic_analysis_Abstracts_v2 Final.ipynb, then Topic_analysis_Ages_v3 test Final.ipynb')
    ''').strip()
    save(notebook, CLUSTER)

    notebook = clone('Topic_analysis_Abstracts_v2.ipynb')
    notebook.cells[0].source = textwrap.dedent('''
        # Topic analysis Abstracts v2 — Final

        **2,056 approved broad Scopus abstracts** with their exact original vectors.
        Run **Clustering Abstracts Final.ipynb** first. This notebook requires its
        saved Final catalog and never falls back to the original corpus.

        The default is Leiden 0.45; choose `kmeans_k5` for the other selected
        partition. Both receive new topic evidence based only on retained papers.
        Exports go to `outputs/topic_analysis_abstracts_final/`.
    ''').strip()
    notebook.cells[1].source = '## 1 Load Final inputs and list saved partitions\n\nThe pinned CSV, vector rows and saved clustering catalog are validated. If the catalog is missing, run Clustering Abstracts Final.ipynb first; this topic notebook does not silently fit replacements.'
    for cell in notebook.cells:
        cell.source = cell.source.replace('load_context(ROOT, use_original=True)', 'load_final_context(ROOT, build_missing=False)')
        cell.source = cell.source.replace('outputs/topic_analysis_abstracts_v2', 'outputs/topic_analysis_abstracts_final')
    notebook.cells[2].source = notebook.cells[2].source.replace(
        'context = load_final_context', 'from batill.abstract_final import load_final_context\n\ncontext = load_final_context')
    notebook.cells[20].source = notebook.cells[20].source.replace('SAVE_RESULTS = False', 'SAVE_RESULTS = True')
    notebook.cells.append(nbformat.v4.new_markdown_cell('## Export the companion selected partition\n\nKeep complete topic tables for both settings used by the Final age analysis. The interactive view above remains your selected partition.'))
    notebook.cells.append(nbformat.v4.new_code_cell(textwrap.dedent('''
        for companion in ('kmeans_k5', 'leiden_r0.45'):
            if partition_settings.get('saved') == companion:
                continue
            companion_labels, companion_settings = choose_partition(context, source='saved', saved=companion)
            companion_analysis = analyse_partition(context, companion_labels, min_df=3, max_df=0.9,
                min_cluster_papers=MIN_CLUSTER_PAPERS, min_cluster_fraction=MIN_CLUSTER_FRACTION)
            companion_export = export_analysis(companion_analysis, context, companion_settings,
                display_settings={'ranking': RANKING, 'kind': TERM_KIND, 'top_n': TOP_N})
            print('Saved companion topic evidence:', companion, companion_export)
    ''').strip()))
    save(notebook, TOPIC)

    config = read_json(ROOT / 'configs/topic-age-v2-inputs.json')
    config.update(abstract_input='final', input_config='configs/topic-age-final-inputs.json',
                  topic_workbooks={'abstracts': TOPIC})
    for spec in config['partitions']:
        if spec['analysis'] == 'abstracts':
            spec['label'] = spec['label'].replace('Abstracts', 'Final abstracts')
            if spec['solution'].startswith('leiden_'):
                spec['clusters'] = 'actual'
    write_json(ROOT / config['input_config'], config)
    notebook = clone('Topic_analysis_Ages_v3 test.ipynb')
    for cell in notebook.cells:
        cell.source = cell.source.replace('Topic analysis Ages v3 test', 'Topic analysis Ages v3 test — Final')
        cell.source = cell.source.replace('configs/topic-age-v2-inputs.json', config['input_config'])
        cell.source = cell.source.replace('reports/topic_age_analysis_v3_test', 'reports/topic_age_analysis_v3_test_final')
        cell.source = cell.source.replace('3,819', '2,056')
        cell.source = cell.source.replace('five-cluster memberships', 'selected memberships')
        cell.source = cell.source.replace('selected selected memberships', 'selected memberships')
        cell.source = cell.source.replace('five clusters', 'selected clusters')
        cell.source = cell.source.replace('five percentages', 'cluster percentages')
        cell.source = cell.source.replace('five fixed clusters', 'fixed clusters')
        cell.source = cell.source.replace('Every partition must have exactly five assigned clusters.',
            'K-means, PDF and citation counts remain fixed; Final abstract Leiden uses its actual count.')
    notebook.cells[0].source = notebook.cells[0].source.replace(
        'The Leiden resolutions are the five-cluster choices highlighted in the clustering\ngraphs.',
        'Only the broad abstract corpus is replaced by the approved 2,056 papers.\nAbstract memberships are freshly fitted by the Final clustering notebook; Leiden\n0.45 may yield a different count. PDF/citation memberships and Elite Papers Abstracts\nremain unchanged. Original historical age reports are preserved.')
    notebook.cells[11].source = notebook.cells[11].source.replace("INSPECT_ANALYSIS = 'citations'", "INSPECT_ANALYSIS = 'abstracts'").replace("INSPECT_SOLUTION = 'citation_r0.9'", "INSPECT_SOLUTION = 'leiden_r0.45'")
    save(notebook, AGES)
    print('Prepared Final corpus:', len(papers), vectors.shape)
    print('Run in order:', CLUSTER, '→', TOPIC, '→', AGES)


if __name__ == '__main__':
    main()
