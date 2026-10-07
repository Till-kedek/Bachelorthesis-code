"""Clone the Final clustering notebook for the 1,531-paper title-screen experiment."""
from pathlib import Path
import textwrap

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NAME = 'Clustering Abstracts Test 1531.ipynb'


def main():
    if (ROOT / NAME).exists():
        raise FileExistsError(f'Preserve the existing experiment: {NAME}')
    notebook = nbformat.read(ROOT / 'Clustering Abstracts Final.ipynb', as_version=4)
    for cell in notebook.cells:
        cell.source = cell.source.replace('Clustering Abstracts Final.ipynb', NAME)
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    notebook.metadata.pop('widgets', None)
    notebook.cells[0].source = textwrap.dedent('''
        # Clustering Abstracts — Test: 1,531 papers

        The same clustering workflow and settings as **Clustering Abstracts Final.ipynb**,
        using the experimental **1,531-paper** corpus. There is no comparison section.

        Start with the approved 2,056 papers. Additionally remove protected title
        matches if their journal fails the keyword rule **OR any PE mention occurs
        in a detected list**: 525 unique papers. The last-third-only rule is **not**
        extended to title matches in this experiment. Earlier exclusions remain.

        The exact approved screening flags are reused, and existing embedding rows
        are selected by paper ID without recomputing vectors. Hierarchy, neighbours,
        PCA/UMAP, Leiden and K-means are fitted to this experimental corpus.
        Run top to bottom with **Python (batill)**. The Final, PDF and Elite Papers
        Abstracts workflows remain unchanged. Outputs are separate and no active
        analysis configuration is switched.
    ''').strip()
    notebook.cells[1].source = notebook.cells[1].source.replace(
        'ABSTRACT_RUN_DIR, ABSTRACT_SOURCE = resolve_final_inputs(ROOT)',
        'PARENT_RUN_DIR, PARENT_SOURCE = resolve_final_inputs(ROOT)')
    notebook.cells[1].source = notebook.cells[1].source.replace(
        "print('Approved final abstract corpus:', ABSTRACT_SOURCE)",
        "print('Approved input for the experiment:', PARENT_SOURCE)")
    notebook.cells[1].source = notebook.cells[1].source.replace(
        "print('Final aligned embedding run:', ABSTRACT_RUN_DIR)",
        "print('Existing vectors to subset:', PARENT_RUN_DIR)")

    notebook.cells[45].source = '## Save the experimental partitions\n\nSave all fitted memberships, per-paper cosine silhouettes, and the selected K-means k=5 and Leiden 0.45 assignments in the experiment folder. No Final workflow outputs are changed.'
    notebook.cells[46].source = textwrap.dedent('''
        from sklearn.metrics import silhouette_samples
        experiment_arrays = {'paper_ids': papers.paper_id.to_numpy(dtype=str)}
        experiment_assignments = []
        partition_rows = []
        for k, fitted in kmeans_results.items():
            solution = f'kmeans_k{k}'
            experiment_arrays[solution] = fitted['labels']
            experiment_arrays[solution + '__silhouettes'] = fitted['silhouettes']
            partition_rows.append({'solution': solution, 'method': 'K-means',
                'clusters': k, 'seed': fitted['seed'], 'resolution': np.nan,
                'mean_cosine_silhouette': float(fitted['silhouettes'].mean())})
        for resolution, fitted in leiden_results.items():
            solution = f'leiden_r{resolution:g}'
            labels = fitted['labels']
            scores = (silhouette_samples(distance_matrix, labels, metric='precomputed')
                      if 1 < len(np.unique(labels)) < len(labels) else np.full(len(labels), np.nan))
            experiment_arrays[solution] = labels
            experiment_arrays[solution + '__silhouettes'] = scores
            partition_rows.append({'solution': solution, 'method': 'Embedding Leiden',
                'clusters': len(np.unique(labels)), 'seed': fitted['seed'], 'resolution': resolution,
                'mean_cosine_silhouette': float(scores.mean())})
        for row in partition_rows:
            solution = row['solution']
            experiment_assignments.append(pd.DataFrame({'paper_id': papers.paper_id,
                'solution': solution, 'cluster': experiment_arrays[solution],
                'cosine_silhouette': experiment_arrays[solution + '__silhouettes']}))
        assignment_table = pd.concat(experiment_assignments, ignore_index=True)
        for solution, rows in assignment_table.groupby('solution'):
            assert rows.paper_id.tolist() == papers.paper_id.tolist()
            assert set(rows.paper_id).isdisjoint(experiment_removed_ids)
        assignment_table.to_csv(EXPERIMENT_DIR / 'all_assignments.csv', index=False)
        selected_assignments = assignment_table.loc[assignment_table.solution.isin(['kmeans_k5', 'leiden_r0.45'])]
        selected_assignments.to_csv(EXPERIMENT_DIR / 'selected_assignments.csv', index=False)
        partition_summary = pd.DataFrame(partition_rows)
        partition_summary.to_csv(EXPERIMENT_DIR / 'partition_summary.csv', index=False)
        np.savez_compressed(EXPERIMENT_DIR / 'partitions.npz', **experiment_arrays)
        kmeans_summary.to_csv(EXPERIMENT_DIR / 'kmeans_diagnostics.csv', index=False)
        leiden_diagnostics.to_csv(EXPERIMENT_DIR / 'leiden_resolution_diagnostics.csv', index=False)
        for relative, digest in protected_hashes.items():
            assert file_hash(ROOT / relative) == digest, f'Protected input changed: {relative}'
        write_json(EXPERIMENT_DIR / 'clustering_manifest.json', {
            'experiment_input_manifest_sha256': file_hash(EXPERIMENT_DIR / 'input_manifest.json'),
            'input_id': kmeans_fit_input_id, 'papers': len(papers),
            'source_notebook': NOTEBOOK_NAME, 'notebook_code_sha256': experiment_code_hash,
            'kmeans': {'k_values': K_VALUES, 'seeds': KMEANS_SEEDS, 'n_init': N_INIT,
                       'selection': 'minimum inertia across seeds', 'subsample_fraction': SUBSAMPLE_FRACTION,
                       'subsample_seeds': SUBSAMPLE_SEEDS},
            'leiden': {'resolutions': LEIDEN_RESOLUTIONS, 'seeds': LEIDEN_SEEDS,
                       'neighbours': LEIDEN_NEIGHBOURS, 'mutual': LEIDEN_MUTUAL,
                       'min_similarity': LEIDEN_MIN_SIMILARITY,
                       'selection': 'highest objective across seeds at each resolution'},
            'diagnostics_directory': str(KMEANS_DIAGNOSTICS_DIR.relative_to(ROOT)),
            'protected_hashes': protected_hashes, 'active_inputs_changed': False,
            'artifact_sha256': {p.name: file_hash(p) for p in EXPERIMENT_DIR.iterdir()
                               if p.is_file() and p.name != 'clustering_manifest.json'},
        })
        display(partition_summary.loc[partition_summary.solution.isin(['kmeans_k5', 'leiden_r0.45'])])
        print('Saved experimental assignments and diagnostics:', EXPERIMENT_DIR)
        print('Verified: 1,531 papers; exact original vectors; approved input files unchanged.')
    ''').strip()

    preparation = nbformat.v4.new_code_cell(textwrap.dedent('''
        import json
        import tempfile
        from batill.storage import file_hash, fingerprint, read_json, write_json

        NOTEBOOK_NAME = 'Clustering Abstracts Test 1531.ipynb'
        final_config = read_json(ROOT / 'configs/abstract_final_input.json')
        approved_audit = pd.read_csv(ROOT / final_config['screening_decisions'], keep_default_na=False)
        experiment_decisions = approved_audit.loc[approved_audit.retained].copy()
        experiment_decisions['experiment_drop_journal'] = (
            experiment_decisions.title_matched & ~experiment_decisions.journal_keyword_match)
        experiment_decisions['experiment_drop_list'] = (
            experiment_decisions.title_matched & experiment_decisions.pe_any_list_mention)
        experiment_decisions['experiment_retained'] = ~(
            experiment_decisions.experiment_drop_journal | experiment_decisions.experiment_drop_list)
        experiment_removed = experiment_decisions.loc[~experiment_decisions.experiment_retained].copy()
        experiment_removed_ids = set(experiment_removed.paper_id)
        experiment_ids = set(experiment_decisions.loc[experiment_decisions.experiment_retained, 'paper_id'])
        assert len(experiment_decisions) == 2056 and len(experiment_removed_ids) == 525 and len(experiment_ids) == 1531
        assert experiment_removed.title_matched.all()
        assert experiment_decisions.loc[~experiment_decisions.title_matched, 'experiment_retained'].all()

        parent_papers, parent_vectors, parent_provenance = load_abstract_embeddings(PARENT_RUN_DIR, PARENT_SOURCE)
        assert set(parent_papers.paper_id) == set(experiment_decisions.paper_id)
        parent_rows = np.flatnonzero(parent_papers.paper_id.isin(experiment_ids).to_numpy())
        experiment_vectors = parent_vectors[parent_rows]
        parent_source = pd.read_csv(PARENT_SOURCE, dtype=str, keep_default_na=False)
        experiment_source = parent_source.loc[parent_source.paper_id.isin(experiment_ids)].copy()
        experiment_selected = pd.read_csv(PARENT_RUN_DIR / 'selected_papers.csv', dtype=str, keep_default_na=False).iloc[parent_rows].copy()
        experiment_selected['embedding_row'] = np.arange(len(parent_rows))
        row_map = pd.DataFrame({'paper_id': experiment_selected.paper_id.to_numpy(),
                               'parent_embedding_row': parent_rows, 'embedding_row': np.arange(len(parent_rows))})
        experiment_tokens = pd.read_csv(PARENT_RUN_DIR / 'token_lengths.csv', dtype=str, keep_default_na=False)
        experiment_tokens = experiment_tokens.set_index('paper_id', drop=False).loc[experiment_selected.paper_id]
        notebook_source = read_json(ROOT / NOTEBOOK_NAME)
        experiment_code_hash = fingerprint([''.join(c['source']) for c in notebook_source['cells'] if c['cell_type'] == 'code'])
        identity = {'approved_input_sha256': file_hash(ROOT / 'configs/abstract_final_input.json'),
                    'notebook_code_sha256': experiment_code_hash,
                    'rule': 'drop retained title matches failing journal OR any-list-mention; no additional position filter',
                    'retained_ids': experiment_selected.paper_id.tolist()}
        EXPERIMENT_DIR = ROOT / 'outputs/abstract_clustering_experiments/title_journal_or_list' / fingerprint(identity)[:16]
        ABSTRACT_RUN_DIR = EXPERIMENT_DIR / 'embedding_run'
        ABSTRACT_SOURCE = EXPERIMENT_DIR / 'abstracts_clean.csv'
        protected_hashes = dict(final_config['input_hashes'])
        # Unrelated notebooks may be edited concurrently; only guard actual inputs.
        protected_paths = [ROOT / 'configs/abstract_final_input.json']
        protected_hashes.update({str(p.relative_to(ROOT)): file_hash(p) for p in protected_paths})
        if not EXPERIMENT_DIR.exists():
            EXPERIMENT_DIR.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='.experiment-', dir=EXPERIMENT_DIR.parent) as temporary:
                staging = Path(temporary) / 'bundle'
                embed_staging = staging / 'embedding_run'
                embed_staging.mkdir(parents=True)
                experiment_source.to_csv(staging / 'abstracts_clean.csv', index=False)
                experiment_decisions.to_csv(staging / 'experiment_decisions.csv', index=False)
                experiment_removed.to_csv(staging / 'removed_title_matches.csv', index=False)
                row_map.to_csv(staging / 'embedding_row_map.csv', index=False)
                experiment_selected.to_csv(embed_staging / 'selected_papers.csv', index=False)
                experiment_selected[['embedding_row', 'paper_id']].to_csv(embed_staging / 'paper_ids.csv', index=False)
                experiment_tokens.to_csv(embed_staging / 'token_lengths.csv', index=False)
                np.save(embed_staging / 'embeddings.npy', experiment_vectors)
                summary = read_json(PARENT_RUN_DIR / 'run_summary.json')
                summary.update(source=str(ABSTRACT_SOURCE), source_sha256=file_hash(staging / 'abstracts_clean.csv'),
                    corpus_size=len(experiment_source), selected_size=len(experiment_source),
                    shape=list(experiment_vectors.shape), parent_run=str(PARENT_RUN_DIR),
                    parent_input_hashes=parent_provenance['input_hashes'],
                    maximum_sample_tokens=int(experiment_selected.token_count.astype(int).max()),
                    maximum_corpus_tokens=int(experiment_selected.token_count.astype(int).max()),
                    derivation='experimental exact subset; title matches screened by journal OR list',
                    experimental=True, screening_stage='approved_final_then_title_journal_or_list')
                write_json(embed_staging / 'run_summary.json', summary)
                load_abstract_embeddings(embed_staging, staging / 'abstracts_clean.csv')
                write_json(staging / 'input_manifest.json', {
                    'identity': identity, 'papers': len(experiment_source), 'removed_title_matches': len(experiment_removed),
                    'shape': list(experiment_vectors.shape), 'embeddings_recomputed': False,
                    'artifacts_sha256': {str(p.relative_to(staging)): file_hash(p) for p in staging.rglob('*') if p.is_file()},
                })
                staging.rename(EXPERIMENT_DIR)
        experiment_manifest = read_json(EXPERIMENT_DIR / 'input_manifest.json')
        assert experiment_manifest['identity'] == identity
        for name, digest in experiment_manifest['artifacts_sha256'].items():
            assert file_hash(EXPERIMENT_DIR / name) == digest, name
        checked_papers, checked_vectors, _ = load_abstract_embeddings(ABSTRACT_RUN_DIR, ABSTRACT_SOURCE)
        assert checked_papers.paper_id.tolist() == experiment_selected.paper_id.tolist()
        assert np.array_equal(checked_vectors, parent_vectors[parent_rows])
        display(pd.DataFrame([
            {'Step': 'Approved input', 'Papers': len(experiment_decisions)},
            {'Step': 'Removed title matches: journal OR list', 'Papers': len(experiment_removed)},
            {'Step': 'Experimental clustering corpus', 'Papers': len(experiment_source)},
        ]))
        print('Experimental input:', ABSTRACT_SOURCE)
        print('Embedding matrix:', checked_vectors.shape)
    ''').strip())
    notebook.cells[2:2] = [nbformat.v4.new_markdown_cell(
        '## Prepare the experimental subset\n\nApply the already audited journal and list rules to protected title matches. A paper triggering both is removed once. Save the exact ID decisions and matching vector rows in an isolated experiment folder.'), preparation]
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            compile(cell.source, NAME, 'exec')
    nbformat.validate(notebook)
    nbformat.write(notebook, ROOT / NAME)
    print('Created:', NAME)


if __name__ == '__main__':
    main()
