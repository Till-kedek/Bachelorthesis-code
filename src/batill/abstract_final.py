"""Pinned inputs for the approved broad-abstract funnel; independent of PDF inputs."""
from pathlib import Path

import pandas as pd

from .storage import file_hash, read_json, write_json

FINAL_CONFIG = 'configs/abstract_final_input.json'


def resolve_final_inputs(root):
    root = Path(root).resolve()
    config = read_json(root / FINAL_CONFIG)
    if config.get('schema_version') != 1:
        raise ValueError('Unsupported final abstract configuration')
    for relative, digest in config['input_hashes'].items():
        path = root / relative
        if not path.is_file() or file_hash(path) != digest:
            raise ValueError(f'Final abstract input changed or missing: {relative}')
    source = root / config['source_csv']
    papers = pd.read_csv(source, dtype=str, keep_default_na=False)
    audit = pd.read_csv(root / config['screening_decisions'], keep_default_na=False)
    retained = audit.loc[audit.retained, 'paper_id']
    if (not papers.paper_id.is_unique or len(papers) != config['papers']
            or set(papers.paper_id) != set(retained)):
        raise ValueError('Final CSV differs from the approved screening decisions')
    return root / config['run_dir'], source


def load_final_context(root, *, build_missing=True):
    from .abstract_topic_explorer import load_context
    context = load_context(root, input_paths=resolve_final_inputs(root), build_missing=build_missing)
    context['export_directory'] = 'outputs/topic_analysis_abstracts_final'
    context['input_hashes'][FINAL_CONFIG] = file_hash(Path(root) / FINAL_CONFIG)
    return context


def export_final_selected(root, context):
    """Save both selected partitions without inheriting any old memberships."""
    root = Path(root)
    out = root / 'outputs/abstract_final' / context['input_id'][:16]
    out.mkdir(parents=True, exist_ok=True)
    selections = ['kmeans_k5', 'leiden_r0.45']
    frames = []
    for solution in selections:
        result = context['solutions'][solution]
        frame = context['papers'].drop(columns='text').copy()
        frame['solution'] = solution
        frame['cluster'] = result['labels']
        frame['cosine_silhouette'] = result['silhouettes']
        frames.append(frame)
    assignments = pd.concat(frames, ignore_index=True)
    assignments.to_csv(out / 'selected_assignments.csv', index=False)
    context['papers'].drop(columns='text').to_csv(out / 'papers.csv', index=False)
    write_json(out / 'manifest.json', {
        'input_id': context['input_id'], 'papers': len(context['papers']),
        'input_config': FINAL_CONFIG, 'input_config_sha256': file_hash(root / FINAL_CONFIG),
        'catalog': str(context['partition_file'].relative_to(root)),
        'catalog_sha256': file_hash(context['partition_file']),
        'selected_solutions': selections, 'embeddings_recomputed': False,
        'artifacts_sha256': {name: file_hash(out / name)
                            for name in ('selected_assignments.csv', 'papers.csv')},
    })
    return out
