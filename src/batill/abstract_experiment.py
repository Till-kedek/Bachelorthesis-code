"""Load pinned saved K-means memberships from a broad-abstract experiment."""
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples
from sklearn.metrics.pairwise import cosine_distances

from .abstract_topic_analysis import abstract_partition_input_id, load_abstract_topic_corpus
from .storage import file_hash, read_json


def load_saved_experiment(root, experiment, solutions):
    """Authenticate the corpus and exact saved partitions; never fit or fall back."""
    root = Path(root).resolve()
    directory = root / experiment['directory']
    manifest_path = directory / 'clustering_manifest.json'
    if file_hash(manifest_path) != experiment['clustering_manifest_sha256']:
        raise ValueError('Pinned abstract experiment manifest changed; review the selected experiment')
    manifest = read_json(manifest_path)
    input_path = directory / 'input_manifest.json'
    if file_hash(input_path) != manifest['experiment_input_manifest_sha256']:
        raise ValueError('Abstract experiment input manifest changed')
    inputs = read_json(input_path)
    hashes = {}
    for recorded in (inputs['artifacts_sha256'], manifest['artifact_sha256']):
        for name, digest in recorded.items():
            if file_hash(directory / name) != digest:
                raise ValueError(f'Abstract experiment artifact changed: {name}')
            hashes[str((directory / name).relative_to(root))] = digest
    hashes[str(manifest_path.relative_to(root))] = file_hash(manifest_path)
    hashes['loader_code_sha256'] = file_hash(Path(__file__))
    papers, vectors, _ = load_abstract_topic_corpus(directory / 'embedding_run', directory / 'abstracts_clean.csv')
    if (len(papers) != manifest['papers'] or len(papers) != experiment['papers']
            or abstract_partition_input_id(papers.paper_id, vectors) != manifest['input_id']):
        raise ValueError('Saved experiment memberships do not match the pinned corpus')
    decisions = pd.read_csv(directory / 'experiment_decisions.csv')
    if set(papers.paper_id) != set(decisions.loc[decisions.experiment_retained, 'paper_id']):
        raise ValueError('Experiment corpus differs from its retained-paper decisions')
    summary = pd.read_csv(directory / 'partition_summary.csv').set_index('solution')
    if not summary.index.is_unique:
        raise ValueError('Duplicate experiment solution summaries')
    assignments = pd.read_csv(directory / 'all_assignments.csv')
    distance = cosine_distances(vectors)
    distance = np.maximum((distance + distance.T) / 2, 0)
    np.fill_diagonal(distance, 0)
    selected = {}
    with np.load(directory / 'partitions.npz', allow_pickle=False) as saved:
        if saved['paper_ids'].tolist() != papers.paper_id.tolist():
            raise ValueError('Saved experiment paper order changed')
        for solution in solutions:
            match = re.fullmatch(r'kmeans_k(\d+)', solution)
            if not match or solution not in saved or solution not in summary.index:
                raise ValueError(f'Missing or unsupported saved experiment solution: {solution}')
            k = int(match[1])
            labels, scores = saved[solution], saved[solution + '__silhouettes']
            if (labels.shape != (len(papers),) or labels.dtype.kind not in 'iu'
                    or set(labels) != set(range(k)) or scores.shape != labels.shape
                    or not np.isfinite(scores).all()):
                raise ValueError(f'Invalid saved experiment membership: {solution}')
            rows = assignments.loc[assignments.solution.eq(solution)]
            if rows.paper_id.tolist() != papers.paper_id.tolist() or not np.array_equal(rows.cluster, labels):
                raise ValueError('Experiment CSV and vector-order memberships disagree')
            if not np.allclose(scores, silhouette_samples(distance, labels, metric='precomputed'), atol=1e-7):
                raise ValueError('Saved experiment silhouettes do not match the corpus and memberships')
            spec = summary.loc[solution]
            if int(spec.clusters) != k or spec.method != 'K-means':
                raise ValueError('Experiment partition summary differs from saved labels')
            selected[solution] = {'method': 'K-means', 'clusters': k, 'seed': int(spec.seed),
                                  'labels': labels.copy(), 'silhouettes': scores.copy()}
    return {'papers': papers, 'embeddings': vectors, 'solutions': selected,
            'evidence': directory, 'partition_file': directory / 'partitions.npz',
            'input_hashes': hashes, 'input_id': manifest['input_id']}
