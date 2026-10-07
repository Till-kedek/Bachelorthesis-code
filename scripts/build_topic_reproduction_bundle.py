"""Package the exact local inputs needed to reproduce all three topic analyses."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[1]
INPUTS = [
    'README.md', 'pyproject.toml', 'src/batill',
    'Topic analysis PDFs.ipynb', 'Topic analysis Abstracts.ipynb', 'Topic analysis Citations.ipynb',
    'Clustering PDFs.ipynb', 'Clustering Abstracts.ipynb',
    'scripts/reproduce_topic_analyses.py', 'scripts/build_topic_reproduction_bundle.py',
    'configs/analysis_exclusions.json', 'configs/pdf_duplicates.json', 'configs/pdf_topic_labels.csv',
    'configs/abstract_topic_labels.csv', 'configs/citation_topic_labels.csv',
    'configs/topic-analysis-requirements.lock', 'configs/topic-analysis-reference.json',
    'outputs/corpus_sections/prepared', 'outputs/corpus_sections/embeddings',
    'data/abstracts/abstracts_clean.csv',
    'outputs/abstract_embeddings/bge_m3/20260930T174845_508054Z_full_3c498a9d',
    'outputs/pdf_text_clusters/d4437a4c2460707a/five_clusters',
    'outputs/abstract_text_clusters/fab1ccb1023ccc11/six_clusters',
    'outputs/citation_topic_partitions/9ad47e071ee12c2f/resolution_1',
    'outputs/citation_graph_openalex/9ad47e071ee12c2f/reviewed',
    # Historical evidence authenticating the adopted citation snapshot.
    'outputs/citation_clustering/9ad47e071ee12c2f/with_manual/all_seed_memberships.csv',
    'outputs/citation_clustering/9ad47e071ee12c2f/with_manual/resolution_1/work_membership.csv',
    'outputs/citation_clustering/9ad47e071ee12c2f/with_manual/resolution_1/leiden_1/run_manifest.json',
    'outputs/citation_clustering/9ad47e071ee12c2f/with_manual/resolution_1/leiden_1/umap_coordinates_and_labels.csv',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/topic-analysis-reproduction.tar.gz')
    args = parser.parse_args()
    paths = set()
    for relative in INPUTS:
        path = ROOT / relative
        if not path.exists():
            raise FileNotFoundError(f'Required reproduction input: {relative}')
        paths.update([path] if path.is_file() else
                     [p for p in path.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'])
    reference = json.loads((ROOT / 'configs/topic-analysis-reference.json').read_text())
    paths.update(ROOT / p for p in reference['files'])
    paths.update((ROOT / p).parent / 'run_manifest.json' for p in reference['files'])
    manifest = {'schema_version': 1, 'purpose': 'Saved-partition topic reproduction; no new clustering or model inference',
                'files': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(paths)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import io
    with tarfile.open(args.output, 'w:gz') as archive:
        for path in sorted(paths):
            archive.add(path, arcname=str(Path('topic-analysis-reproduction') / path.relative_to(ROOT)), recursive=False)
        data = (json.dumps(manifest, indent=2) + '\n').encode()
        entry = tarfile.TarInfo('topic-analysis-reproduction/BUNDLE_MANIFEST.json')
        entry.size = len(data)
        archive.addfile(entry, io.BytesIO(data))
    print(f'{args.output}: {len(paths)} files, {args.output.stat().st_size / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
