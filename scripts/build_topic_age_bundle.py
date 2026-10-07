"""Bundle the age notebook, frozen CSV inputs and reproducible results for handoff."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = json.loads((ROOT / 'configs/topic-age-inputs.json').read_text())
    reference = json.loads((ROOT / 'configs/topic-age-reference.json').read_text())
    specs = [item for source in config['sources'].values() for item in (source['memberships'], source['profiles'])]
    specs += [config['reviewed_nodes'], config['pdf_to_work']]
    expected = {spec['path']: spec['sha256'] for spec in specs}
    expected.update(reference['files'])
    expected.update(reference['code_and_config'])
    for relative, checksum in expected.items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != checksum:
            raise ValueError(f'Changed input or reference: {relative}; run the checker before packaging.')
    paths = set(expected) | {
        'Topic analysis Ages.ipynb', 'src/batill/__init__.py',
        'scripts/reproduce_topic_ages.py', 'scripts/build_topic_age_bundle.py',
        'configs/topic-age-reference.json', 'reports/topic_age_analysis/run_manifest.json',
    }
    paths.update(str(p.relative_to(ROOT)) for p in (ROOT / 'reports/topic_age_analysis/figures').glob('*') if p.is_file())
    manifest = {'schema_version': 1, 'purpose': 'Reproduce paper ages and temporal shares with all topic memberships/names fixed',
                'files': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sorted(paths)}}
    readme = '''Topic publication ages — supervisor handoff

Open Topic analysis Ages.ipynb to read the executed analysis and its methods.
Inputs and all five source partitions/names are fixed. There are no API calls,
new embeddings, new clusters or newly generated names.

Inside this folder, use Python 3.12 (tested: 3.12.7 on macOS Apple Silicon):
    python3.12 -m venv .venv
    .venv/bin/python -m pip install -r configs/topic-age-requirements.lock
    .venv/bin/python scripts/reproduce_topic_ages.py --check

Installation needs internet access; analysis runs offline. Every CSV output is
compared byte for byte with the supplied reference after a fresh execution.
The checker stops on changed inputs or numerical results. Different operating
systems are not certified; plot rendering is not compared byte for byte.

For interactive use, register the environment and choose it in your editor:
    .venv/bin/python -m ipykernel install --user --name batill-topic-ages

The notebook explains the fixed 2026 age reference, source dating differences,
PDF/canonical-work units, citation isolates, denominators and interpretation limits.
reports/topic_age_analysis contains the complete tables and PNG/PDF figures.
BUNDLE_MANIFEST.json records SHA-256 checksums of the packaged files.
This compact bundle reproduces the age analysis; the separate full topic-analysis
bundle reproduces the upstream topic interpretations and naming evidence.
'''
    destination = ROOT / 'outputs/topic-age-reproduction.zip'
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for relative in sorted(paths):
            bundle.write(ROOT / relative, f'topic-age-reproduction/{relative}')
        bundle.writestr('topic-age-reproduction/BUNDLE_MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
        bundle.writestr('topic-age-reproduction/README.txt', readme)
    print(f'{destination}: {len(paths)} files, {destination.stat().st_size / 1e6:.2f} MB')


if __name__ == '__main__':
    main()
