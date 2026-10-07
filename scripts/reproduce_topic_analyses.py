"""Execute saved-partition topic notebooks offline and verify reference artifacts.

No Jupyter server is needed: each notebook runs in a fresh Python subprocess
through IPython, with captured notebook outputs and no inherited cell variables.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    'pdfs': 'Topic analysis PDFs.ipynb',
    'abstracts': 'Topic analysis Abstracts.ipynb',
    'citations': 'Topic analysis Citations.ipynb',
}
REFERENCE = ROOT / 'configs/topic-analysis-reference.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_notebook(name):
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / 'src'))
    import nbformat
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    from sklearn.cluster import KMeans
    from unittest.mock import patch
    from contextlib import ExitStack
    import batill.topic_analysis as topics
    import batill.graph_clustering as graphs
    import batill.citation_clustering as citations
    import batill.embedding as embedding
    import batill.extraction as extraction
    import batill.pipeline as pipeline
    import socket
    import leidenalg

    def forbidden(*args, **kwargs):
        raise AssertionError('Topic reproduction forbids clustering, inference, extraction and network calls')

    targets = [(KMeans, 'fit'), (topics, 'fit_partitions'), (topics, 'leiden_partition'),
               (graphs, 'leiden_partition'), (graphs, 'leiden_sweep'),
               (citations, 'citation_leiden_sweep'), (leidenalg, 'find_partition'),
               (embedding, 'load_model'), (embedding, 'load_tokenizer'),
               (pipeline, 'load_model'), (pipeline, 'load_tokenizer'),
               (extraction, 'extract_papers'), (pipeline, 'extract_papers'),
               (socket.socket, 'connect'), (socket.socket, 'connect_ex'),
               (socket, 'create_connection')]
    path = ROOT / NOTEBOOKS[name]
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    shell = InteractiveShell.instance()
    # This command reproduces frozen reference tables, not the active scope run.
    # Preserve the original abstracts even after Abstract Cleaning activates a subset.
    shell.user_ns['USE_ORIGINAL_ABSTRACTS'] = True
    with ExitStack() as stack:
        for obj, attr in targets:
            stack.enter_context(patch.object(obj, attr, forbidden))
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type != 'code':
                continue
            print(f'{path.name}: cell {i}', flush=True)
            with capture_output() as captured:
                result = shell.run_cell(cell.source, store_history=True)
            if not result.success:
                print(captured.stdout, captured.stderr, file=sys.stderr)
                raise RuntimeError(result.error_before_exec or result.error_in_exec)
            cell.execution_count = shell.execution_count - 1
            cell.outputs = [nbformat.v4.new_output('stream', name=stream, text=getattr(captured, stream))
                            for stream in ('stdout', 'stderr') if getattr(captured, stream)]
            cell.outputs += [nbformat.v4.new_output('display_data', data=out.data, metadata=out.metadata)
                             for out in captured.outputs]
    if not shell.user_ns['named_profiles'].proposal_matched.all():
        raise AssertionError('Names are missing or do not match the saved memberships/texts')
    nbformat.write(notebook, path)
    report = shell.user_ns['RESULTS']
    manifest = json.loads((report / 'run_manifest.json').read_text())
    if manifest['clustering_performed'] or not manifest['all_names_match_memberships']:
        raise AssertionError('Invalid interpretation manifest')
    for filename, expected in manifest['artifacts_sha256'].items():
        if sha(report / filename) != expected:
            raise AssertionError(f'Invalid generated artifact checksum: {filename}')
    print(f'{name}: completed; {len(shell.user_ns["named_profiles"])} matched names; {report}', flush=True)
    return report


def validate_reference(reference):
    for relative, expected in reference['files'].items():
        path = ROOT / relative
        if not path.is_file() or sha(path) != expected:
            raise AssertionError(f'Reference artifact changed or missing: {relative}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', choices=['all', *NOTEBOOKS], default='all')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='Require exact reference artifact checksums before and after execution')
    mode.add_argument('--record-reference', action='store_true', help='Explicitly replace the reference after reviewed changes (all analyses only)')
    parser.add_argument('--worker', choices=list(NOTEBOOKS), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        run_notebook(args.worker)
        return
    if args.record_reference and args.analysis != 'all':
        parser.error('--record-reference requires all three analyses')
    reference = json.loads(REFERENCE.read_text()) if args.check else None
    if reference:
        validate_reference(reference)
    names = list(NOTEBOOKS) if args.analysis == 'all' else [args.analysis]
    for name in names:
        env = dict(os.environ, PYTHONHASHSEED='0', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', MPLCONFIGDIR=str(ROOT / '.cache/topic-matplotlib'))
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', name],
                       cwd=ROOT, env=env, check=True)
    if args.record_reference:
        files = {}
        for name in names:
            # One current reference report per notebook; derive it from the source snapshot.
            if name == 'abstracts':
                base, snapshot = 'topic_analysis_abstracts/fab1ccb1023ccc11', 'abstract_text_clusters/fab1ccb1023ccc11/six_clusters'
            elif name == 'pdfs':
                base, snapshot = 'topic_analysis_pdfs/d4437a4c2460707a', 'pdf_text_clusters/d4437a4c2460707a/five_clusters'
            else:
                base, snapshot = 'topic_analysis_citations/9ad47e071ee12c2f', 'citation_topic_partitions/9ad47e071ee12c2f/resolution_1'
            source = json.loads((ROOT / 'outputs' / snapshot / 'partition_manifest.json').read_text())
            report = ROOT / 'reports' / base / source['assignments_sha256'][:16]
            manifest = json.loads((report / 'run_manifest.json').read_text())
            for filename in manifest['artifacts_sha256']:
                path = report / filename
                files[str(path.relative_to(ROOT))] = sha(path)
        REFERENCE.write_text(json.dumps({'schema_version': 1,
            'comparison': 'Exact SHA-256 of deterministic CSV, JSON and review-card artifacts. Run timestamps, absolute paths in run manifests and notebook display outputs are excluded.',
            'files': files}, indent=2) + '\n')
        print(f'Recorded {len(files)} reference artifacts.')
    if reference:
        validate_reference(reference)
        print(f'PASS: all {len(reference["files"])} reference artifacts match byte for byte.')


if __name__ == '__main__':
    main()
