"""Execute the frozen-topic age notebook offline and verify all reference tables."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = 'Topic analysis Ages.ipynb'
REFERENCE = 'configs/topic-age-reference.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def notebook_source_hash():
    notebook = json.loads((ROOT / NOTEBOOK).read_text())
    source = [(cell['cell_type'], ''.join(cell['source'])) for cell in notebook['cells']]
    return hashlib.sha256(json.dumps(source, ensure_ascii=False).encode()).hexdigest()


def run_notebook():
    import socket
    from contextlib import ExitStack
    from unittest.mock import patch
    import nbformat
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output

    def forbidden(*args, **kwargs):
        raise AssertionError('The topic-age notebook must run offline.')

    os.chdir(ROOT)
    notebook = nbformat.read(ROOT / NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    shell = InteractiveShell.instance()
    from matplotlib_inline.backend_inline import configure_inline_support
    configure_inline_support(shell, 'module://matplotlib_inline.backend_inline')
    with ExitStack() as stack:
        for obj, attr in [(socket.socket, 'connect'), (socket.socket, 'connect_ex'), (socket, 'create_connection')]:
            stack.enter_context(patch.object(obj, attr, forbidden))
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type != 'code':
                continue
            print(f'{NOTEBOOK}: cell {i}', flush=True)
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
    nbformat.write(notebook, ROOT / NOTEBOOK)
    manifest = json.loads((ROOT / 'reports/topic_age_analysis/run_manifest.json').read_text())
    if manifest['clustering_performed'] or manifest['network_required']:
        raise AssertionError('Age analysis must use the frozen inputs offline.')
    for name, expected in manifest['artifacts_sha256'].items():
        if sha(ROOT / 'reports/topic_age_analysis' / name) != expected:
            raise AssertionError(f'Generated artifact checksum mismatch: {name}')


def validate_files(reference, section):
    for relative, expected in reference[section].items():
        path = ROOT / relative
        if not path.is_file() or sha(path) != expected:
            raise AssertionError(f'Reference {section} mismatch: {relative}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='Verify frozen code/config and exact output table hashes (default)')
    mode.add_argument('--record-reference', action='store_true', help='Explicitly record a new table/code reference after a reviewed change')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        run_notebook()
        return
    reference = None if args.record_reference else json.loads((ROOT / REFERENCE).read_text())
    if reference:
        validate_files(reference, 'code_and_config')
        if notebook_source_hash() != reference['notebook_source_sha256']:
            raise AssertionError('Notebook source changed; restore the shared notebook to reproduce the reference.')
    env = dict(os.environ, PYTHONHASHSEED='0', MPLCONFIGDIR=str(ROOT / '.cache/topic-age-matplotlib'),
               MPLBACKEND='module://matplotlib_inline.backend_inline', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', IPYTHONDIR=str(ROOT / '.cache/topic-age-ipython'))
    for variable in ('MPLCONFIGDIR', 'IPYTHONDIR'):
        Path(env[variable]).mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker'], cwd=ROOT, env=env, check=True)
    if args.record_reference:
        paths = sorted((ROOT / 'reports/topic_age_analysis').glob('*.csv'))
        reference = {
            'schema_version': 1,
            'comparison': 'Exact CSV bytes after offline notebook execution; displays, figures and environment manifest excluded.',
            'notebook_source_sha256': notebook_source_hash(),
            'code_and_config': {p: sha(ROOT / p) for p in [
                'src/batill/topic_age_analysis.py', 'configs/topic-age-inputs.json',
                'scripts/create_topic_age_notebook.py', 'scripts/reproduce_topic_ages.py',
                'configs/topic-age-requirements.lock']},
            'files': {str(p.relative_to(ROOT)): sha(p) for p in paths},
        }
        (ROOT / REFERENCE).write_text(json.dumps(reference, indent=2) + '\n')
        print('Recorded a new reviewed reference.')
    validate_files(reference, 'files')
    print(f"PASS: all {len(reference['files'])} reference tables match byte for byte.")


if __name__ == '__main__':
    main()
