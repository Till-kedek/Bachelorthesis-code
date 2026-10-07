"""Independent paper workers with deterministic partitioning and a validated merge.

Each process owns one complete model and one GPU. All segments of a paper stay
with the same worker; no cross-GPU tensor communication is needed. This module
has no scheduler dependency and performs no work at import time.
"""

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import subprocess
import sys
from time import sleep, perf_counter

from .embedding import (EmbeddingConfig, RuntimeConfig, load_plans, validate_plans,
                        load_model, embed_corpus, validate_result, assemble_corpus)
from .storage import fingerprint, read_json, write_json


def partition_papers(plans, workers):
    """Greedily balance total input tokens, assigning longer papers first.

    Token count is a workload heuristic, not a prediction of inference time.
    Original manifest order breaks ties and is retained within every partition.
    Empty partitions are supported when there are fewer papers than workers.
    """
    validate_plans(plans)
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError('workers must be a positive integer')
    groups, loads = [[] for _ in range(workers)], [0] * workers
    costs = [sum(s['token_count'] for s in p['segments']) for p in plans]
    for index in sorted(range(len(plans)), key=lambda i: (-costs[i], i)):
        worker = min(range(workers), key=lambda w: (loads[w], w))
        groups[worker].append(index)
        loads[worker] += costs[index]
    result = {'schema_version': 1, 'method': 'longest_total_tokens_first',
              'corpus_id': fingerprint([p['id'] for p in plans]), 'workers': workers,
              'assignments': [[plans[i]['id'] for i in sorted(g)] for g in groups],
              'estimated_tokens': loads}
    result['id'] = fingerprint(result)
    return result


def prepare_workers(output_dir, workers):
    """Create a partition manifest once, before any workers are launched."""
    output_dir = Path(output_dir)
    partition = partition_papers(load_plans(output_dir / 'prepared'), workers)
    directory = output_dir / 'parallel' / partition['id']
    path = directory / 'partition.json'
    if path.exists() and read_json(path) != partition:
        raise ValueError('Stored partition differs from current inputs')
    write_json(path, partition)
    return partition


def _selection(output_dir, workers):
    output_dir = Path(output_dir)
    plans = load_plans(output_dir / 'prepared')
    partition = partition_papers(plans, workers)
    directory = output_dir / 'parallel' / partition['id']
    if read_json(directory / 'partition.json') != partition:
        raise ValueError('Partition mismatch; prepare workers again')
    return plans, partition, directory


def run_worker(output_dir, workers, worker_index, runtime):
    """Encode only this worker's assigned papers, resuming valid caches."""
    started = perf_counter()
    plans, partition, directory = _selection(output_dir, workers)
    if not 0 <= worker_index < workers:
        raise ValueError('worker_index is outside the partition range')
    by_id = {p['id']: p for p in plans}
    selected = [by_id[key] for key in partition['assignments'][worker_index]]
    print(f"Worker {worker_index}: {len(selected)} papers; "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}", flush=True)
    destination = directory / f'worker-{worker_index}'
    completion = destination / 'complete.json'
    completion.unlink(missing_ok=True)
    try:
        missing = False
        for plan in selected:
            path = destination / 'papers' / f"{plan['id']}.json"
            canonical = Path(output_dir) / 'embeddings' / 'papers' / path.name
            if not path.exists() and canonical.exists():
                result = read_json(canonical)
                validate_result(plan, result)
                write_json(path, result)
            if path.exists():
                validate_result(plan, read_json(path))
            else:
                missing = True
        if missing:
            if runtime.device != 'cuda':
                raise ValueError('Independent GPU workers require device=cuda')
            import torch
            if torch.cuda.device_count() != 1:
                raise RuntimeError('Each worker must see exactly one CUDA GPU; check GPU binding')
            model = load_model(EmbeddingConfig(**selected[0]['config']), runtime=runtime)
            embed_corpus(selected, model, destination)
        # Empty or fully cached workers do not load any model or initialize CUDA.
        for plan in selected:
            validate_result(plan, read_json(destination / 'papers' / f"{plan['id']}.json"))
        write_json(completion, {'partition_id': partition['id'], 'worker_index': worker_index,
                              'input_ids': [p['id'] for p in selected],
                              'status': 'embedded' if missing else ('cached' if selected else 'empty'),
                              'elapsed_seconds': perf_counter() - started})
        (destination / 'failure.json').unlink(missing_ok=True)
    except (Exception, KeyboardInterrupt) as error:
        write_json(destination / 'failure.json', {'error': f'{type(error).__name__}: {error}'})
        raise
    return completion


def merge_workers(output_dir, workers):
    """Validate every worker before publishing the matrix in original paper order."""
    output_dir = Path(output_dir)
    plans, partition, directory = _selection(output_dir, workers)
    by_id = {p['id']: p for p in plans}
    sources = {}
    completions = []
    for worker, input_ids in enumerate(partition['assignments']):
        folder = directory / f'worker-{worker}'
        expected = {'partition_id': partition['id'], 'worker_index': worker, 'input_ids': input_ids}
        completion = read_json(folder / 'complete.json')
        if any(completion.get(key) != value for key, value in expected.items()):
            raise ValueError(f'Worker {worker} has missing or mismatched completion metadata')
        completions.append(completion)
        for input_id in input_ids:
            path = folder / 'papers' / f'{input_id}.json'
            validate_result(by_id[input_id], read_json(path))
            sources[input_id] = path
    destination = output_dir / 'embeddings'
    # Nothing in the canonical output is changed until all worker results pass.
    for plan in plans:
        write_json(destination / 'papers' / f"{plan['id']}.json", read_json(sources[plan['id']]))
    assemble_corpus(plans, destination)
    write_json(destination / 'last_run.json', {
        'mode': 'parallel', 'partition_id': partition['id'], 'workers': completions})
    return destination


def _launch_commands(commands, envs, log_paths):
    """Wait for all processes; a failure or interruption prevents merge."""
    processes, handles = [], []
    try:
        for command, env, log in zip(commands, envs, log_paths):
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open('w', encoding='utf-8')
            handles.append(handle)
            processes.append(subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT))
        while True:
            codes = [p.poll() for p in processes]
            if any(code is not None and code != 0 for code in codes):
                raise RuntimeError('An embedding worker failed; inspect worker.log files. Completed papers are cached.')
            if all(code == 0 for code in codes):
                break
            sleep(0.2)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for handle in handles:
            handle.close()


def embed_parallel(output_dir, runtime=None, workers=4):
    """Launch one isolated process per visible local GPU; publish only on success."""
    runtime = runtime or RuntimeConfig()
    if runtime.device != 'cuda':
        raise ValueError('Multi-GPU execution requires device=cuda')
    import torch
    count = torch.cuda.device_count()
    if workers < 1 or workers > count:
        raise ValueError(f'Requested {workers} workers, but only {count} CUDA GPUs are visible')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    devices = [v.strip() for v in visible.split(',')] if visible is not None else [str(i) for i in range(count)]
    if len(devices) < workers or len(set(devices[:workers])) != workers:
        raise ValueError('CUDA_VISIBLE_DEVICES does not identify distinct GPUs')
    output_dir = Path(output_dir).resolve()
    partition = prepare_workers(output_dir, workers)
    base = output_dir / 'parallel' / partition['id']
    commands, envs, logs = [], [], []
    for worker in range(workers):
        command = [sys.executable, '-m', 'batill.parallel', 'worker', '--output-dir', str(output_dir),
                   '--workers', str(workers), '--worker-index', str(worker)]
        for key, value in asdict(runtime).items():
            command.extend(['--' + key.replace('_', '-'), value])
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices[worker])
        env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get('PYTHONPATH', '')
        commands.append(command)
        envs.append(env)
        logs.append(base / f'worker-{worker}' / 'worker.log')
    _launch_commands(commands, envs, logs)
    return merge_workers(output_dir, workers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'worker', 'merge'])
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--worker-index', type=int)
    for key, value in asdict(RuntimeConfig()).items():
        parser.add_argument('--' + key.replace('_', '-'), default=value)
    args = parser.parse_args()
    if args.stage == 'prepare':
        print(prepare_workers(args.output_dir, args.workers))
    elif args.stage == 'merge':
        print(merge_workers(args.output_dir, args.workers))
    else:
        if args.worker_index is None:
            parser.error('worker requires --worker-index')
        runtime = RuntimeConfig(**{key: getattr(args, key) for key in asdict(RuntimeConfig())})
        run_worker(args.output_dir, args.workers, args.worker_index, runtime)


if __name__ == '__main__':
    main()
