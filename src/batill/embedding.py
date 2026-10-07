"""Content-addressed paper embeddings with portable, validated output files."""

from dataclasses import asdict, dataclass
from contextlib import nullcontext
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
import platform
import re
import warnings

import numpy as np

from .documents import embedding_blocks
from .storage import fingerprint, read_json, write_json, write_npz, file_hash


@dataclass(frozen=True)
class EmbeddingConfig:
    """Semantic choices: these determine the identity of prepared inputs."""
    model: str = "Qwen/Qwen3-Embedding-4B"
    revision: str = "5cf2132abc99cad020ac570b19d031efec650f2b"
    max_tokens: int = 32768
    embedding_dimension: int = 2560
    math_policy: str = "omit"
    instruction: str = "Represent this research paper for clustering by its economic topic, research question, and mechanisms."
    chunking: str = "adaptive"


@dataclass(frozen=True)
class RuntimeConfig:
    """Execution choices, recorded in provenance rather than content identity."""
    device: str = "cuda"
    dtype: str = "float16"
    attention: str = "sdpa"
    sdpa_backend: str = "auto"


def embedding_config_dict(config):
    """Keep legacy adaptive identities stable; record the new section strategy."""
    values = asdict(config)
    if values['chunking'] == 'adaptive':
        values.pop('chunking')
    return values


def load_tokenizer(config, output_dir):
    """Load a tokenizer at an immutable revision, without loading model weights."""
    from transformers import AutoTokenizer
    if not re.fullmatch(r"[0-9a-f]{40}", config.revision):
        raise ValueError("Use an immutable 40-character model revision in the configuration")
    write_json(Path(output_dir) / 'model.json', embedding_config_dict(config))
    return AutoTokenizer.from_pretrained(config.model, revision=config.revision), config.revision


def prepare_inputs(document, tokenizer, config, *, _blocks=None, _group=None):
    """Prepare adaptive whole-paper inputs or consistently grouped section inputs.

    Every included character has a source block and character range. Token counts include
    the repeated instruction/title and special tokens. No text is silently truncated.
    In sections mode, each group is independently packed up to the token limit.
    """
    if config.chunking not in {'adaptive', 'sections'}:
        raise ValueError('chunking must be adaptive or sections')
    if config.chunking == 'sections' and _blocks is None:
        from .sections import group_sections
        blocks = embedding_blocks(document, config.math_policy)
        if not blocks:
            raise ValueError(f"Document contains no usable blocks: {document['source_filename']}")
        groups, flags = group_sections(blocks, document['title'])
        segments = []
        for index, group in enumerate(groups):
            segments.extend(prepare_inputs(document, tokenizer, config,
                _blocks=group['blocks'], _group={'section_group': index,
                    'section_name': group['name'], 'section_review_flags': flags}))
        return segments
    if not 128 <= config.max_tokens <= 32768:
        raise ValueError("max_tokens must be between 128 and 32768 for Qwen3 embeddings")
    prefix = f"Instruct: {config.instruction}\nQuery: Title: {document['title']}\n\n"

    def count(text):
        return len(tokenizer.encode(text, add_special_tokens=True))

    if count(prefix) >= config.max_tokens - 16:
        raise ValueError("Title/instruction leaves insufficient space for paper text")
    blocks = embedding_blocks(document, config.math_policy) if _blocks is None else _blocks
    if not blocks:
        raise ValueError(f"Document contains no usable blocks: {document['source_filename']} "
                         f"(PDF ID: {document['pdf_sha256']})")

    def segment(parts):
        text = prefix + "\n\n".join(part["text"] for part in parts)
        return {**(_group or {}), "text": text, "token_count": count(text),
                "content_tokens": max(1, count("\n\n".join(p["text"] for p in parts))),
                "sources": [{k: v for k, v in part.items() if k != "text"} for part in parts]}

    all_parts = [{"block_id": b["id"], "page": b["page"], "section": b["section"],
                  "start": 0, "end": len(b["text"]), "text": b["text"]} for b in blocks]
    whole = segment(all_parts)
    if whole["token_count"] <= config.max_tokens:
        return [whole]

    output, pending = [], []
    for part in all_parts:
        if config.chunking == 'adaptive' and pending and pending[-1]["section"] != part["section"]:
            output.append(segment(pending))
            pending = []
        if pending and segment(pending + [part])["token_count"] > config.max_tokens:
            output.append(segment(pending))
            pending = []
        if segment([part])["token_count"] <= config.max_tokens:
            pending.append(part)
            continue
        # Exceptional oversized block: split by character offsets, rechecking the actual tokenizer.
        start = 0
        while start < len(part["text"]):
            remaining = part["text"][start:]
            length = len(remaining)
            while count(prefix + remaining[:length]) > config.max_tokens:
                length //= 2
                if length == 0:
                    raise ValueError("Cannot fit even one character in the token budget")
            if length < len(remaining):
                boundaries = list(re.finditer(r"\s+", remaining[:length]))
                if boundaries and boundaries[-1].end() > length // 2:
                    length = boundaries[-1].end()
            piece = {**part, "text": remaining[:length], "start": start, "end": start + length}
            output.append(segment([piece]))
            start += length
    if pending:
        output.append(segment(pending))
    assert all(s["token_count"] <= config.max_tokens for s in output)
    return output


def prepare_corpus(documents, tokenizer, config, revision, output_dir, *, skip_empty=False):
    """Save exact inputs; optionally exclude empty papers with an auditable report.

    Other preparation errors still abort. Empty papers never receive title-only vectors.
    """
    output_dir = Path(output_dir)
    documents = list(documents)
    usable, excluded = [], []
    for doc in documents:
        if embedding_blocks(doc, config.math_policy):
            usable.append(doc)
        else:
            excluded.append({
                'pdf_sha256': doc['pdf_sha256'], 'source_filename': doc['source_filename'],
                'title': doc['title'], 'reason': 'no_usable_blocks',
                'quality_flags': doc.get('quality_flags', []),
            })
    write_json(output_dir / 'preparation_report.json', {
        'selected_count': len(documents), 'usable_count': len(usable),
        'skip_empty': skip_empty, 'excluded': excluded,
    })
    if excluded:
        names = '\n'.join(f"- {d['source_filename']} ({d['pdf_sha256']})" for d in excluded)
        if not skip_empty:
            raise ValueError(f"{len(excluded)} document(s) contain no usable blocks:\n{names}\n"
                             "Inspect preparation_report.json. Recover their text or explicitly "
                             "set skip_empty=True to exclude them.")
        warnings.warn(f"Excluding {len(excluded)} empty document(s):\n{names}\n"
                      f"Report: {output_dir / 'preparation_report.json'}", UserWarning, stacklevel=2)
    if not usable:
        raise ValueError('No usable documents remain; inspect preparation_report.json')
    plans = []
    for doc in usable:
        segments = prepare_inputs(doc, tokenizer, config)
        plan = {"pipeline_version": 3, "source_offsets": "character offsets in formula-filtered block text",
                "pdf_sha256": doc["pdf_sha256"], "Key": doc.get("Key"),
                "title": doc["title"], "source_filename": doc["source_filename"],
                "config": {**embedding_config_dict(config), "revision": revision}, "segments": segments}
        plan["id"] = fingerprint(plan)
        write_json(output_dir / "inputs" / f"{plan['id']}.json", plan)
        readable = output_dir / "readable"
        readable.mkdir(parents=True, exist_ok=True)
        for index, segment in enumerate(segments):
            (readable / f"{doc['pdf_sha256']}.input-{index + 1}.txt").write_text(segment["text"], encoding="utf-8")
        plans.append(plan)
    validate_plans(plans)
    write_json(output_dir / 'manifest.json', {
        'schema_version': 1, 'corpus_id': fingerprint([p['id'] for p in plans]),
        'inputs': [f"inputs/{p['id']}.json" for p in plans], 'excluded': excluded})
    return plans


def validate_plans(plans):
    if not plans:
        raise ValueError("No prepared documents selected")
    if len({p['pdf_sha256'] for p in plans}) != len(plans):
        raise ValueError('Duplicate PDF IDs in prepared corpus')
    if len({fingerprint(p['config']) for p in plans}) != 1:
        raise ValueError('Do not combine embedding configurations in one corpus')
    for plan in plans:
        if fingerprint({k: v for k, v in plan.items() if k != 'id'}) != plan['id']:
            raise ValueError('Prepared input checksum mismatch')
        if not plan['segments'] or any(s['token_count'] > plan['config']['max_tokens'] for s in plan['segments']):
            raise ValueError('Invalid segment lengths')


def load_plans(prepared_dir):
    prepared_dir = Path(prepared_dir)
    manifest = read_json(prepared_dir / 'manifest.json')
    plans = [read_json(prepared_dir / p) for p in manifest['inputs']]
    validate_plans(plans)
    if fingerprint([p['id'] for p in plans]) != manifest['corpus_id']:
        raise ValueError('Prepared corpus checksum mismatch')
    return plans


def load_model(config, revision=None, runtime=None):
    """Explicit expensive step; nothing is loaded when this module is imported."""
    from sentence_transformers import SentenceTransformer
    import torch
    runtime = runtime or RuntimeConfig()
    revision = revision or config.revision
    if runtime.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Choose an explicit supported device; no automatic fallback.')
    if runtime.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable on this machine')
    if runtime.dtype not in {'float32', 'float16', 'bfloat16'}:
        raise ValueError('Unsupported model dtype')
    if runtime.attention == 'batill_sdpa_repeat_kv':
        from .attention import register_repeated_kv_attention
        register_repeated_kv_attention()
    model = SentenceTransformer(config.model, revision=revision, device=runtime.device,
        model_kwargs={'torch_dtype': getattr(torch, runtime.dtype), 'attn_implementation': runtime.attention},
        config_kwargs={'use_cache': False}, tokenizer_kwargs={'padding_side': 'left'})
    if config.max_tokens > model.max_seq_length:
        raise ValueError(f'Requested context exceeds model limit {model.max_seq_length}')
    model.max_seq_length = config.max_tokens
    model._batill_embedding_config = {**embedding_config_dict(config), 'revision': revision}
    model._batill_runtime = asdict(runtime)
    return model


def validate_result(plan, result):
    """Reject stale, mismatched or damaged per-paper caches before reuse/import."""
    if result['input_id'] != plan['id'] or result['config'] != plan['config']:
        raise ValueError('Embedding does not match the prepared input/model')
    segments = np.asarray(result['segment_vectors'], dtype=np.float32)
    vector = np.asarray(result['paper_vector'], dtype=np.float32)
    expected = (len(plan['segments']), plan['config']['embedding_dimension'])
    if segments.shape != expected or vector.shape != (expected[1],):
        raise ValueError('Unexpected embedding dimensions')
    if not np.isfinite(segments).all() or not np.isfinite(vector).all():
        raise ValueError('Non-finite embedding values')
    weights = np.asarray([s['content_tokens'] for s in plan['segments']], dtype=float)
    if result['weights'] != weights.tolist() or np.any(weights <= 0):
        raise ValueError('Segment weights do not match prepared inputs')
    aggregate = np.average(segments, axis=0, weights=weights)
    norm = np.linalg.norm(aggregate)
    if norm == 0 or not np.allclose(vector, aggregate / norm, atol=1e-6):
        raise ValueError('Paper vector differs from the declared aggregation')
    if not np.isclose(np.linalg.norm(vector), 1, atol=1e-5):
        raise ValueError('Paper vector is not normalized')
    return vector


def assemble_corpus(plans, output_dir):
    """Publish the canonical matrix in manifest order, without loading a model."""
    validate_plans(plans)
    output_dir = Path(output_dir)
    vectors, rows = [], []
    for plan in plans:
        relative = f"papers/{plan['id']}.json"
        result = read_json(output_dir / relative)
        vectors.append(validate_result(plan, result))
        rows.append({'pdf_sha256': plan['pdf_sha256'], 'title': plan['title'], 'Key': plan['Key'],
                     'source_filename': plan['source_filename'], 'input_id': plan['id'],
                     'embedding_path': relative, 'execution': result['execution']})
    write_npz(output_dir / 'embeddings.npz', embeddings=np.asarray(vectors, dtype=np.float32),
              pdf_sha256=np.array([p['pdf_sha256'] for p in plans]),
              keys=np.array([p['Key'] or '' for p in plans]))
    write_json(output_dir / 'manifest.json', {
        'schema_version': 1, 'corpus_id': fingerprint([p['id'] for p in plans]),
        'matrix_sha256': file_hash(output_dir / 'embeddings.npz'),
        'config': plans[0]['config'], 'rows': rows,
        'aggregation': 'L2-normalized token-weighted mean of segment vectors'})
    return output_dir


def embed_corpus(plans, model, output_dir):
    """Save each completed paper; reuse validated caches and publish one matrix."""
    validate_plans(plans)
    if getattr(model, '_batill_embedding_config', None) != plans[0]['config']:
        raise ValueError('Loaded model does not match prepared inputs')
    output_dir = Path(output_dir)
    runtime = getattr(model, '_batill_runtime', asdict(RuntimeConfig()))
    execution = {'runtime': runtime, 'host': platform.node(), 'platform': platform.platform(),
                 'packages': {p: version(p) for p in ('sentence-transformers', 'transformers', 'torch')},
                 'generation_cache': False}
    attempts = []
    for index, plan in enumerate(plans, 1):
        started = perf_counter()
        path = output_dir / 'papers' / f"{plan['id']}.json"
        row = {'input_id': plan['id'], 'title': plan['title']}
        print(f"[{index}/{len(plans)}] {plan['title']}", flush=True)
        try:
            if path.exists():
                validate_result(plan, read_json(path))
                row['status'] = 'cached'
            else:
                texts = [s['text'] for s in plan['segments']]
                counts = [len(model.tokenizer.encode(t, add_special_tokens=True)) for t in texts]
                if counts != [s['token_count'] for s in plan['segments']]:
                    raise ValueError('Tokenizer differs from prepared input')
                if max(counts) > model.max_seq_length:
                    raise ValueError('Model would truncate a prepared input')
                context = nullcontext()
                if runtime['sdpa_backend'] == 'efficient':
                    from torch.nn.attention import sdpa_kernel, SDPBackend
                    context = sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)
                with context:
                    segments = np.asarray(model.encode(texts, batch_size=1, normalize_embeddings=True,
                        prompt='', show_progress_bar=True), dtype=np.float32)
                weights = [s['content_tokens'] for s in plan['segments']]
                paper = np.average(segments, axis=0, weights=weights)
                norm = np.linalg.norm(paper)
                if not np.isfinite(norm) or norm == 0:
                    raise ValueError('Invalid paper embedding')
                result = {'input_id': plan['id'], 'config': plan['config'], 'weights': weights,
                          'segment_vectors': segments.tolist(), 'paper_vector': (paper / norm).tolist(),
                          'execution': execution}
                validate_result(plan, result)
                write_json(path, result)
                row['status'] = 'embedded'
        except (Exception, KeyboardInterrupt) as error:
            row.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            row['elapsed_seconds'] = perf_counter() - started
            attempts.append(row)
            write_json(output_dir / 'last_run.json', {'execution': execution, 'papers': attempts})
            print(f"  {row['status']} ({row['elapsed_seconds']:.1f}s)", flush=True)
    return assemble_corpus(plans, output_dir)
