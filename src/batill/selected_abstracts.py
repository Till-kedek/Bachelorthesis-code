"""Auditable title/abstract inputs and BGE-M3 vectors for the selected PDF works.

The canonical selection is shared with the PDF notebook. Full-paper vectors are
never used as inputs here. Every candidate retains exact cached-document spans;
uncertain or absent abstracts remain in the coverage audit rather than becoming
introductions or generated summaries. Explicitly approved external abstracts and
publisher descriptions are stored separately with source checksums and labels.
"""
from datetime import datetime, timezone
from difflib import SequenceMatcher
from importlib.metadata import version
import json
from pathlib import Path
import re
import runpy
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd

from .abstract_clustering import load_abstract_embeddings
from .selection import filter_analysis_corpus, deduplicate_pdf_corpus
from .storage import file_hash, fingerprint, read_json, write_json


REFERENCE_RUN = 'outputs/abstract_embeddings/bge_m3/20260930T174845_508054Z_full_3c498a9d'
WORKFLOW_CONFIG = 'configs/selected_abstracts_input.json'
MARKER = re.compile(r'^\s*(?:a\s*b\s*s\s*t\s*r\s*a\s*c\s*t|summary)\b\s*[:.\-–—]?\s*', re.I)
STRUCTURED = re.compile(r'^(?:importance|objective[s]?|design\b|setting\b|participants?\b|data sources\b|study selection\b|eligibility criteria\b|data extraction\b|data synthesis\b|exposures?|main outcomes|results|conclusions?\b|manuscript type\b|research (?:question|issue|findings)|theoretical|academic implications|practitioner)', re.I)
BOUNDARY = re.compile(r'^(?:(?:\d+|[IVX]+)[.\s:–-]+)?(?:introduction|keywords?|key\s+words|jel\b|acknowledg|references|bibliography|article\s+history|received\b|accepted\b|available online|copyright|©|corresponding author|e-?mail\b|author (?:information|affiliations|contributions))', re.I)
BOILERPLATE = re.compile(r'^(?:we (?:thank|are grateful)|the authors (?:thank|acknowledge)|this (?:paper|work|research) (?:was|is) (?:supported|funded)|this (?:article|paper) is (?:brought|distributed)|it is not permitted|if you believe|download date|electronic copy|all rights reserved)', re.I)
EXCLUDED_BLOCKS = {'PageHeader', 'PageFooter', 'Footnote', 'Reference', 'Table', 'TableOfContents', 'Figure', 'Picture', 'Caption', 'Equation'}


def _cleaners(root):
    """Use exactly the cleaning functions used for the large abstract corpus."""
    namespace = runpy.run_path(str(Path(root) / 'scripts/clean_abstracts.py'))
    return namespace['normalize'], namespace['clean_abstract']


def load_selected_documents(root):
    """Reproduce the PDF notebook's scope/duplicate decisions using manifest rows."""
    root = Path(root).resolve()
    manifest_path = root / 'outputs/corpus_sections/embeddings/manifest.json'
    manifest = read_json(manifest_path)
    papers = pd.DataFrame([{
        'paper_id': f'P{i+1:03d}', 'Key': row['pdf_sha256'],
        'pdf_sha256': row['pdf_sha256'], 'title': row['title'],
        'source_filename': row['source_filename'],
    } for i, row in enumerate(manifest['rows'])])
    # These are row indices for shared selection helpers, NOT embedding vectors.
    row_indices = np.arange(len(papers)).reshape(-1, 1)
    exclusion_file = root / 'configs/analysis_exclusions.json'
    papers, row_indices, exclusions, selection = filter_analysis_corpus(
        papers, row_indices, exclusion_file)
    papers, _, selection = deduplicate_pdf_corpus(papers, row_indices, exclusion_file, selection)
    extraction_root = root / 'outputs/corpus/extraction'
    report = read_json(extraction_root / 'report.json')
    entries = {}
    for entry in report:
        if entry.get('status') == 'extracted':
            key = entry['pdf_sha256']
            if key in entries:
                raise ValueError(f'Duplicate extracted document for {key}')
            entries[key] = entry
    documents, hashes = {}, {
        'pdf_manifest': file_hash(manifest_path),
        'exclusions': file_hash(exclusion_file),
        'duplicates': file_hash(exclusion_file.with_name('pdf_duplicates.json')),
        'extraction_report': file_hash(extraction_root / 'report.json'),
        'cleaning_code': file_hash(root / 'scripts/clean_abstracts.py'),
    }
    for row in papers.itertuples():
        if row.pdf_sha256 not in entries:
            raise ValueError(f'Missing cached extraction for {row.paper_id}')
        path = (extraction_root / entries[row.pdf_sha256]['document_path']).resolve()
        if not path.is_relative_to(extraction_root.resolve()):
            raise ValueError('Extraction document must be inside the extraction folder')
        document = read_json(path)
        if document['pdf_sha256'] != row.pdf_sha256:
            raise ValueError(f'Document identity mismatch for {row.paper_id}')
        documents[row.pdf_sha256] = (document, path)
        hashes[f'document:{row.paper_id}'] = file_hash(path)
    # Metadata is used for inspection only, never substituted for PDF abstract text.
    graph_dir = root / 'outputs/citation_graph_openalex' / selection['scope_selection_id'][:16] / 'reviewed'
    nodes = pd.read_csv(graph_dir / 'nodes.csv', dtype=str, keep_default_na=False)
    if not nodes.Key.is_unique or not set(papers.Key) <= set(nodes.Key):
        raise ValueError('Reviewed work metadata must cover every selected PDF exactly once')
    nodes = nodes.set_index('Key').loc[papers.Key].reset_index()
    if nodes.paper_id.tolist() != papers.paper_id.tolist():
        raise ValueError('Canonical Paper IDs disagree with the reviewed work mapping')
    papers['doi'] = nodes['openalex_doi'].where(nodes['openalex_doi'].ne(''), nodes['doi'])
    papers['year'] = nodes['openalex_year']
    papers['journal'] = ''  # The reviewed PDF table does not supply this field.
    papers['document_language'] = ''
    papers['document_type'] = ''
    papers['reviewed_version_title'] = nodes['approved_version_title']
    hashes['reviewed_nodes'] = file_hash(graph_dir / 'nodes.csv')
    return papers, documents, exclusions, selection, hashes


def span_text(document, spans):
    """Validate and reproduce exact source ranges; never accept invented text."""
    blocks = {b['id']: b for b in document['blocks']}
    if len(blocks) != len(document['blocks']):
        raise ValueError('Duplicate source block IDs')
    text, occupied = [], {}
    for span in spans:
        block = blocks[span['block_id']]
        a, b = span['start'], span['end']
        if (type(a) is not int or type(b) is not int or not 0 <= a < b <= len(block['text'])
                or span['page'] != block['page']):
            raise ValueError('Invalid abstract source span')
        previous = occupied.setdefault(span['block_id'], [])
        if any(a < end and b > start for start, end in previous):
            raise ValueError('Overlapping or duplicated source spans')
        previous.append((a, b))
        text.append(block['text'][a:b])
    return '\n\n'.join(text)


def _span(block, start=0, end=None):
    return {'block_id': block['id'], 'page': block['page'],
            'start': start, 'end': len(block['text']) if end is None else end}


def extract_abstract(document, *, max_front_pages=5):
    """Find explicit/structured abstracts; leave unlabeled prose for review.

    Marker section hierarchy can include footnotes and the whole introduction,
    so boundaries are examined explicitly rather than concatenating a section.
    """
    blocks = document['blocks']
    front = [(i, b) for i, b in enumerate(blocks)
             if b.get('page') is not None and b['page'] <= max_front_pages
             and b.get('include', True) and b['type'] not in EXCLUDED_BLOCKS]
    markers = [(i, MARKER.match(b['text'])) for i, b in front if MARKER.match(b['text'])]
    mode, start, offset = None, None, 0
    if markers:
        start, match = markers[0]
        offset = match.end()
        mode = 'explicit_abstract_heading'
    else:
        starts = [i for i, b in front if re.match(r'^\s*IMPORTANCE\b', b['text'])]
        if starts:
            start = starts[0]
            later = ' '.join(b['text'] for i, b in front if i >= start)
            if re.search(r'\bOBJECTIVES?\b', later) and re.search(r'\bCONCLUSIONS?\b', later):
                mode = 'structured_abstract'
    spans, ended, flags = [], False, []
    if mode:
        first_page = blocks[start]['page']
        concluded = False
        for i in range(start, len(blocks)):
            b = blocks[i]
            if b.get('page') is None or b['page'] > min(max_front_pages, first_page + 2):
                break
            if b['type'] in EXCLUDED_BLOCKS or not b.get('include', True):
                continue
            text = b['text']
            a = offset if i == start else 0
            remaining = text[a:].strip()
            if not remaining:
                continue
            if i > start and (BOUNDARY.match(remaining) or BOILERPLATE.match(remaining)
                              or MARKER.match(remaining)):
                ended = True
                break
            if i > start and b['type'] == 'SectionHeader' and not STRUCTURED.match(remaining):
                ended = True
                break
            if mode == 'structured_abstract' and concluded:
                ended = True
                break
            # Keywords/JEL and publication notices sometimes share the last prose block.
            cut = re.search(r'(?im)(?:\n|(?<=[.!?])\s+)(?:key\s*words?\s*:|jel\s+(?:classifications?|codes?)\s*:|©|copyright\b|\d+\.?\s+introduction\b)', text[a:])
            end = a + cut.start() if cut else len(text)
            if end > a and text[a:end].strip():
                spans.append(_span(b, a, end))
            if cut:
                ended = True
                break
            if re.match(r'^CONCLUSIONS?\b', remaining, re.I) and len(remaining.split()) > 8:
                concluded = True
        raw = span_text(document, spans)
        words = len(raw.split())
        if not ended and not concluded:
            flags.append('no_confirmed_end_boundary')
        if words < 30 or words > 1000:
            flags.append('inspect_abstract_length')
        if len(markers) > 1:
            flags.append('multiple_abstract_markers')
        if not raw.strip():
            flags.append('empty_after_heading')
        if re.search(r'\b(?:we thank|acknowledg|corresponding author|electronic copy available)\b', raw, re.I):
            flags.append('possible_front_matter_contamination')
        # A heading alone cannot establish boundaries: PDF layout frequently
        # interleaves affiliations or omits abstract continuations. Require a
        # same-DOI text validation or a checksum-bound source review below.
        flags.append('abstract_boundary_requires_review')
        return {'status': 'review', 'method': mode,
                'spans': spans, 'raw_abstract': raw, 'flags': ';'.join(flags)}
    # Show candidates for human inspection; NEVER label the opening prose an abstract.
    candidates = []
    for i, b in front:
        text = b['text'].strip()
        if b['type'] in {'Text', 'TextInlineMath'} and 40 <= len(text.split()) <= 1000:
            if not BOUNDARY.match(text) and not BOILERPLATE.match(text):
                candidates.append(_span(b))
    return {'status': 'review' if candidates else 'missing',
            'method': 'unlabelled_candidate' if candidates else 'no_abstract_found',
            'spans': candidates[:1],
            'candidate_spans': candidates[:8],
            'raw_abstract': span_text(document, candidates[:1]),
            'flags': 'abstract_boundary_requires_review' if candidates else 'no_abstract_found'}


def _doi(value):
    return re.sub(r'^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)', '', str(value).strip().casefold())


def match_pdf_abstract(document, reference_abstract):
    """Validate unlabeled PDF prose against a same-DOI abstract, without copying it.

    Only exact PDF source spans are returned. Near-complete token agreement is
    required; a publisher abstract from a different version is not substituted.
    """
    tokens = lambda text: re.findall(r'\w+', text.casefold().replace('\u00ad', ''))
    target = tokens(reference_abstract)
    if not target:
        return None
    blocks = [b for b in document['blocks'] if b.get('page') is not None and b['page'] <= 5
              and b.get('include', True) and b['type'] not in EXCLUDED_BLOCKS and b['text'].strip()]
    best = None
    for i, block in enumerate(blocks):
        spans = []
        for following in blocks[i:i+8]:
            text = following['text']
            match = MARKER.match(text)
            start = match.end() if match else 0
            if len(text[start:].strip()) == 0:
                continue
            if BOUNDARY.match(text[start:]) or BOILERPLATE.match(text[start:]):
                break
            # Citation/keyword suffixes are audited by the common cleaner later.
            stop = re.search(r'(?i)(?:\n|(?<=[.!?])\s+)(?:key\s*words?\s*:|jel\s+(?:classification|codes?)\s*:|©)', text[start:])
            end = start + stop.start() if stop else len(text)
            if start >= end:
                break
            spans.append(_span(following, start, end))
            candidate = tokens(span_text(document, spans))
            if len(candidate) > len(target) * 1.12:
                break
            if len(candidate) < len(target) * .9:
                continue
            ratio = SequenceMatcher(None, target, candidate, autojunk=False).ratio()
            if ratio >= .94 and (best is None or ratio > best['match_score']):
                best = {'status': 'ready', 'method': 'pdf_spans_validated_against_same_doi_abstract',
                        'spans': spans.copy(), 'raw_abstract': span_text(document, spans),
                        'flags': '', 'match_score': ratio}
            if stop:
                break
    return best


def prepare_selected_abstracts(root, *, overrides=None):
    """Audit all selected works; return only ready inputs plus the complete audit.

    Overrides select exact block spans or mark a work as having no abstract.
    Each decision must cite the current document hash and a review reason.
    """
    root = Path(root).resolve()
    papers, documents, exclusions, selection, hashes = load_selected_documents(root)
    normalize, clean_abstract = _cleaners(root)
    reference_source = root / 'data/abstracts/abstracts_clean.csv'
    reference = pd.read_csv(reference_source, dtype=str, keep_default_na=False)
    reference['_doi'] = reference.doi.map(_doi)
    duplicate_dois = set(reference.loc[reference._doi.duplicated(keep=False), '_doi'])
    by_doi = {row['_doi']: row for row in reference.to_dict('records')
              if row['_doi'] and row['_doi'] not in duplicate_dois}
    hashes['reference_abstract_source'] = file_hash(reference_source)
    supplement_file = root / 'configs/selected_abstracts_supplements.json'
    supplements = read_json(supplement_file) if supplement_file.exists() else {}
    if not set(supplements) <= set(papers.pdf_sha256):
        raise ValueError('Abstract supplements contain unknown or excluded PDF identities')
    if supplements:
        hashes['abstract_supplements'] = file_hash(supplement_file)
    overrides = {} if overrides is None else overrides
    if not set(overrides) <= set(papers.pdf_sha256):
        raise ValueError('Abstract overrides contain unknown or excluded PDF identities')
    rows = []
    for paper in papers.to_dict('records'):
        document, path = documents[paper['pdf_sha256']]
        document_hash = file_hash(path)
        result = extract_abstract(document)
        reference_row = by_doi.get(_doi(paper['doi']))
        # This is validation of PDF spans, not a metadata fallback. It also
        # catches partial/misbounded abstracts produced by document layout.
        if reference_row:
            matched = match_pdf_abstract(document, reference_row['abstract'])
            if matched:
                result = matched
        review_reason = ''
        if paper['pdf_sha256'] in overrides:
            decision = overrides[paper['pdf_sha256']]
            if decision.get('document_sha256') != document_hash or not decision.get('reason', '').strip():
                raise ValueError(f'Stale/unexplained abstract override for {paper["paper_id"]}')
            action = decision.get('action', 'keep_extraction')
            if action == 'select_spans':
                spans = decision['spans']
                raw = span_text(document, spans)
                if not raw.strip():
                    raise ValueError('Reviewed abstract cannot be empty')
                result.update(status='ready', method='reviewed_pdf_spans', spans=spans,
                              raw_abstract=raw, flags='')
            elif action == 'no_abstract':
                result.update(status='missing', method='reviewed_no_abstract', spans=[], raw_abstract='', flags='no_abstract')
            elif action == 'needs_review':
                result.update(status='review', flags='reviewed_extraction_problem')
            elif action != 'keep_extraction':
                raise ValueError('Unsupported abstract review action')
            if action != 'keep_extraction':
                result.pop('match_score', None)
            review_reason = decision['reason']
        source_kind, source_url, source_record = 'pdf_abstract', '', ''
        source_path, source_checksum = str(path.relative_to(root)), document_hash
        supplement = supplements.get(paper['pdf_sha256'])
        if supplement:
            if supplement.get('paper_id') != paper['paper_id'] or not supplement.get('identity_review', '').strip():
                raise ValueError('Supplement identity is missing or differs from the selected paper')
            source_kind = supplement['source_kind']
            if source_kind not in {'external_full_abstract', 'publisher_short_description'}:
                raise ValueError('Unsupported supplementary abstract source kind')
            source_path, source_checksum = supplement['source_path'], supplement['source_sha256']
            cached = (root / source_path).resolve()
            if not cached.is_relative_to(root) or file_hash(cached) != source_checksum:
                raise ValueError('Supplement source checksum mismatch')
            record = read_json(cached)
            origin = (root / record['origin_path']).resolve()
            if not origin.is_relative_to(root) or file_hash(origin) != record['origin_sha256']:
                raise ValueError('Supplement original source checksum mismatch')
            source_url, source_record = record['source_url'], record['source_record_id']
            if (record['paper_id'] != paper['paper_id'] or record['pdf_sha256'] != paper['pdf_sha256']
                    or record['source_kind'] != source_kind or not source_url
                    or not record['abstract'].strip() or not record.get('retrieved_utc')):
                raise ValueError('Supplement source record is incomplete or mismatched')
            if record.get('doi') and _doi(record['doi']) != _doi(paper['doi']):
                raise ValueError('Supplement DOI differs from the selected paper')
            hashes[f'supplement:{paper["paper_id"]}'] = source_checksum
            result.update(status='ready', method=source_kind, spans=[], candidate_spans=[],
                          raw_abstract=record['abstract'], flags='short_description' if source_kind == 'publisher_short_description' else '')
            result.pop('match_score', None)
            review_reason = supplement['identity_review']
        title_raw = paper['title']
        title_source, title_spans = 'pdf_embedding_manifest', []
        decision = overrides.get(paper['pdf_sha256'], {})
        if decision.get('title_spans'):
            title_spans = decision['title_spans']
            title_raw = span_text(document, title_spans)
            title_source = 'reviewed_pdf_spans'
        elif decision.get('use_reviewed_version_title'):
            title_raw = paper['reviewed_version_title']
            if not title_raw.strip():
                raise ValueError('No previously reviewed version title is available')
            title_source = 'previously_reviewed_version_metadata'
        title = normalize(re.sub(r'[∗*†‡§¶✩☆]+$', '', title_raw).strip())
        abstract, removed_notice = clean_abstract(result['raw_abstract'])
        if not title or not abstract:
            result['status'] = 'missing' if not abstract else 'review'
        rows.append({**paper, 'manifest_title': paper['title'], 'title': title, 'title_raw': title_raw,
                     'title_source': title_source, 'title_spans': json.dumps(title_spans, ensure_ascii=False),
                     'abstract': abstract, 'abstract_word_count': len(abstract.split()),
                     'embedding_text': '\n\n'.join(part for part in (title, abstract) if part),
                     'status': result['status'], 'extraction_method': result['method'],
                     'abstract_source_kind': source_kind, 'abstract_source_url': source_url,
                     'abstract_source_record_id': source_record,
                     'abstract_source_path': source_path, 'abstract_source_sha256': source_checksum,
                     'is_short_description': source_kind == 'publisher_short_description',
                     'extraction_flags': result['flags'], 'review_reason': review_reason,
                     'document_path': str(path.relative_to(root)), 'document_sha256': document_hash,
                     'abstract_spans': json.dumps(result['spans'], ensure_ascii=False),
                     'candidate_spans': json.dumps(result.get('candidate_spans', []), ensure_ascii=False),
                     'validation_scopus_id': reference_row['paper_id'] if result.get('match_score') else '',
                     'validation_match_score': result.get('match_score', ''),
                     'raw_abstract': result['raw_abstract'], 'removed_notice': removed_notice})
    audit = pd.DataFrame(rows)
    ready = audit.loc[audit.status.eq('ready')].copy().reset_index(drop=True)
    provenance = {'selection': selection, 'input_hashes': hashes,
                  'overrides_id': fingerprint(overrides),
                  'selected_works': len(audit), 'ready_abstracts': len(ready),
                  'review_abstracts': int(audit.status.eq('review').sum()),
                  'missing_abstracts': int(audit.status.eq('missing').sum()),
                  'external_full_abstracts': int(audit.abstract_source_kind.eq('external_full_abstract').sum()),
                  'publisher_short_descriptions': int(audit.is_short_description.sum()),
                  'abstract_source': 'PDF spans plus explicitly reviewed external abstracts and short descriptions',
                  'extraction_code_sha256': file_hash(__file__)}
    return ready, audit, provenance


def save_preparation(root, ready, audit, provenance):
    """Save an immutable preparation bundle, including all pending/missing works."""
    root = Path(root).resolve()
    identity = fingerprint({'records': audit.to_dict('records'), 'provenance': provenance})
    folder = root / 'outputs/selected_abstracts/prepared' / identity[:20]
    folder.mkdir(parents=True, exist_ok=True)
    artifacts = {'abstracts_clean.csv': ready, 'extraction_audit.csv': audit,
                 'supplementary_abstracts.csv': audit.loc[audit.abstract_source_kind.ne('pdf_abstract')],
                 'pending_review.csv': audit.loc[~audit.status.eq('ready')],
                 'missing_abstracts.csv': audit.loc[audit.status.eq('missing')],
                 'extraction_review_needed.csv': audit.loc[audit.status.eq('review')]}
    for name, table in artifacts.items():
        text = table.to_csv(index=False)
        path = folder / name
        if path.exists() and path.read_text() != text:
            raise ValueError(f'Preparation bundle changed unexpectedly: {path}')
        if not path.exists():
            path.write_text(text, encoding='utf-8')
    manifest = {**provenance, 'preparation_id': identity,
                'artifact_sha256': {name: file_hash(folder / name) for name in artifacts}}
    manifest_path = folder / 'preparation_manifest.json'
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ValueError('Preparation manifest differs from existing immutable bundle')
    if not manifest_path.exists():
        write_json(manifest_path, manifest)
    return folder, manifest


def reference_settings(root):
    """Authenticate the large-corpus representation against its completed run."""
    folder = Path(root) / REFERENCE_RUN
    summary = read_json(folder / 'run_summary.json')
    if (summary.get('status') != 'complete' or summary.get('run_mode') != 'full'
            or summary.get('model') != 'BAAI/bge-m3' or summary.get('shape', [0, 0])[1] != 1024
            or summary.get('pooling') != 'model-defined CLS' or summary.get('normalization') != 'L2'
            or summary.get('truncation') is not False or summary.get('chunking') is not False
            or summary.get('input_format') != 'title + blank line + abstract'
            or summary.get('input_column') != 'embedding_text'
            or summary.get('max_tokens') != 8192 or summary.get('dtype') != 'float32'
            or summary.get('model_revision') != '5617a9f61b028005a4858fdac845db406aefb181'):
        raise ValueError('Reference abstract run has incompatible representation settings')
    return summary, folder


def embed_selected_abstracts(root, prepared_dir, *, allow_partial=False, device='auto', batch_size=8):
    """Run the same pinned BGE-M3 encoder; reuse an authenticated completed run.

    No remote model/API calls are needed: use the already downloaded exact
    reference revision. Pending records require an explicit partial-corpus choice.
    """
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    root, prepared_dir = Path(root).resolve(), Path(prepared_dir).resolve()
    preparation = read_json(prepared_dir / 'preparation_manifest.json')
    for name, checksum in preparation['artifact_sha256'].items():
        if file_hash(prepared_dir / name) != checksum:
            raise ValueError(f'Preparation file changed: {name}')
    if preparation['ready_abstracts'] != preparation['selected_works'] and not allow_partial:
        raise ValueError(f"{preparation['ready_abstracts']} of {preparation['selected_works']} selected works have ready abstracts. "
                         'Review pending_review.csv or explicitly set ALLOW_PARTIAL_CORPUS=True; no introduction is substituted.')
    source = prepared_dir / 'abstracts_clean.csv'
    papers = pd.read_csv(source, dtype=str, keep_default_na=False)
    if len(papers) < 4 or not papers.paper_id.is_unique or not papers.pdf_sha256.is_unique:
        raise ValueError('Need at least four uniquely identified verified abstracts')
    if papers.abstract.str.strip().eq('').any() or not papers.status.eq('ready').all():
        raise ValueError('Only nonempty ready abstract inputs may be embedded')
    expected = ['\n\n'.join(p for p in (t, a) if p) for t, a in zip(papers.title, papers.abstract)]
    if papers.embedding_text.tolist() != expected:
        raise ValueError('Embedding inputs must equal title + blank line + abstract')
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be positive')
    reference, ref_folder = reference_settings(root)
    identity = fingerprint({'source_sha256': file_hash(source), 'preparation_id': preparation['preparation_id'],
                            'reference_summary_sha256': file_hash(ref_folder / 'run_summary.json')})
    base = root / 'outputs/selected_abstracts/embeddings' / identity[:20]
    base.mkdir(parents=True, exist_ok=True)
    pointer = base / 'completed_run.json'
    if pointer.exists():
        run = root / read_json(pointer)['run_dir']
        _, _, provenance = load_selected_embeddings(run, source)
        if provenance['representation_id'] != identity:
            raise ValueError('Saved representation identity differs')
        print('Reusing verified BGE-M3 run:', run)
        return run
    snapshot = root / '.cache/huggingface/hub/models--BAAI--bge-m3/snapshots' / reference['model_revision']
    if not snapshot.is_dir():
        raise FileNotFoundError(f'The reference model revision is not cached: {snapshot}')
    actual_device = device
    if device == 'auto':
        actual_device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if actual_device not in {'cpu', 'mps', 'cuda'}:
        raise ValueError('Choose auto, cpu, mps or cuda')
    if actual_device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS is unavailable in this process')
    if actual_device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable in this process')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run = base / f'{stamp}_{uuid4().hex[:8]}'
    run.mkdir()
    summary = {key: reference[key] for key in ('model', 'model_revision', 'max_tokens', 'truncation',
        'chunking', 'input_column', 'input_format', 'pooling', 'normalization', 'dtype')}
    summary.update(status='started', run_mode='full', corpus_size=len(papers), selected_size=len(papers),
                   source=str(source), source_sha256=file_hash(source), device=actual_device,
                   batch_size=batch_size, representation_id=identity,
                   selected_pdf_works=preparation['selected_works'],
                   external_full_abstracts=preparation.get('external_full_abstracts', 0),
                   publisher_short_descriptions=preparation.get('publisher_short_descriptions', 0),
                   pending_or_missing_works=preparation['selected_works'] - len(papers),
                   partial_corpus_allowed=bool(allow_partial),
                   preparation_dir=str(prepared_dir.relative_to(root)),
                   preparation_manifest_sha256=file_hash(prepared_dir / 'preparation_manifest.json'),
                   reference_run=str(ref_folder.relative_to(root)),
                   reference_summary_sha256=file_hash(ref_folder / 'run_summary.json'),
                   packages={name: version(name) for name in ('torch', 'sentence-transformers',
                       'transformers', 'numpy', 'pandas')})
    started = perf_counter()
    write_json(run / 'run_summary.json', summary)

    def synchronize():
        if actual_device == 'cuda':
            torch.cuda.synchronize()
        elif actual_device == 'mps':
            torch.mps.synchronize()

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
        counts = []
        for start in range(0, len(papers), 128):
            encoded = tokenizer(papers.embedding_text.iloc[start:start+128].tolist(), truncation=False,
                                padding=False, add_special_tokens=True, return_token_type_ids=False, verbose=False)
            counts.extend(map(len, encoded['input_ids']))
        papers['token_count'] = counts
        papers['over_limit'] = papers.token_count > reference['max_tokens']
        papers.insert(0, 'embedding_row', np.arange(len(papers)))
        papers.to_csv(run / 'selected_papers.csv', index=False)
        papers[['paper_id', 'title', 'token_count', 'over_limit']].to_csv(run / 'token_lengths.csv', index=False)
        summary.update(corpus_over_limit=int(papers.over_limit.sum()), selected_over_limit=int(papers.over_limit.sum()),
                       maximum_corpus_tokens=int(max(counts)), maximum_sample_tokens=int(max(counts)))
        if papers.over_limit.any():
            raise ValueError('An input exceeds the reference model token limit; no truncation is permitted')
        loading_start = perf_counter()
        model = SentenceTransformer(str(snapshot), device=actual_device, local_files_only=True,
                                    model_kwargs={'dtype': torch.float32}, trust_remote_code=False)
        if model.get_sentence_embedding_dimension() != 1024 or model.max_seq_length < 8192:
            raise ValueError('Cached model dimensions/context differ from the reference run')
        model.eval()
        synchronize()
        summary['model_load_seconds'] = perf_counter() - loading_start

        def encode(texts):
            return model.encode(texts, batch_size=batch_size, show_progress_bar=False, prompt='',
                                convert_to_numpy=True, normalize_embeddings=True, precision='float32')

        with torch.inference_mode():
            warm = perf_counter()
            encode(papers.embedding_text.iloc[:batch_size].tolist())
            synchronize()
            summary['warmup_seconds'] = perf_counter() - warm
            inference_start = perf_counter()
            vectors, timings = [], []
            for start in range(0, len(papers), batch_size):
                batch_start = perf_counter()
                texts = papers.embedding_text.iloc[start:start+batch_size].tolist()
                vectors.append(encode(texts))
                synchronize()
                timings.append({'start_row': start, 'papers': len(texts), 'seconds': perf_counter() - batch_start})
                print(f'Embedded {start+len(texts)}/{len(papers)} selected abstracts', flush=True)
            summary['inference_seconds'] = perf_counter() - inference_start
        vectors = np.concatenate(vectors).astype(np.float32, copy=False)
        if (vectors.shape != (len(papers), 1024) or not np.isfinite(vectors).all()
                or not np.allclose(np.linalg.norm(vectors, axis=1), 1., atol=1e-5, rtol=0)):
            raise ValueError('Invalid BGE-M3 output vectors')
        np.save(run / 'embeddings.npy', vectors, allow_pickle=False)
        papers[['embedding_row', 'paper_id']].to_csv(run / 'paper_ids.csv', index=False)
        pd.DataFrame(timings).to_csv(run / 'batch_timings.csv', index=False)
        summary.update(status='complete', shape=list(vectors.shape), total_seconds=perf_counter()-started,
                       artifact_sha256={name: file_hash(run / name) for name in
                           ('embeddings.npy', 'selected_papers.csv', 'paper_ids.csv', 'token_lengths.csv')})
        write_json(run / 'run_summary.json', summary)
        load_selected_embeddings(run, source)
        write_json(pointer, {'run_dir': str(run.relative_to(root)), 'representation_id': identity})
        return run
    except Exception as exc:
        summary.update(status='failed', error=f'{type(exc).__name__}: {exc}', total_seconds=perf_counter()-started)
        write_json(run / 'run_summary.json', summary)
        raise


def load_selected_embeddings(run_dir, source_csv):
    """Use the same vector validation, retaining the selected PDF identities."""
    run_dir = Path(run_dir)
    summary = read_json(run_dir / 'run_summary.json')
    for name, checksum in summary.get('artifact_sha256', {}).items():
        if file_hash(run_dir / name) != checksum:
            raise ValueError(f'Selected embedding artifact checksum mismatch: {name}')
    if not summary.get('artifact_sha256') or not summary.get('representation_id'):
        raise ValueError('Expected an authenticated selected-PDF abstract embedding run')
    papers, vectors, provenance = load_abstract_embeddings(run_dir, source_csv)
    if (vectors.shape[1] != 1024 or summary.get('pooling') != 'model-defined CLS'
            or summary.get('dtype') != 'float32'
            or summary.get('model_revision') != '5617a9f61b028005a4858fdac845db406aefb181'):
        raise ValueError('Selected embeddings differ from the reference BGE-M3 representation')
    if 'pdf_sha256' not in papers or not papers.pdf_sha256.is_unique:
        raise ValueError('Selected abstract rows must preserve unique PDF hashes')
    papers['Key'] = papers.pdf_sha256
    papers['text'] = papers.embedding_text
    return papers, vectors, provenance


def activate_selected_run(root, run_dir, prepared_dir):
    """Explicit local handoff for the two downstream notebooks; no latest-run search."""
    root = Path(root).resolve()
    run_dir, prepared_dir = Path(run_dir).resolve(), Path(prepared_dir).resolve()
    source = prepared_dir / 'abstracts_clean.csv'
    papers, _, provenance = load_selected_embeddings(run_dir, source)
    config = {'schema_version': 1, 'run_dir': str(run_dir.relative_to(root)),
              'source_csv': str(source.relative_to(root)), 'corpus_size': len(papers),
              'representation_id': provenance['representation_id'],
              'run_summary_sha256': file_hash(run_dir / 'run_summary.json')}
    write_json(root / WORKFLOW_CONFIG, config)
    return config


def resolve_selected_run(root):
    root = Path(root).resolve()
    path = root / WORKFLOW_CONFIG
    if not path.exists():
        raise FileNotFoundError('Run Embedding Elite Paper Abstracts.ipynb to prepare and activate the new vectors first.')
    config = read_json(path)
    run, source = root / config['run_dir'], root / config['source_csv']
    papers, _, provenance = load_selected_embeddings(run, source)
    if (file_hash(run / 'run_summary.json') != config['run_summary_sha256']
            or provenance['representation_id'] != config['representation_id']
            or len(papers) != config['corpus_size']):
        raise ValueError('Selected abstract handoff is stale; rerun the embedding handoff cell')
    return run, source
