"""Apply recorded manual citation review without altering the original API graph.

Graph nodes represent unique works. The PDF-to-work table preserves all selected
PDF identities and their relation to the unchanged embedding corpus.
"""
from copy import deepcopy
from pathlib import Path
import re

import networkx as nx
import pandas as pd
from scipy import sparse

from .citation_graph import normalize_title, normalize_doi
from .openalex import export_graph
from .storage import read_json, write_json, file_hash


def apply_identity_review(matches, works, aliases, review):
    """Resolve only explicitly reviewed duplicates/records; reject stale identities."""
    matches, works, aliases = matches.copy(), deepcopy(works), aliases.copy()
    keys = set(matches.Key)
    canonical = {key: key for key in keys}
    by_key = matches.set_index('Key')
    for decision in review.get('duplicate_pdfs', []):
        duplicate, keep = decision['duplicate_key'], decision['canonical_key']
        if duplicate not in keys or keep not in keys or duplicate == keep:
            raise ValueError('Duplicate review references invalid PDF identities')
        if (canonical[duplicate] != duplicate or canonical[keep] != keep
                or duplicate in {v for k, v in canonical.items() if k != v}):
            raise ValueError('Duplicate groups must be direct, non-overlapping mappings')
        if not by_key.loc[keep, 'openalex_id'] or by_key.loc[duplicate, 'openalex_id'] != by_key.loc[keep, 'openalex_id']:
            raise ValueError('Reviewed duplicate work identities changed')
        canonical[duplicate] = keep
        matches.loc[matches.Key.eq(keep), ['approved', 'match_status']] = [True, 'reviewed_canonical_duplicate']
    for decision in review.get('work_overrides', []):
        key, work = decision['Key'], decision['work']
        if key not in keys or not work.get('id'):
            raise ValueError('Reviewed work override references an invalid identity')
        if normalize_doi(work.get('doi')) != normalize_doi(by_key.loc[key, 'openalex_doi']):
            raise ValueError('Reviewed work override DOI changed')
        if normalize_title(work['display_name']) != normalize_title(by_key.loc[key, 'openalex_title']):
            raise ValueError('Reviewed work override title changed')
        works[key] = work
        mask = matches.Key.eq(key)
        refs = work.get('referenced_works')
        updates = {'openalex_id': work['id'], 'openalex_title': work['display_name'],
                   'openalex_year': work['publication_year'], 'approved': True,
                   'match_status': 'reviewed_provider_duplicate', 'match_method': 'manual_record_review',
                   'identity_review_note': decision.get('reason', ''),
                   'review_version_status': decision.get('version_status', ''),
                   'reference_records': len(refs) if isinstance(refs, list) else None,
                   'reference_status': 'available_nonempty' if refs else 'empty_provider_list' if isinstance(refs, list) else 'missing_provider_list'}
        for column, value in updates.items():
            matches.loc[mask, column] = value
        aliases = pd.concat([aliases, pd.DataFrame([
            {'Key': key, 'openalex_id': identifier, 'method': 'manual_record_review'}
            for identifier in [work['id'], *decision.get('alternative_work_ids', [])]])], ignore_index=True)
    mapping = matches[['Key', 'paper_id', 'source_filename']].copy()
    mapping['canonical_key'] = mapping.Key.map(canonical)
    mapping['canonical_paper_id'] = mapping.canonical_key.map(matches.set_index('Key').paper_id)
    mapping['is_duplicate_pdf'] = mapping.Key.ne(mapping.canonical_key)
    aliases['Key'] = aliases.Key.map(canonical)
    matches = matches.loc[matches.Key.map(canonical).eq(matches.Key)].copy()
    return matches, works, aliases.drop_duplicates(), mapping


def export_reviewed_graph(root, base_dir, selection, client, *, review_file=None):
    """Reproduce the manually reviewed graph entirely from saved evidence.

    Manual additions recover internal citations only. They do not supply complete
    external bibliographies and must not be treated as complete coupling profiles.
    """
    root, base_dir = Path(root), Path(base_dir)
    review_file = Path(review_file) if review_file else root / 'configs/citation_graph_manual_review.json'
    review = read_json(review_file)
    if review['selection_id'] != selection['selection_id']:
        raise ValueError('Manual citation review belongs to a different corpus selection')
    pages_path = root / review['evidence_pages']
    if file_hash(pages_path) != review['evidence_pages_sha256']:
        raise ValueError('PDF citation evidence changed; repeat the manual review')
    pages = {(r['source_key'], r['pdf_page']): r['text'] for r in read_json(pages_path)}
    matches = pd.read_csv(base_dir / 'matches.csv').fillna('')
    works = read_json(base_dir / 'resolved_works.json')
    aliases = pd.read_csv(base_dir / 'identity_aliases.csv').fillna('')
    matches, works, aliases, mapping = apply_identity_review(matches, works, aliases, review)
    active = set(matches.loc[matches.approved, 'Key'])
    canonical = mapping.set_index('Key').canonical_key.to_dict()
    manual = []
    for row in review['manual_edges']:
        if row.get('approved') is not True:
            continue
        source, target = canonical[row['source_key']], canonical[row['target_key']]
        if source not in active or target not in active or source == target:
            raise ValueError('Manual citation must join two different approved works')
        page = pages.get((row['source_key'], row['pdf_page']), '')
        normalized = normalize_title(re.sub(r'[-\u00ad]\s*\r?\n\s*', '', page))
        if row['normalized_excerpt'] not in normalized:
            raise ValueError('Manual citation excerpt does not match its saved PDF page')
        manual.append({**row, 'source_key': source, 'target_key': target})
    manual = pd.DataFrame(manual).drop_duplicates(['source_key', 'target_key'])
    output = base_dir / 'reviewed'
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'matching_settings.json', {
        'baseline_settings': read_json(base_dir / 'matching_settings.json'),
        'manual_review_sha256': file_hash(review_file), 'unit': 'unique work',
    })
    nodes, edges, graph, summary = export_graph(matches, works, aliases, output, selection, client)
    edges['citation_source'] = 'openalex'
    edges['evidence_pdf_page'] = ''
    existing = set(zip(edges.source_key, edges.target_key))
    names = nodes.set_index('Key').paper_id.to_dict()
    additions = []
    for row in manual.to_dict('records'):
        pair = (row['source_key'], row['target_key'])
        if pair in existing:
            continue
        additions.append({'source_key': pair[0], 'target_key': pair[1],
                          'source_paper_id': names[pair[0]], 'target_paper_id': names[pair[1]],
                          'target_match_method': 'manually_verified_pdf_citation',
                          'citation_source': 'local_pdf_manual_review', 'evidence_pdf_page': row['pdf_page']})
        graph.add_edge(*pair)
    edges = pd.concat([edges, pd.DataFrame(additions)], ignore_index=True)
    nodes['internal_in_degree'] = nodes.Key.map(dict(graph.in_degree()))
    nodes['internal_out_degree'] = nodes.Key.map(dict(graph.out_degree()))
    nodes['manual_internal_citations'] = nodes.Key.map(manual.groupby('source_key').size()).fillna(0).astype(int)
    nodes['reference_coverage_note'] = nodes.manual_internal_citations.map(
        lambda n: 'partial_internal_pdf_recovery; external_bibliography_incomplete' if n else 'provider_list_completeness_unknown')
    refs = pd.read_csv(output / 'all_references.csv').fillna('')
    refs['citation_source'] = 'openalex'
    oa_ids = nodes.set_index('Key').openalex_id.to_dict()
    extra_refs = pd.DataFrame([{'source_key': r['source_key'], 'source_paper_id': names[r['source_key']],
        'source_openalex_id': oa_ids[r['source_key']], 'target_openalex_id': oa_ids[r['target_key']],
        'citation_source': 'local_pdf_manual_review'} for r in additions])
    refs = pd.concat([refs, extra_refs], ignore_index=True)
    refs.to_csv(output / 'all_references.csv', index=False)
    nodes.to_csv(output / 'nodes.csv', index=False)
    edges.to_csv(output / 'edges.csv', index=False)
    mapping.to_csv(output / 'pdf_to_work.csv', index=False)
    manual.to_csv(output / 'manual_citation_evidence.csv', index=False)
    aliases.to_csv(output / 'identity_aliases.csv', index=False)
    node_lookup = nodes.set_index('Key')
    for key in graph.nodes:
        row = node_lookup.loc[key]
        graph.nodes[key]['reference_coverage_note'] = row.reference_coverage_note
    for row in edges.to_dict('records'):
        graph.edges[row['source_key'], row['target_key']]['citation_source'] = row['citation_source']
    nx.write_graphml(graph, output / 'citation_graph.graphml')
    sparse.save_npz(output / 'adjacency.npz', nx.to_scipy_sparse_array(graph, nodelist=nodes.Key.tolist(), dtype='int8', format='csr'))
    summary.update(selected_pdfs=len(mapping), graph_work_nodes=len(nodes),
                   duplicate_pdf_aliases=int(mapping.is_duplicate_pdf.sum()),
                   unapproved_pdfs=int((~mapping.canonical_key.isin(active)).sum()),
                   manually_added_internal_edges=len(additions), internal_citation_edges=len(edges),
                   all_reference_links=len(refs), partially_recovered_reference_lists=int(nodes.manual_internal_citations.gt(0).sum()),
                   isolated_pdfs=None, isolated_works=len(list(nx.isolates(graph))),
                   weak_component_sizes=sorted((len(c) for c in nx.weakly_connected_components(graph)), reverse=True))
    write_json(output / 'graph_summary.json', summary)
    manifest = read_json(output / 'run_manifest.json')
    manifest.update(summary=summary, curation_code_sha256=file_hash(__file__),
                    manual_review_sha256=file_hash(review_file), evidence_pages_sha256=file_hash(pages_path),
                    baseline_files={name: file_hash(base_dir / name) for name in ['matches.csv', 'resolved_works.json', 'identity_aliases.csv']})
    write_json(output / 'run_manifest.json', manifest)
    return nodes, edges, graph, summary, mapping


def extract_pdf_review_candidates(root, base_dir):
    """Recreate PDF-page evidence and title candidates; never approve them.

    Scans only the baseline's eight (or current selection's) empty provider lists.
    Full-title occurrences can be false positives; decisions live separately in
    the review configuration. Local PDFs must match the manifest SHA-256 values.
    """
    import json
    import pypdfium2 as pdfium

    root, base_dir = Path(root), Path(base_dir)
    nodes = pd.read_csv(base_dir / 'nodes.csv').fillna('')
    output = base_dir / 'manual_review'
    output.mkdir(parents=True, exist_ok=True)
    targets = nodes.to_dict('records')
    pages, candidates = [], []
    for source in nodes.loc[nodes.reference_status.eq('empty_provider_list')].to_dict('records'):
        path = root / 'data/pdfs' / source['source_filename']
        if file_hash(path) != source['Key']:
            raise ValueError(f"PDF hash changed: {source['paper_id']}")
        doc = pdfium.PdfDocument(path)
        try:
            for index in range(len(doc)):
                page = doc[index]
                textpage = page.get_textpage()
                text = textpage.get_text_range()
                textpage.close()
                page.close()
                pages.append({'source_key': source['Key'], 'source_paper_id': source['paper_id'],
                              'pdf_page': index + 1, 'text': text})
                normalized = normalize_title(re.sub(r'[-\u00ad]\s*\r?\n\s*', '', text))
                for target in targets:
                    if target['Key'] == source['Key']:
                        continue
                    for column in ['query_title', 'openalex_title', 'approved_version_title']:
                        title = normalize_title(target.get(column, ''))
                        if len(title.split()) < 4:
                            continue
                        pos = normalized.find(title)
                        if pos >= 0:
                            candidates.append({'source_key': source['Key'], 'target_key': target['Key'],
                                'source_paper_id': source['paper_id'], 'target_paper_id': target['paper_id'],
                                'pdf_page': index + 1, 'target_title': target['query_title'],
                                'matched_title': target[column],
                                'normalized_excerpt': normalized[max(0, pos-170):pos+len(title)+180]})
                            break
        finally:
            doc.close()
    (output / 'pdf_pages.json').write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding='utf-8')
    result = pd.DataFrame(candidates)
    result.to_csv(output / 'pdf_citation_candidates.csv', index=False)
    return result
