"""Match selected PDFs to Semantic Scholar and export their induced citation graph."""
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from importlib.metadata import version
import re
import unicodedata

import numpy as np
import pandas as pd

from .selection import filter_analysis_corpus
from .semantic_scholar import ScholarError
from .storage import fingerprint, read_json, write_json, file_hash


def normalize_title(value):
    value = unicodedata.normalize('NFKD', str(value)).casefold().replace('&', ' and ')
    value = ''.join(c for c in value if not unicodedata.combining(c))
    return ' '.join(re.findall(r'[^\W_]+', value))


def normalize_doi(value):
    if value is None or pd.isna(value):
        return ''
    return re.sub(r'^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)', '', str(value).strip(), flags=re.I).lower()


def citation_selection(output_dir, exclusion_file, bibliography_files=()):
    """Load the same manifest/hash filter as clustering and add conservative metadata.

    Bibliography lookup is exact after normalization, never fuzzy. Conflicting DOIs
    are left blank for review. Extracted body references are not mined for a DOI,
    since they could identify another paper.
    """
    output_dir = Path(output_dir)
    manifest = read_json(output_dir / 'embeddings' / 'manifest.json')
    papers = pd.DataFrame([{'Key': r['pdf_sha256'], 'paper_id': f'P{i + 1:03d}',
        'extracted_title': r['title'], 'source_filename': r['source_filename']}
        for i, r in enumerate(manifest['rows'])])
    with np.load(output_dir / 'embeddings' / 'embeddings.npz', allow_pickle=False) as saved:
        if papers.Key.tolist() != saved['pdf_sha256'].tolist():
            raise ValueError('Embedding manifest and archive paper order differ')
        selected, _, audit, selection = filter_analysis_corpus(papers, saved['embeddings'], exclusion_file)
    bibliography = {}
    for path in bibliography_files:
        table = pd.read_csv(path, dtype=str).fillna('')
        for _, row in table.iterrows():
            title = row.get('Title', '')
            bibliography.setdefault(normalize_title(title), []).append({
                'title': title, 'doi': normalize_doi(row.get('DOI', '')),
                'year': row.get('Publication Year', row.get('Year', '')), 'source': Path(path).name})
    records = []
    for _, row in selected.iterrows():
        stem = Path(row.source_filename).stem
        split = re.split(r'\s+-\s+((?:19|20)\d{2})\s+-\s+', stem, maxsplit=1)
        title, year = (split[2], split[1]) if len(split) == 3 else (row.extracted_title, '')
        candidates = bibliography.get(normalize_title(title), [])
        if not candidates and len(normalize_title(row.extracted_title).split()) >= 4:
            candidates = bibliography.get(normalize_title(row.extracted_title), [])
        dois = sorted({c['doi'] for c in candidates if c['doi']})
        years = sorted({c['year'] for c in candidates if c['year']})
        records.append({**row.to_dict(), 'query_title': candidates[0]['title'] if candidates else title,
            'query_year': year or (years[0] if len(years) == 1 else ''),
            'doi': dois[0] if len(dois) == 1 else '',
            'metadata_status': 'bibliography_conflicting_dois' if len(dois) > 1 else
                'bibliography_exact_title' if candidates else 'filename_or_extracted_title',
            'bibliography_sources': '; '.join(sorted({c['source'] for c in candidates}))})
    return pd.DataFrame(records), audit, selection


def resolve_papers(client, papers, output_dir, *, overrides=None, approve_papers=(), auto_accept=True):
    """Reviewable proposed matches; overrides use stable Pxxx IDs or PDF hashes.

    An override is an explicit Semantic Scholar identifier/DOI, or None to leave
    a paper unmatched. Manual approvals refer to the current suggested match.
    Strong automatic matches require exact normalized titles plus either matching
    DOI or exact year with no similarly scored competing search candidate.
    """
    output_dir = Path(output_dir)
    overrides, approvals = overrides or {}, set(approve_papers)
    if papers.empty:
        raise ValueError('The current selection contains no papers')
    valid = set(papers.paper_id) | set(papers.Key)
    if (set(overrides) | approvals) - valid:
        raise ValueError('Unknown paper ID/hash in overrides or approvals')
    rows, all_candidates = [], []
    for index, (_, paper) in enumerate(papers.iterrows(), 1):
        print(f'[{index}/{len(papers)}] Match {paper.paper_id}: {paper.query_title}', flush=True)
        identifiers = [key for key in (paper.Key, paper.paper_id) if key in overrides]
        if len(identifiers) > 1:
            raise ValueError(f'Duplicate override selectors for {paper.paper_id}')
        manual = bool(identifiers)
        identifier = overrides[identifiers[0]] if manual else None
        base = {**paper.to_dict(), 's2_id': '', 's2_title': '', 's2_year': '', 's2_doi': '',
                's2_url': '', 'title_similarity': 0., 'approved': False, 'match_status': 'unmatched'}
        try:
            candidates, method = [], 'title_search'
            if manual and identifier is None:
                base['match_status'] = 'manually_unmatched'
            else:
                if manual or paper.doi:
                    candidate = client.paper(identifier if manual else f'DOI:{paper.doi}')
                    if candidate:
                        candidates = [candidate]
                        method = 'manual_identifier' if manual else 'bibliography_doi'
                if not candidates and not manual:
                    candidates = client.search(paper.query_title)
                ranked = []
                for candidate in candidates:
                    if not candidate or not candidate.get('paperId'):
                        continue
                    score = SequenceMatcher(None, normalize_title(paper.query_title),
                                            normalize_title(candidate.get('title', ''))).ratio()
                    ranked.append((score, candidate))
                ranked.sort(key=lambda r: (-r[0], r[1]['paperId']))
                for rank, (score, c) in enumerate(ranked, 1):
                    all_candidates.append({'paper_id': paper.paper_id, 'Key': paper.Key, 'rank': rank,
                        'query_title': paper.query_title, 'query_year': paper.query_year,
                        's2_id': c['paperId'], 's2_title': c.get('title', ''), 's2_year': c.get('year'),
                        's2_doi': (c.get('externalIds') or {}).get('DOI', ''),
                        'authors': '; '.join(a.get('name', '') for a in (c.get('authors') or [])),
                        'score': score, 'method': method, 'url': c.get('url', '')})
                if ranked:
                    score, chosen = ranked[0]
                    gap = score - ranked[1][0] if len(ranked) > 1 else 1.
                    doi = normalize_doi((chosen.get('externalIds') or {}).get('DOI', ''))
                    exact = normalize_title(paper.query_title) == normalize_title(chosen.get('title', ''))
                    same_year = str(chosen.get('year', '')) == str(paper.query_year) and bool(paper.query_year)
                    doi_match = bool(paper.doi) and doi == paper.doi
                    doi_conflict = bool(paper.doi and doi and paper.doi != doi)
                    strong = exact and len(normalize_title(paper.query_title).split()) >= 4 and (
                        doi_match or (same_year and gap >= .05 and not doi_conflict))
                    approved = manual or paper.Key in approvals or paper.paper_id in approvals or (auto_accept and strong)
                    status = 'manual_identifier' if manual else 'manually_approved' if (
                        paper.Key in approvals or paper.paper_id in approvals) else 'auto_strict_match' if approved else 'needs_review'
                    base.update(s2_id=chosen['paperId'], s2_title=chosen.get('title', ''),
                        s2_year=chosen.get('year'), s2_doi=doi, s2_url=chosen.get('url', ''),
                        title_similarity=score, approved=bool(approved), match_status=status,
                        reference_count=chosen.get('referenceCount'), citation_count=chosen.get('citationCount'))
                elif manual:
                    base['match_status'] = 'manual_identifier_not_found'
        except ScholarError as error:
            if error.status in (0, 401, 403, 429) or error.status >= 500:
                raise
            base['match_status'] = f'lookup_http_{error.status}'
        rows.append(base)
        # Checkpoints allow inspection even if a later API call is interrupted.
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output_dir / 'matches.csv', index=False)
        pd.DataFrame(all_candidates).to_csv(output_dir / 'match_candidates.csv', index=False)
    matches = pd.DataFrame(rows)
    approved = matches.approved & matches.s2_id.ne('')
    duplicate = matches.loc[approved, 's2_id'].duplicated(keep=False)
    conflict_ids = matches.loc[approved].loc[duplicate, 's2_id']
    conflict = matches.s2_id.isin(conflict_ids)
    matches.loc[conflict, 'approved'] = False
    matches.loc[conflict, 'match_status'] = 'duplicate_s2_identity_review_required'
    matches.to_csv(output_dir / 'matches.csv', index=False)
    return matches


def retrieve_references(client, papers, matches, output_dir):
    """Retrieve outgoing references of approved papers; retain failures as missing data."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if matches.Key.tolist() != papers.Key.tolist():
        raise ValueError('Matches must cover the current selection in its original order; rerun matching')
    active = matches.loc[matches.approved.eq(True)]
    if active.s2_id.duplicated().any() or active.s2_id.eq('').any():
        raise ValueError('Resolve duplicate or empty approved Semantic Scholar IDs first')
    records, observations = [], []
    for index, (_, paper) in enumerate(active.iterrows(), 1):
        print(f'[{index}/{len(active)}] References for {paper.paper_id}: {paper.s2_title}', flush=True)
        count, unresolved, pages = 0, 0, 0
        status = 'complete_api_pagination'
        error_message = ''
        try:
            for page in client.references(paper.s2_id):
                pages += 1
                for reference in page:
                    count += 1
                    cited = reference.get('citedPaper') or {}
                    if not cited.get('paperId'):
                        unresolved += 1
                        continue
                    observations.append({'source_s2_id': paper.s2_id, 'target_s2_id': cited['paperId'],
                        'cited_title': cited.get('title', ''), 'cited_year': cited.get('year'),
                        'cited_doi': (cited.get('externalIds') or {}).get('DOI', '')})
        except ScholarError as error:
            if error.status in (401, 403, 429) or error.status >= 500:
                raise
            status = f'incomplete_http_{error.status}'
            error_message = str(error)
        records.append({'Key': paper.Key, 's2_id': paper.s2_id, 'paper_id': paper.paper_id,
            'status': status, 'pages': pages, 'reference_records': count, 'unresolved_reference_ids': unresolved,
            'reported_reference_count': paper.get('reference_count'),
            'error_message': error_message,
            'reported_count_differs': pd.notna(paper.get('reference_count')) and count != paper.get('reference_count')})
        pd.DataFrame(records).to_csv(output_dir / 'retrieval_status.csv', index=False)
        pd.DataFrame(observations, columns=['source_s2_id', 'target_s2_id', 'cited_title', 'cited_year', 'cited_doi']).to_csv(
            output_dir / 'all_resolved_references.csv', index=False)
    return pd.DataFrame(records, columns=['Key', 's2_id', 'paper_id', 'status', 'pages',
                        'reference_records', 'unresolved_reference_ids', 'reported_reference_count',
                        'reported_count_differs', 'error_message']), pd.DataFrame(observations,
                        columns=['source_s2_id', 'target_s2_id', 'cited_title', 'cited_year', 'cited_doi'])


def export_citation_graph(papers, matches, status, references, output_dir, selection, client):
    """A -> B means A cites B. Include every selected PDF node, even unmatched ones."""
    import networkx as nx
    output_dir = Path(output_dir)
    if matches.Key.tolist() != papers.Key.tolist():
        raise ValueError('Matches do not describe the current selection in order')
    active = matches.loc[matches.approved.eq(True)]
    if active.s2_id.duplicated().any():
        raise ValueError('Duplicate approved identities cannot define distinct paper nodes')
    by_s2 = active.set_index('s2_id')['Key'].to_dict()
    # References may identify the published edition of an approved working paper.
    # Use exact DOI aliases only when they identify one approved selected PDF.
    doi_keys = {}
    for _, row in active.iterrows():
        for column in ('doi', 's2_doi'):
            doi = normalize_doi(row.get(column, '') or '')
            if doi:
                doi_keys.setdefault(doi, set()).add(row.Key)
    by_doi = {doi: next(iter(keys)) for doi, keys in doi_keys.items() if len(keys) == 1}
    nodes = matches.copy()
    nodes = nodes.merge(status[['Key', 'status']], on='Key', how='left', validate='one_to_one')
    nodes['status'] = nodes['status'].fillna('not_retrieved_unmatched_or_unapproved')
    pair_methods, conflicts = {}, []
    for _, row in references.iterrows():
        source = by_s2.get(row.source_s2_id)
        target_id = by_s2.get(row.target_s2_id)
        target_doi = by_doi.get(normalize_doi(row.get('cited_doi', '') or ''))
        if target_id and target_doi and target_id != target_doi:
            conflicts.append(row.to_dict())
            continue
        target = target_id or target_doi
        if source and target and source != target:
            pair_methods.setdefault((source, target), set()).add('s2_id' if target_id else 'exact_unique_doi')
    pairs = sorted(pair_methods)
    id_by_key = papers.set_index('Key')['paper_id'].to_dict()
    edges = pd.DataFrame([{'source_key': a, 'target_key': b, 'source_paper': id_by_key[a],
        'target_paper': id_by_key[b], 'relation': 'cites',
        'target_match_method': ';'.join(sorted(pair_methods[(a, b)]))} for a, b in pairs],
        columns=['source_key', 'target_key', 'source_paper', 'target_paper', 'relation', 'target_match_method'])
    graph = nx.DiGraph()
    for _, row in nodes.iterrows():
        graph.add_node(row.Key, paper_id=row.paper_id, title=str(row.query_title),
                       semantic_scholar_id=str(row.s2_id), match_status=row.match_status,
                       retrieval_status=row.status,
                       review_version_status=str(row.get('review_version_status', '')))
    graph.add_edges_from(pairs)
    nodes['citations_received_within_selection'] = nodes.Key.map(dict(graph.in_degree()))
    nodes['references_within_selection'] = nodes.Key.map(dict(graph.out_degree()))
    output_dir.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(output_dir / 'nodes.csv', index=False)
    edges.to_csv(output_dir / 'edges.csv', index=False)
    pd.DataFrame(conflicts, columns=references.columns).to_csv(output_dir / 'reference_identity_conflicts.csv', index=False)
    nx.write_graphml(graph, output_dir / 'citation_graph.graphml')
    from scipy.sparse import csr_matrix, save_npz
    positions = {key: i for i, key in enumerate(papers.Key)}
    matrix = csr_matrix((np.ones(len(pairs), dtype=np.int8),
        ([positions[a] for a, _ in pairs], [positions[b] for _, b in pairs])), shape=(len(papers), len(papers)))
    save_npz(output_dir / 'adjacency.npz', matrix)
    papers[['Key', 'paper_id']].to_csv(output_dir / 'adjacency_order.csv', index=False)
    summary = {'selected_papers': len(papers), 'approved_matches': len(active),
        'unmatched_or_unapproved': len(papers) - len(active), 'internal_citation_edges': len(edges),
        'complete_api_reference_lists': int(status.status.eq('complete_api_pagination').sum()),
        'incomplete_api_reference_lists': int(status.status.ne('complete_api_pagination').sum()),
        'edges_using_only_doi_alias': sum(methods == {'exact_unique_doi'} for methods in pair_methods.values()),
        'reference_identity_conflicts': len(conflicts),
        'reference_records_without_id': int(status.unresolved_reference_ids.sum()),
        'isolated_nodes_in_observed_graph': len(list(nx.isolates(graph))),
        'weak_components_in_observed_graph': nx.number_weakly_connected_components(graph)}
    if 'review_version_status' in active:
        summary['approved_version_proxies'] = int(active.review_version_status.eq('same_work_version_not_verified').sum())
    write_json(output_dir / 'graph_summary.json', summary)
    snapshots = [{'file': str(p), 'sha256': file_hash(p), 'fetched_at_utc': read_json(p)['fetched_at_utc']}
                 for p in sorted(client.used_cache_files)]
    write_json(output_dir / 'run_manifest.json', {'generated_utc': datetime.now(timezone.utc).isoformat(),
        'provider': 'Semantic Scholar Academic Graph API', 'selection': selection, 'summary': summary,
        'edge_direction': 'citing paper -> cited paper', 'match_snapshot_id': fingerprint(matches.fillna('').to_dict('records')),
        'review_config_id': matches.attrs.get('review_config_id'),
        'versions': {name: version(name) for name in ('requests', 'networkx', 'numpy', 'pandas', 'scipy')},
        'code_sha256': {Path(__file__).name: file_hash(__file__),
                        'semantic_scholar.py': file_hash(Path(__file__).with_name('semantic_scholar.py'))},
        'output_sha256': {name: file_hash(output_dir / name) for name in
            ('nodes.csv', 'edges.csv', 'citation_graph.graphml', 'adjacency.npz', 'adjacency_order.csv')},
        'api_snapshots': snapshots, 'limitation': 'Observed API graph, not proof of complete citation coverage; unmatched nodes and missing reference IDs remain explicit.'})
    return nodes, edges, graph, summary
