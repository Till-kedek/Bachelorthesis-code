"""Interpret saved direct-citation memberships, without detecting communities."""
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .citation_clustering import load_citation_corpus, citation_network, citation_cluster_tables
from .pdf_topic_analysis import load_pdf_topic_corpus
from .storage import file_hash, fingerprint, read_json, write_json


GRAPH_FILES = ('nodes.csv', 'edges.csv', 'pdf_to_work.csv', 'run_manifest.json')


def citation_input_hashes(input_dir, exclusion_file, graph_dir):
    root, graph = Path(input_dir), Path(graph_dir)
    return {**{f'graph/{name}': file_hash(graph / name) for name in GRAPH_FILES},
        **{name: file_hash(root / name) for name in
           ('embeddings/embeddings.npz', 'embeddings/manifest.json', 'prepared/manifest.json')},
        'analysis_exclusions.json': file_hash(exclusion_file)}


def citation_topic_input_id(input_dir, exclusion_file, graph_dir, include_manual=True):
    return fingerprint({'inputs': citation_input_hashes(input_dir, exclusion_file, graph_dir),
                        'include_manual': bool(include_manual)})


def load_citation_topic_corpus(input_dir, exclusion_file, graph_dir, *, include_manual=True):
    papers, _, passages, audit, selection, validation = load_pdf_topic_corpus(
        input_dir, exclusion_file, deduplicate=False)
    papers['Key'], papers['Paper ID'] = papers.pdf_sha256, papers.paper_id
    nodes, edges, mapping = load_citation_corpus(graph_dir, papers, selection['selection_id'])
    works = papers.set_index('pdf_sha256', drop=False).loc[nodes.Key].reset_index(drop=True)
    old_to_new = dict(zip(works.row, range(len(works))))
    passages = passages.loc[passages.row.isin(old_to_new)].copy()
    passages['row'] = passages.row.map(old_to_new)
    works['row'] = np.arange(len(works))
    works['embedding_title'] = works.title
    works['title'] = nodes.query_title.to_numpy()  # Reviewed display titles, source text unchanged.
    works['doi'] = nodes.doi.to_numpy()
    works['year'] = nodes.query_year.to_numpy()
    network = citation_network(nodes, edges, include_manual=include_manual)
    return {'papers': works, 'passages': passages, 'nodes': nodes, 'edges': edges,
        'mapping': mapping, 'network': network, 'selection': selection,
        'validation': validation, 'exclusion_audit': audit,
        'input_hashes': citation_input_hashes(input_dir, exclusion_file, graph_dir),
        'input_id': citation_topic_input_id(input_dir, exclusion_file, graph_dir, include_manual)}


def _labels(labels, corpus):
    labels = np.asarray(labels)
    if labels.shape != (len(corpus['papers']),) or labels.dtype.kind not in 'iu' or (labels < 0).any():
        raise ValueError('Citation labels must align with all canonical works and be nonnegative integers')
    if not np.array_equal(labels == 0, corpus['network']['degree'] == 0):
        raise ValueError('Only citation isolates must have label zero (unassigned)')
    groups = sorted(set(labels) - {0})
    if len(groups) < 2 or groups != list(range(1, len(groups) + 1)):
        raise ValueError('Expected original consecutive positive citation community IDs')
    return labels


def save_citation_topic_partition(destination, corpus, labels, *, resolution, seed,
                                   fit_input_id, source_evidence):
    """Save an existing fit only; no graph optimizer or embedding clustering is called."""
    if fit_input_id != corpus['input_id']:
        raise ValueError('Stale citation fit: graph, text inputs or scope changed')
    labels = _labels(labels, corpus)
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError('Citation resolution must be positive')
    frame = corpus['papers'][['pdf_sha256', 'paper_id']].copy()
    frame['cluster'] = labels
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / 'assignments.csv', index=False)
    manifest = {'schema_version': 1, 'source_kind': 'direct_citations',
        'source_notebook': 'Clustering PDFs.ipynb', 'source_variable': 'citation_selected',
        'unit': 'canonical_work', 'graph_rule': 'binary undirected union; isolates unassigned',
        'include_manual': corpus['network']['include_manual'],
        'input_id': corpus['input_id'], 'input_hashes': corpus['input_hashes'],
        'selection_id': corpus['selection']['selection_id'], 'resolution': float(resolution),
        'seed': int(seed), 'communities': len(set(labels) - {0}), 'works': len(labels),
        'unassigned': int(np.sum(labels == 0)),
        'assignment_id': fingerprint(list(zip(frame.pdf_sha256, labels.tolist()))),
        'assignments_sha256': file_hash(destination / 'assignments.csv'),
        'source_evidence': source_evidence}
    write_json(destination / 'partition_manifest.json', manifest)
    return manifest


def import_existing_citation_partition(destination, corpus, source_dir, comparison_dir):
    """Verify historical source exports against their recorded graph and hashed labels.

    This adoption step reuses the existing work memberships and selected seed;
    the old UMAP coordinates only authenticate saved labels, never define topics.
    """
    source_dir, comparison_dir = Path(source_dir), Path(comparison_dir)
    provenance = read_json(comparison_dir / 'run_manifest.json')
    if (provenance['selection_id'] != corpus['selection']['selection_id']
            or provenance['include_manual_citations'] != corpus['network']['include_manual']):
        raise ValueError('Historical citation selection or graph variant differs')
    for name, sha in provenance['graph_files'].items():
        if corpus['input_hashes'].get(f'graph/{name}') != sha:
            raise ValueError('Historical citation graph checksum mismatch')
    if provenance['embedding_manifest_sha256'] != corpus['input_hashes']['embeddings/manifest.json']:
        raise ValueError('Historical embedding inputs differ')
    coordinates = comparison_dir / 'umap_coordinates_and_labels.csv'
    if file_hash(coordinates) != provenance['coordinates_and_labels_sha256']:
        raise ValueError('Historical citation label evidence checksum mismatch')
    keys = corpus['papers'].pdf_sha256

    def align(frame):
        if not frame.Key.is_unique or set(frame.Key) != set(keys):
            raise ValueError('Historical assignments do not cover exactly the canonical works')
        frame = frame.set_index('Key').loc[keys]
        if frame.paper_id.tolist() != corpus['papers'].paper_id.tolist():
            raise ValueError('Historical paper IDs do not match canonical works')
        return frame

    membership = align(pd.read_csv(source_dir / 'work_membership.csv'))
    recorded = align(pd.read_csv(coordinates))
    labels = _labels(membership.citation_label.to_numpy(), corpus)
    expected_names = ['Unassigned (no links)' if c == 0 else f'C{c}' for c in labels]
    if recorded['Citation community'].tolist() != expected_names:
        raise ValueError('Saved work memberships differ from authenticated source labels')
    resolution, seed = provenance['citation_resolution'], provenance['citation_selected_seed']
    runs = pd.read_csv(source_dir.parent / 'all_seed_memberships.csv')
    selected = align(runs.loc[(runs.resolution == resolution) & (runs.seed == seed)])
    if not np.array_equal(selected.citation_label.to_numpy(), labels):
        raise ValueError('Saved labels differ from the recorded selected citation seed')
    files = [source_dir / 'work_membership.csv', source_dir.parent / 'all_seed_memberships.csv',
             comparison_dir / 'run_manifest.json', coordinates]
    return save_citation_topic_partition(destination, corpus, labels, resolution=resolution,
        seed=seed, fit_input_id=corpus['input_id'], source_evidence={
            'origin': 'verified existing Clustering PDFs citation exports; no refitting',
            'files': {p.name: file_hash(p) for p in files}, 'packages': provenance['versions']})


def load_citation_topic_partition(destination, corpus):
    destination = Path(destination)
    if not all((destination / name).is_file() for name in ('assignments.csv', 'partition_manifest.json')):
        raise FileNotFoundError('Export the citation partition from Clustering PDFs first; no reclustering fallback')
    manifest = read_json(destination / 'partition_manifest.json')
    if (manifest.get('schema_version') != 1 or manifest.get('source_kind') != 'direct_citations'
            or manifest.get('unit') != 'canonical_work'
            or manifest.get('source_notebook') != 'Clustering PDFs.ipynb'
            or manifest.get('source_variable') != 'citation_selected'):
        raise ValueError('Expected the direct-citation partition, not an embedding partition')
    if manifest['input_id'] != corpus['input_id'] or manifest['input_hashes'] != corpus['input_hashes']:
        raise ValueError('Citation graph, source text or selection changed since export')
    if (manifest['selection_id'] != corpus['selection']['selection_id']
            or manifest['include_manual'] != corpus['network']['include_manual']
            or manifest['graph_rule'] != 'binary undirected union; isolates unassigned'
            or not np.isfinite(manifest['resolution']) or manifest['resolution'] <= 0
            or not isinstance(manifest['seed'], int)):
        raise ValueError('Citation graph settings or selection metadata differ')
    if file_hash(destination / 'assignments.csv') != manifest['assignments_sha256']:
        raise ValueError('Citation assignment checksum mismatch')
    frame = pd.read_csv(destination / 'assignments.csv', dtype={'pdf_sha256': str, 'paper_id': str})
    keys = corpus['papers'].pdf_sha256
    if not frame.pdf_sha256.is_unique or set(frame.pdf_sha256) != set(keys):
        raise ValueError('Citation assignments must cover every canonical work exactly once')
    frame = frame.set_index('pdf_sha256').loc[keys].reset_index()
    if frame.paper_id.tolist() != corpus['papers'].paper_id.tolist():
        raise ValueError('Citation work IDs do not match their keys')
    labels = _labels(frame.cluster.to_numpy(), corpus)
    if (manifest['assignment_id'] != fingerprint(list(zip(frame.pdf_sha256, labels.tolist())))
            or manifest['works'] != len(labels) or manifest['communities'] != len(set(labels) - {0})
            or manifest['unassigned'] != int(np.sum(labels == 0))):
        raise ValueError('Citation membership fingerprint/count mismatch')
    return labels, manifest


def describe_citation_topics(corpus, labels, terms, random_seed=42):
    """Graph representatives, external-link boundary candidates, and source excerpts."""
    labels = _labels(labels, corpus)
    papers, network = corpus['papers'], corpus['network']
    _, membership = citation_cluster_tables(corpus['nodes'], network, {'labels': labels})
    membership = membership.rename(columns={'citation_label': 'cluster'})
    membership['external_degree'] = membership.citation_degree - membership.within_community_degree
    membership['external_link_fraction'] = np.divide(membership.external_degree,
        membership.citation_degree, out=np.zeros(len(membership)), where=membership.citation_degree > 0)
    incoming = np.asarray(network['directed'].sum(axis=0)).ravel()
    profiles, examples = [], []
    for cluster in sorted(set(labels) - {0}):
        members = np.flatnonzero(labels == cluster)
        internal = membership.iloc[members].within_community_degree.to_numpy()
        central = members[np.lexsort((members, -incoming[members], -internal))[:3]]
        fractions = membership.iloc[members].external_link_fraction.to_numpy()
        # A wholly disconnected community has no external-link boundary papers.
        boundary = members[np.argsort(-fractions, kind='stable')]
        boundary = boundary[membership.iloc[boundary].external_degree.to_numpy() > 0][:2]
        rest = np.setdiff1d(members, np.union1d(central, boundary))
        random = np.random.default_rng(random_seed + int(cluster)).choice(rest, size=min(2, len(rest)), replace=False)
        subset = terms.loc[terms.cluster.eq(cluster)]
        contrast = subset.loc[subset.ranking.eq('tfidf_contrast')].sort_values('rank').term.tolist()
        ctfidf = subset.loc[subset.ranking.eq('c_tf_idf')].sort_values('rank').term.tolist()
        profiles.append({'cluster': int(cluster), 'papers': len(members),
            'membership_id': fingerprint(sorted(zip(papers.iloc[members].pdf_sha256, papers.iloc[members].text_sha256))),
            'within_community_links': int(internal.sum() // 2),
            'mean_external_link_fraction': float(fractions.mean()),
            'ctfidf_terms': '; '.join(ctfidf[:10]), 'contrast_terms': '; '.join(contrast[:10]),
            'representatives': '\n'.join(f'{papers.iloc[i].paper_id}: {papers.iloc[i].title}' for i in central)})
        roles = {}
        for role, indices in [('representative', central), ('boundary', boundary), ('random', random)]:
            for i in indices:
                roles.setdefault(int(i), []).append(role)
        for i, role in roles.items():
            options = corpus['passages'].loc[corpus['passages'].row.eq(i)]
            found, matched, begin = None, '', 0
            for term in contrast[:10]:
                pattern = r'(?<!\w)' + r'\s+'.join(map(re.escape, term.split())) + r'(?!\w)'
                for _, passage in options.iterrows():
                    match = re.search(pattern, passage.text, re.I)
                    if match:
                        found, matched, begin = passage, term, max(0, match.start() - 130)
                        break
                if found is not None:
                    break
            if found is None:
                found = options.iloc[0]
            end = min(len(found.text), begin + 600)
            examples.append({**membership.iloc[i].to_dict(), 'title': papers.iloc[i].title,
                'selection': '; '.join(role), 'term': matched, 'page': found.page,
                'block_id': found.block_id, 'source_start': found.source_start,
                'excerpt_start': begin, 'excerpt_end': end, 'excerpt': found.text[begin:end]})
    return pd.DataFrame(profiles), membership, pd.DataFrame(examples)


def write_citation_review_cards(profiles, terms, examples, proposals, destination):
    keys = ['solution', 'cluster', 'membership_id']
    if proposals.proposed_label.isna().any() or proposals.proposed_label.str.strip().eq('').any():
        raise ValueError('Citation proposals must contain nonempty names')
    columns = keys + ['proposed_label', 'status'] + [c for c in ('rationale', 'review_caveat') if c in proposals]
    joined = profiles.merge(proposals[columns], on=keys, how='left', validate='one_to_one')
    joined['proposal_matched'] = joined.proposed_label.notna()
    joined.loc[~joined.proposal_matched, 'proposed_label'] = 'Name pending review'
    joined.loc[~joined.proposal_matched, 'status'] = 'Review changed membership or source text'
    lines = ['# Citation topic review cards', '',
        'AI-assisted analyst proposals for saved citation communities. Isolates are unassigned. '
        'Names describe canonical-work source texts, not citation counts alone.', '']
    for _, row in joined.iterrows():
        lines += [f'## C{row.cluster}: {row.proposed_label}', '',
            f'{row.papers} works; {row.within_community_links} internal links; '
            f'mean external-link fraction {row.mean_external_link_fraction:.3f}. {row.status}.', '',
            f'c-TF-IDF: {row.ctfidf_terms}', '', f'TF-IDF contrast: {row.contrast_terms}', '']
        for field in ('rationale', 'review_caveat'):
            if pd.notna(row.get(field)):
                lines += [str(row[field]), '']
        subset = terms.loc[(terms.cluster == row.cluster) & terms.ranking.eq('tfidf_contrast')].head(8)
        lines += ['| Term | Works inside | Works outside |', '|---|---:|---:|']
        lines += [f'| {t.term} | {t.inside_count}/{t.inside_n} | {t.outside_count}/{t.outside_n} |'
                  for _, t in subset.iterrows()]
        lines += ['', 'Citation-central representatives:', ''] + [f'- {s}' for s in row.representatives.splitlines()]
        boundary = examples.loc[examples.cluster.eq(row.cluster) & examples.selection.str.contains('boundary')]
        lines += ['', 'Boundary candidates (largest share of links to other communities):', '']
        lines += [f'- {p.paper_id}: {p.title} ({p.external_link_fraction:.1%} external links)'
                  for _, p in boundary.iterrows()] or ['- No external citation links.']
        lines += ['', '']
    Path(destination).write_text('\n'.join(lines), encoding='utf-8')
    return joined
