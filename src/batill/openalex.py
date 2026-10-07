"""Batch DOI resolution and an auditable induced citation graph from OpenAlex.

Each source uses one work record's reference list. Alternative, verified DOI
versions are target aliases only: their reference lists are never pooled.
"""
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
import time

import networkx as nx
import pandas as pd
import requests
from scipy import sparse

from .citation_graph import citation_selection, normalize_doi, normalize_title
from .storage import fingerprint, read_json, write_json, file_hash

BASE_URL = "https://api.openalex.org/works"
FIELDS = "id,doi,display_name,publication_year,authorships,referenced_works"


class OpenAlexClient:
    """Cache successful queries, omit credentials from disk, stop promptly on 429.

    A limited keyless budget is sufficient for a small batch run. An optional key
    can be provided in memory if that budget is exhausted. No long retry loop.
    """
    def __init__(self, cache_dir, api_key=None, *, session=None):
        self.cache_dir = Path(cache_dir)
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "batill-thesis-citation-graph/1.0"})
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
        self.used_cache_files = set()
        self.network_requests = 0
        self.ambiguous_dois = set()

    def get(self, params):
        request = {"url": BASE_URL, "params": params}
        path = self.cache_dir / (fingerprint(request) + ".json")
        if path.exists():
            saved = read_json(path)
            if saved["request"] != request:
                raise ValueError("OpenAlex cache identity mismatch")
            payload = saved["payload"]
        else:
            try:
                response = self.session.get(BASE_URL, params=params, timeout=60)
            except requests.RequestException:
                raise RuntimeError("OpenAlex connection failed; rerun to reuse completed batches") from None
            self.network_requests += 1
            if response.status_code != 200:
                raise RuntimeError(f"OpenAlex HTTP {response.status_code}; completed batches are cached. "
                                   "For 401/403/429, supply an OpenAlex key or retry later.")
            payload = response.json()
            if not isinstance(payload.get("results"), list):
                raise ValueError("OpenAlex returned a malformed results list")
            write_json(path, {"request": request, "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                              "payload": payload})
        self.used_cache_files.add(path)
        return payload

    def by_dois(self, dois):
        dois = sorted({normalize_doi(d) for d in dois if normalize_doi(d)})
        records = {}
        for start in range(0, len(dois), 100):
            batch = dois[start:start + 100]
            params = {"filter": "doi:" + "|".join("https://doi.org/" + d for d in batch),
                      "per_page": 100, "select": FIELDS}
            payload = self.get(params)
            results = list(payload["results"])
            page = 1
            while len(results) < payload.get("meta", {}).get("count", len(results)):
                page += 1
                payload = self.get({**params, "page": page})
                if not payload["results"]:
                    raise ValueError("OpenAlex returned an incomplete DOI batch")
                results.extend(payload["results"])
            for work in results:
                doi = normalize_doi(work.get("doi"))
                if doi in records and records[doi]["id"] != work["id"]:
                    self.ambiguous_dois.add(doi)
                records[doi] = work
            print(f"DOI batch: {min(start + 100, len(dois))}/{len(dois)} identifiers", flush=True)
        return {doi: work for doi, work in records.items() if doi not in self.ambiguous_dois}

    def search(self, title):
        # Question marks in article titles are interpreted as query wildcards.
        return self.get({"search": title.replace("?", " ").replace("*", " "),
                         "per_page": 5, "select": FIELDS})["results"]


def load_inputs(root):
    """Local bibliography first; existing approved Semantic Scholar DOIs optional.

    The current selection is always reloaded. Reviewed metadata are reused only
    when their PDF order and query metadata still agree with that selection.
    """
    root = Path(root)
    papers, exclusions, selection = citation_selection(
        root / "outputs/corpus_sections", root / "configs/analysis_exclusions.json",
        sorted((root / "data/bibliography").glob("*.csv")))
    folder = root / "outputs/citation_graph" / selection["selection_id"][:16]
    papers["approved_version_doi"] = ""
    papers["approved_version_title"] = ""
    raw_path = folder / "matches.csv"
    review_path = root / "configs/citation_match_review.json"
    if raw_path.exists() and review_path.exists():
        from .citation_review import apply_match_review
        raw = pd.read_csv(raw_path).fillna("")
        for column in ("Key", "query_title", "doi"):
            if raw[column].tolist() != papers[column].tolist():
                raise ValueError("Saved Semantic Scholar metadata differ from this selection")
        reviewed, _ = apply_match_review(raw, review_path)
        for column, target in (("s2_doi", "approved_version_doi"), ("s2_title", "approved_version_title")):
            papers[target] = reviewed[column].where(reviewed.approved, "").values
    return papers, exclusions, selection


def resolve_works(client, papers, output_dir, *, overrides=None, search_missing=True):
    """Resolve exact trusted DOIs, then conservative title searches for gaps.

    Overrides are PDF-hash keyed, manually verified DOIs. Low title similarity
    remains a review item for ordinary DOI matches. Search accepts only a unique
    exact normalized title of >=4 words, exact known year, and no competing
    candidate within 0.05 title similarity. Duplicate work identities are held.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overrides = overrides or {}
    unknown = set(overrides) - set(papers.Key)
    if unknown:
        raise ValueError("Override references PDF hashes outside the current selection")
    requests_by_key = {}
    for row in papers.to_dict("records"):
        candidates = [("verified_override", overrides.get(row["Key"], "")),
                      ("bibliography_doi", row.get("doi", "")),
                      ("approved_version_doi", row.get("approved_version_doi", ""))]
        requests_by_key[row["Key"]] = [(kind, normalize_doi(doi)) for kind, doi in candidates if doi]
    works = client.by_dois([doi for values in requests_by_key.values() for _, doi in values])
    write_json(output_dir / "ambiguous_provider_dois.json", sorted(getattr(client, "ambiguous_dois", set())))
    records, candidates_log, source_works, alias_rows = [], [], {}, []
    for row in papers.to_dict("records"):
        chosen, method, score = None, "unmatched", 0.0
        available = [(kind, works[doi]) for kind, doi in requests_by_key[row["Key"]] if doi in works]
        if available:
            method, chosen = available[0]
            titles = [row["query_title"]]
            if method == "approved_version_doi":
                titles.append(row.get("approved_version_title", ""))
            score = max(SequenceMatcher(None, normalize_title(t), normalize_title(chosen["display_name"])).ratio()
                        for t in titles if t)
            accepted = method == "verified_override" or score >= 0.8
            status = "accepted_doi" if accepted else "doi_title_needs_review"
        elif search_missing:
            found = client.search(row["query_title"])
            ranked = sorted([(SequenceMatcher(None, normalize_title(row["query_title"]),
                       normalize_title(w.get("display_name", ""))).ratio(), w) for w in found],
                       key=lambda pair: (-pair[0], pair[1]["id"]))
            for rank, (similarity, work) in enumerate(ranked, 1):
                candidates_log.append({"paper_id": row["paper_id"], "Key": row["Key"], "rank": rank,
                    "title_similarity": similarity, "openalex_id": work["id"], "title": work["display_name"],
                    "year": work.get("publication_year"), "doi": work.get("doi"),
                    "authors": "; ".join(a["author"]["display_name"] for a in work.get("authorships", []))})
            if ranked:
                score, chosen = ranked[0]
                year = str(row.get("query_year", "")).removesuffix(".0")
                accepted = (score == 1 and len(normalize_title(row["query_title"]).split()) >= 4
                            and year == str(chosen.get("publication_year"))
                            and (len(ranked) == 1 or score - ranked[1][0] >= 0.05))
                method, status = "title_search", "accepted_exact_title_year" if accepted else "search_needs_review"
            else:
                accepted, status = False, "not_found"
        else:
            accepted, status = False, "no_resolved_doi"
        refs = chosen.get("referenced_works") if chosen else None
        record = {**row, "openalex_id": chosen["id"] if chosen else "",
                  "openalex_title": chosen["display_name"] if chosen else "",
                  "openalex_doi": normalize_doi(chosen.get("doi")) if chosen else "",
                  "openalex_year": chosen.get("publication_year") if chosen else "",
                  "match_method": method, "match_status": status, "approved": accepted,
                  "title_similarity": score, "reference_records": len(refs) if isinstance(refs, list) else None,
                  "reference_status": "available_nonempty" if isinstance(refs, list) and refs else
                                      "empty_provider_list" if isinstance(refs, list) else "missing_provider_list"}
        records.append(record)
        if chosen:
            source_works[row["Key"]] = chosen
            # DOI versions are linked only when both metadata titles are compatible;
            # a manually pinned primary override is sufficient for its own identity.
            for kind, work in available:
                titles = [row["query_title"], row.get("approved_version_title", "")]
                compatible = max(SequenceMatcher(None, normalize_title(t), normalize_title(work["display_name"])).ratio()
                                 for t in titles if t) >= .8
                if compatible or (kind == "verified_override" and work["id"] == chosen["id"]):
                    alias_rows.append({"Key": row["Key"], "openalex_id": work["id"], "method": kind})
            alias_rows.append({"Key": row["Key"], "openalex_id": chosen["id"], "method": method})
    matches = pd.DataFrame(records)
    accepted_ids = matches.loc[matches.approved, "openalex_id"]
    duplicates = accepted_ids[accepted_ids.duplicated(keep=False)]
    mask = matches.openalex_id.isin(duplicates)
    matches.loc[mask, "approved"] = False
    matches.loc[mask, "match_status"] = "duplicate_work_identity_review_required"
    aliases = pd.DataFrame(alias_rows, columns=["Key", "openalex_id", "method"]).drop_duplicates()
    matches.to_csv(output_dir / "matches.csv", index=False)
    pd.DataFrame(candidates_log).to_csv(output_dir / "search_candidates.csv", index=False)
    aliases.to_csv(output_dir / "identity_aliases.csv", index=False)
    write_json(output_dir / "resolved_works.json", source_works)
    write_json(output_dir / "matching_settings.json", {"overrides": overrides, "search_missing": search_missing,
        "doi_title_similarity_threshold": .8, "search_rule": "exact long title, exact year, gap >= .05"})
    return matches, source_works, aliases


def export_graph(matches, works, aliases, output_dir, selection, client):
    """Export directed citations plus all external reference IDs for later coupling.

    Keep every selected PDF as a node. Missing/ambiguous identities are explicit;
    an empty API list is not proof that the PDF has no references.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    active = matches.loc[matches.approved].copy()
    if active.openalex_id.duplicated().any() or active.openalex_id.eq("").any():
        raise ValueError("Approved OpenAlex identities must be nonempty and unique")
    candidates = aliases.loc[aliases.Key.isin(active.Key), ["Key", "openalex_id"]].drop_duplicates()
    groups = candidates.groupby("openalex_id").Key.agg(list)
    # Reserve identities held for duplicate review; do not assign to another alias.
    reserved = set(matches.loc[~matches.approved, "openalex_id"]) - set(active.openalex_id)
    lookup = {identifier: keys[0] for identifier, keys in groups.items() if len(keys) == 1 and identifier not in reserved}
    conflicts = candidates.loc[candidates.openalex_id.isin([i for i, keys in groups.items() if len(keys) > 1])]
    conflicts.to_csv(output_dir / "ambiguous_aliases.csv", index=False)
    paper_ids = matches.set_index("Key").paper_id.to_dict()
    primary = active.set_index("Key").openalex_id.to_dict()
    references, edges = [], []
    for row in active.to_dict("records"):
        work = works[row["Key"]]
        if work["id"] != row["openalex_id"]:
            raise ValueError("Source work differs from approved match")
        refs = work.get("referenced_works")
        if not isinstance(refs, list):
            continue
        for target in sorted(set(refs)):
            references.append({"source_key": row["Key"], "source_paper_id": row["paper_id"],
                               "source_openalex_id": work["id"], "target_openalex_id": target})
            key = lookup.get(target)
            if key and key != row["Key"]:
                edges.append({"source_key": row["Key"], "target_key": key,
                    "source_paper_id": row["paper_id"], "target_paper_id": paper_ids[key],
                    "target_match_method": "primary_work_id" if primary[key] == target else "verified_doi_version_alias"})
    references = pd.DataFrame(references, columns=["source_key", "source_paper_id", "source_openalex_id", "target_openalex_id"])
    edges = pd.DataFrame(edges, columns=["source_key", "target_key", "source_paper_id", "target_paper_id", "target_match_method"])
    edges = edges.drop_duplicates(["source_key", "target_key"])
    graph = nx.DiGraph()
    for row in matches.to_dict("records"):
        graph.add_node(row["Key"], paper_id=row["paper_id"], title=row["query_title"],
                       openalex_id=row["openalex_id"], approved=bool(row["approved"]),
                       match_status=row["match_status"], reference_status=row["reference_status"])
    graph.add_edges_from(zip(edges.source_key, edges.target_key))
    nodes = matches.copy()
    nodes["internal_in_degree"] = nodes.Key.map(dict(graph.in_degree()))
    nodes["internal_out_degree"] = nodes.Key.map(dict(graph.out_degree()))
    nodes.to_csv(output_dir / "nodes.csv", index=False)
    edges.to_csv(output_dir / "edges.csv", index=False)
    references.to_csv(output_dir / "all_references.csv", index=False)
    nx.write_graphml(graph, output_dir / "citation_graph.graphml")
    sparse.save_npz(output_dir / "adjacency.npz", nx.to_scipy_sparse_array(graph, nodelist=nodes.Key.tolist(), dtype="int8", format="csr"))
    nodes[["Key", "paper_id"]].to_csv(output_dir / "adjacency_order.csv", index=False)
    components = sorted((len(c) for c in nx.weakly_connected_components(graph)), reverse=True)
    summary = {"provider": "OpenAlex", "selected_pdfs": len(nodes), "approved_work_identities": len(active),
               "unapproved_pdfs": len(nodes) - len(active), "approved_nonempty_reference_lists": int(active.reference_status.eq("available_nonempty").sum()),
               "approved_empty_reference_lists": int(active.reference_status.eq("empty_provider_list").sum()),
               "approved_missing_reference_lists": int(active.reference_status.eq("missing_provider_list").sum()),
               "internal_citation_edges": len(edges), "all_reference_links": len(references),
               "edges_using_version_alias": int(edges.target_match_method.eq("verified_doi_version_alias").sum()),
               "isolated_pdfs": len(list(nx.isolates(graph))), "weak_component_sizes": components,
               "network_requests_this_client": client.network_requests}
    write_json(output_dir / "graph_summary.json", summary)
    write_json(output_dir / "selection.json", selection)
    write_json(output_dir / "run_manifest.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(),
               "selection_id": selection["selection_id"], "provider": BASE_URL,
               "input_metadata_fingerprint": fingerprint(matches.fillna("").to_dict("records")),
               "code_sha256": file_hash(__file__),
               "matching_settings_sha256": file_hash(output_dir / "matching_settings.json"),
               "cache_files": [{"path": str(p), "sha256": file_hash(p)} for p in sorted(client.used_cache_files)],
               "direction": "row/source cites column/target", "summary": summary})
    return nodes, edges, graph, summary


def run(root, api_key=None):
    """Command-line convenience; the notebook exposes each individual stage."""
    root = Path(root).resolve()
    start = time.monotonic()
    papers, exclusions, selection = load_inputs(root)
    output = root / "outputs/citation_graph_openalex" / selection["selection_id"][:16]
    output.mkdir(parents=True, exist_ok=True)
    papers.to_csv(output / "selected_papers.csv", index=False)
    exclusions.to_csv(output / "exclusion_audit.csv", index=False)
    client = OpenAlexClient(root / "outputs/citation_graph_openalex/api_cache", api_key)
    override_file = root / "configs/openalex_doi_overrides.json"
    overrides = read_json(override_file)["overrides"] if override_file.exists() else {}
    # DOI evidence is keyed by immutable PDF hash; selection changes do not apply it elsewhere.
    overrides = {key: value["doi"] for key, value in overrides.items() if key in set(papers.Key)}
    matches, works, aliases = resolve_works(client, papers, output, overrides=overrides)
    result = export_graph(matches, works, aliases, output, selection, client)
    print(result[3])
    print(f"Completed in {time.monotonic() - start:.1f}s. Results: {output}")
    return result


if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    run(args.root, os.environ.get("OPENALEX_API_KEY"))
