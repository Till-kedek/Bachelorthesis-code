"""Independent consistency checks for the frozen 292-work chapter evidence.

These assertions prevent accidental use of old-corpus tables or stale prose.
They do not validate semantic truth or completeness of citation retrieval.
"""

from pathlib import Path
from itertools import combinations
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/thesis/evidence"


def verify():
    manifest = json.loads((OUT / "run_manifest.json").read_text())
    for path, expected in manifest["inputs"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    for path, expected in manifest["code"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    papers = pd.read_csv(OUT / "corpus.csv")
    mapping = pd.read_csv(OUT / "pdf_to_work.csv")
    assert len(papers) == 292 and papers.Key.is_unique
    assert len(mapping) == 294 and mapping.canonical_key.nunique() == 292
    assert set(papers.Key) == set(mapping.canonical_key)
    assert (pd.read_csv(OUT / "scope_exclusions.csv").status == "excluded").sum() == 24
    cm = pd.read_csv(OUT / "citation_membership_r1.csv")
    assert cm.Key.tolist() == papers.Key.tolist()
    assert (cm.citation_label == 0).sum() == 3
    counts = cm.query("citation_label > 0").groupby("citation_label").size()
    assert counts.to_list() == [46, 97, 29, 89, 25, 3]
    segments = pd.read_csv(OUT / "segments.csv")
    assert len(segments) == 2149
    assert segments.token_count.max() <= 32768
    members = pd.read_csv(OUT / "memberships.csv")
    for name, group in members.groupby("solution"):
        assert group.Key.tolist() == papers.Key.tolist(), name
    profiles = pd.read_csv(OUT / "labelled_profiles.csv")
    healthcare = []
    for _, row in profiles[
        profiles.proposed_label.eq("Healthcare ownership and outcomes")
    ].iterrows():
        group = members[
            (members.solution == row.solution) & (members.cluster == row.cluster)
        ]
        assert len(group) == 29
        healthcare.append(set(group.Key))
    assert all(h == healthcare[0] for h in healthcare)
    terms = pd.read_csv(OUT / "topic_terms.csv")
    assert (terms.inside_n + terms.outside_n).eq(292).all()
    assert (terms.inside_count / terms.inside_n).sub(
        terms.inside_prevalence
    ).abs().max() < 1e-12
    assert (terms.outside_count / terms.outside_n).sub(
        terms.outside_prevalence
    ).abs().max() < 1e-12
    metrics = pd.read_csv(OUT / "text_metrics.csv")
    runs = json.loads((OUT / "text_seed_memberships.json").read_text())
    for _, row in metrics[metrics.method.eq("K-means")].iterrows():
        labels = [
            r["labels"] for r in runs if r["run"].startswith(f"kmeans_k{row.k}_seed")
        ]
        assert len(labels) == 5
        stability = np.mean(
            [adjusted_rand_score(a, b) for a, b in combinations(labels, 2)]
        )
        assert abs(stability - row.seed_ARI) < 1e-12
    overlap = pd.read_csv(OUT / "overlaps/citation_r1__kmeans_k6.csv", index_col=0)
    assert overlap.to_numpy().sum() == 289
    assert overlap.sum(axis=1).to_list() == counts.to_list()
    assert (
        overlap.loc["C4", "5"] == 81
        and overlap.loc["C5", "6"] == 24
        and overlap.loc["C6", "6"] == 3
    )
    assert (
        metrics.query("method == 'K-means'")
        .sort_values("cosine_silhouette", ascending=False)
        .iloc[0]
        .k
        == 3
    )
    results = {
        "status": "passed",
        "canonical_works": 292,
        "compared_citation_works": 289,
        "checks": [
            "input and code SHA-256",
            "canonical identity joins",
            "scope counts",
            "segment counts and token limit",
            "all membership row orders",
            "healthcare membership invariance",
            "term prevalence arithmetic",
            "K-means seed ARI recalculation",
            "citation contingency margins",
            "reported broad-count criterion",
        ],
        "limitations": "Consistency checks do not establish topic-label accuracy or citation completeness.",
    }
    (OUT / "evidence_verification.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    verify()
