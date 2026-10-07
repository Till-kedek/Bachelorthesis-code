"""Check the revised evidence and prove that version-one evidence is unchanged."""

from pathlib import Path
from itertools import combinations
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/thesis_v2/evidence"


def verify():
    manifest = json.loads((OUT / "run_manifest_v2.json").read_text())
    for path, digest in manifest["original_evidence_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    assert (
        hashlib.sha256(
            (ROOT / "scripts/thesis_v2_evidence.py").read_bytes()
        ).hexdigest()
        == manifest["source_code_sha256"]
    )
    main = pd.read_csv(OUT / "expression_rankings_all_solutions.csv").query(
        "solution == 'kmeans_k3' and ranking == 'tfidf_contrast' and phrase_rank <= 3"
    )
    supporting = pd.read_csv(OUT / "main_expression_support_papers.csv")
    for _, r in main.iterrows():
        evidence = supporting[
            (supporting.cluster == r.cluster) & (supporting.term == r.term)
        ]
        assert evidence.paper_id.is_unique
        assert evidence.inside.sum() == r.inside_count
        assert (~evidence.inside).sum() == r.outside_count
    runs = json.loads((OUT / "additional_experiment_runs.json").read_text())
    saved = pd.read_csv(OUT / "leiden_subsample_stability.csv")
    subset_counts = []
    for k in range(2, 7):
        selected = []
        for r in [r for r in runs if r["experiment"] == "subsample" and r["k"] == k]:
            best = max(r["runs"], key=lambda v: v["quality"])
            selected.append(pd.Series(best["labels"], index=r["indices"]))
            subset_counts.append(
                dict(
                    k=k,
                    subset_seed=r["subset_seed"],
                    resulting_k=len(set(best["labels"])),
                )
            )
        assert len(selected) == 5
        for a, b in combinations(range(5), 2):
            shared = selected[a].index.intersection(selected[b].index)
            expected = adjusted_rand_score(selected[a][shared], selected[b][shared])
            row = saved[(saved.k == k) & (saved.left == a) & (saved.right == b)].iloc[0]
            assert abs(row.ARI - expected) < 1e-12
            assert row.shared == len(shared)
    pd.DataFrame(subset_counts).to_csv(OUT / "leiden_subset_counts.csv", index=False)
    selector = pd.read_csv(OUT / "selection_extended.csv")
    for col in ["kmeans_subsample_ARI", "leiden_subsample_ARI", "kmeans_silhouette"]:
        assert int(selector.loc[selector[col].idxmax(), "k"]) == 3
    matched = pd.read_csv(OUT / "matched_cluster_diagnostics.csv").query("k == 3")
    assert matched.common.sum() == 274
    assert matched.kmeans_n.sum() == 292 and matched.leiden_n.sum() == 292
    assert matched.iloc[-1].common == 29
    table = pd.read_csv(OUT / "overlaps/citation_r1__kmeans_k3.csv", index_col=0)
    assert table.to_numpy().sum() == 289
    assert (
        table.loc["C4", "1"] == 87
        and table.loc["C5", "3"] == 24
        and table.loc["C6", "3"] == 3
    )
    text = (OUT.parent / "thesis_sections.md").read_text()
    method = text.split("# 4 Clusters")[0]
    assert "healthcare" not in method.lower()
    assert text.index("{{TABLE:expressions}}") < text.lower().index("healthcare")
    result = dict(
        status="passed",
        original_evidence_files_unchanged=len(manifest["original_evidence_sha256"]),
        checks=[
            "original evidence SHA-256",
            "new analysis source SHA-256",
            "expression supporting paper counts",
            "Leiden subset ARI independently recalculated",
            "main-count selection criteria",
            "matched cluster totals",
            "citation contingency totals",
            "theme evidence precedes healthcare naming",
        ],
        limitations="Consistency checks do not validate topic names or citation completeness.",
    )
    (OUT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    verify()
