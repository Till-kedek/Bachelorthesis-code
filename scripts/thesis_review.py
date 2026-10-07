"""Freeze qualitative names against exact memberships and build compact tables.

Names are analyst interpretations of topic_profiles.csv and review_papers.csv;
they are not model predictions or independently validated labels.
"""

from pathlib import Path
import pandas as pd
from batill.topic_analysis import write_review_cards

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/thesis/evidence"
NAMES = {
    "kmeans_k2": [
        "Fund investment and performance",
        "Buyout transactions and ownership effects",
    ],
    "leiden_k2": [
        "General private equity research",
        "Healthcare ownership and outcomes",
    ],
    "kmeans_k3": [
        "Fund investment and performance",
        "Buyout transactions and ownership effects",
        "Healthcare ownership and outcomes",
    ],
    "leiden_k3": [
        "Fund investment and performance",
        "Buyout transactions and ownership effects",
        "Healthcare ownership and outcomes",
    ],
    "kmeans_k4": [
        "Fund investment and performance",
        "Buyout financing and transaction value",
        "Governance and portfolio company outcomes",
        "Healthcare ownership and outcomes",
    ],
    "leiden_k4": [
        "Governance and portfolio company outcomes",
        "Buyout financing and transaction value",
        "Fund investment and performance",
        "Healthcare ownership and outcomes",
    ],
    "kmeans_k5": [
        "Governance and portfolio company outcomes",
        "Buyout financing and transaction value",
        "Public versus private ownership and listing",
        "Fund investment and performance",
        "Healthcare ownership and outcomes",
    ],
    "leiden_k5": [
        "Governance and portfolio company outcomes",
        "Buyout financing and transaction value",
        "Public versus private ownership and listing",
        "Fund investment and performance",
        "Healthcare ownership and outcomes",
    ],
    "kmeans_k6": [
        "Sponsor practices and portfolio company governance",
        "Buyout financing and transaction value",
        "Public versus private financing and listing",
        "Employment and operating outcomes",
        "Fund investment and performance",
        "Healthcare ownership and outcomes",
    ],
    "leiden_k6": [
        "Sponsor practices and portfolio company governance",
        "Buyout financing and transaction value",
        "Public versus private ownership and listing",
        "Employment and operating outcomes",
        "Fund investment and performance",
        "Healthcare ownership and outcomes",
    ],
}


def build_review(approve_changed_memberships=False):
    profiles = pd.read_csv(OUT / "topic_profiles.csv")
    terms = pd.read_csv(OUT / "topic_terms.csv")
    examples = pd.read_csv(OUT / "review_papers.csv")
    proposals = []
    for _, r in profiles.iterrows():
        label = NAMES[r.solution][int(r.cluster) - 1]
        note = "Term rankings, central titles and sampled boundary/random excerpts; not exhaustive or independent annotation."
        if r.solution == "leiden_k6" and r.cluster == 1:
            note += " P258 is both central and a negative-silhouette boundary case: fund scale and sponsor governance overlap."
        proposals.append(
            dict(
                solution=r.solution,
                cluster=r.cluster,
                membership_id=r.membership_id,
                proposed_label=label,
                status="Provisional analyst interpretation",
                review_basis=note,
            )
        )
    proposals = pd.DataFrame(proposals)
    existing = OUT / "analyst_labels.csv"
    if existing.exists() and not approve_changed_memberships:
        previous = pd.read_csv(existing)
        identity = ["solution", "cluster", "membership_id"]
        if set(map(tuple, previous[identity].to_numpy())) != set(
            map(tuple, proposals[identity].to_numpy())
        ):
            raise ValueError(
                "Memberships changed. Review the evidence and labels before running with --approve-current-memberships."
            )
    proposals.to_csv(existing, index=False)
    labelled = write_review_cards(
        profiles, terms, examples, proposals, OUT / "cluster_review_cards.md"
    )
    labelled.to_csv(OUT / "labelled_profiles.csv", index=False)
    metrics = pd.read_csv(OUT / "text_metrics.csv")
    sub = pd.read_csv(OUT / "kmeans_subsample_stability.csv").groupby("k").ARI.mean()
    agreement = pd.read_csv(OUT / "text_agreement.csv").query(
        "comparison == 'same count'"
    )
    rows = []
    for k in range(2, 7):
        a = metrics[metrics.solution.eq(f"kmeans_k{k}")].iloc[0]
        b = metrics[metrics.solution.eq(f"leiden_k{k}")].iloc[0]
        cross = agreement[agreement.left.eq(f"kmeans_k{k}")].iloc[0]
        rows.append(
            dict(
                k=k,
                kmeans_silhouette=a.cosine_silhouette,
                leiden_silhouette=b.cosine_silhouette,
                kmeans_seed_ARI=a.seed_ARI,
                leiden_seed_ARI=b.seed_ARI,
                kmeans_subsample_ARI=sub[k],
                cross_method_ARI=cross.ARI,
                matched_works=cross.matched_papers,
                leiden_resolution=b.resolution,
            )
        )
    pd.DataFrame(rows).to_csv(OUT / "main_selection_table.csv", index=False)
    # Specific prevalence evidence for manuscript themes, not an added classifier.
    selected = []
    for c, word in [
        (1, "pe backed"),
        (2, "announcement"),
        (3, "going public"),
        (4, "employment"),
        (5, "lps"),
        (6, "physician"),
    ]:
        row = terms[
            (terms.solution == "kmeans_k6")
            & (terms.cluster == c)
            & (terms.ranking == "tfidf_contrast")
            & (terms.term == word)
        ].iloc[0]
        selected.append(row)
    pd.DataFrame(selected).to_csv(OUT / "illustrative_term_prevalence.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--approve-current-memberships",
        action="store_true",
        help="Use only after reviewing the new memberships and updating NAMES",
    )
    args = parser.parse_args()
    build_review(args.approve_current_memberships)
