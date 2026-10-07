"""Add auditable evidence for the revised thesis without changing earlier results.

Saved partitions, source text and coordinates are reused. New experiments vary
the text graph's neighbour count and refit Leiden on five 80% subsets. No PDF
extraction, embedding inference or API request is performed.
"""

from pathlib import Path
from itertools import combinations
from datetime import datetime, timezone
import hashlib
import json
import re
import shutil

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from batill.thesis_analysis import canonical_corpus, save_json
from batill.graph_clustering import similarity_graph, leiden_partition
from batill.topic_analysis import build_vocabulary, topic_terms

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "reports/thesis/evidence"
REPORT = ROOT / "reports/thesis_v2"
OUT = REPORT / "evidence"
MAIN_K = 3


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def graph_checks(x, metrics, partitions):
    neighbours, subset_records, saved = [], [], []
    for n in (10, 15, 25):
        graph = similarity_graph(x, neighbours=n)
        for k in range(3, 7):
            resolution = float(
                metrics.loc[metrics.solution.eq(f"leiden_k{k}"), "resolution"].iloc[0]
            )
            runs = [leiden_partition(graph, resolution, seed) for seed in (42, 43, 44)]
            best = max(runs, key=lambda r: r["quality"])
            neighbours.append(
                dict(
                    k=k,
                    neighbours=n,
                    resolution=resolution,
                    resulting_k=len(set(best["labels"])),
                    ARI=adjusted_rand_score(partitions[f"leiden_k{k}"], best["labels"]),
                )
            )
            saved.append(
                dict(
                    experiment="neighbours",
                    k=k,
                    neighbours=n,
                    runs=[
                        dict(seed=r["seed"], quality=r["quality"], labels=r["labels"])
                        for r in runs
                    ],
                )
            )
    # Resolution is held fixed; a subset is allowed to yield a different count.
    for k in range(2, 7):
        resolution = float(
            metrics.loc[metrics.solution.eq(f"leiden_k{k}"), "resolution"].iloc[0]
        )
        selected = []
        for seed in (101, 202, 303, 404, 505):
            indices = np.sort(
                np.random.default_rng(seed).choice(
                    len(x), int(0.8 * len(x)), replace=False
                )
            )
            graph = similarity_graph(x[indices], neighbours=15)
            runs = [leiden_partition(graph, resolution, s) for s in (42, 43, 44)]
            best = max(runs, key=lambda r: r["quality"])
            selected.append(pd.Series(best["labels"], index=indices))
            saved.append(
                dict(
                    experiment="subsample",
                    k=k,
                    subset_seed=seed,
                    indices=indices,
                    resulting_k=len(set(best["labels"])),
                    runs=[
                        dict(seed=r["seed"], quality=r["quality"], labels=r["labels"])
                        for r in runs
                    ],
                )
            )
        for a, b in combinations(range(len(selected)), 2):
            shared = selected[a].index.intersection(selected[b].index)
            subset_records.append(
                dict(
                    k=k,
                    left=a,
                    right=b,
                    shared=len(shared),
                    ARI=adjusted_rand_score(selected[a][shared], selected[b][shared]),
                )
            )
    pd.DataFrame(neighbours).to_csv(OUT / "neighbour_sensitivity.csv", index=False)
    pd.DataFrame(subset_records).to_csv(
        OUT / "leiden_subsample_stability.csv", index=False
    )
    save_json(OUT / "additional_experiment_runs.json", saved)


def lexical_evidence(papers, partitions):
    vocab = build_vocabulary(papers)
    profiles = pd.read_csv(OLD / "labelled_profiles.csv")
    frames = []
    # A phrase-only ranking is a transparent filter on the unchanged joint
    # unigram/bigram vocabulary and scores, not a new fitted topic model.
    for solution in profiles.solution.unique():
        terms = topic_terms(vocab, partitions[solution], top_n=len(vocab["terms"]))
        terms.insert(0, "solution", solution)
        terms["is_expression"] = terms.term.str.contains(" ")
        phrases = terms[terms.is_expression].copy()
        phrases["phrase_rank"] = phrases.groupby(["cluster", "ranking"]).cumcount() + 1
        frames.append(phrases[phrases.phrase_rank.le(10)])
    phrases = pd.concat(frames, ignore_index=True)
    phrases.to_csv(OUT / "expression_rankings_all_solutions.csv", index=False)
    for name in [
        "corpus.csv",
        "scope_exclusions.csv",
        "pdf_to_work.csv",
        "segments.csv",
        "corpus_statistics.json",
        "text_metrics.csv",
        "main_selection_table.csv",
        "text_leiden_sweep.csv",
        "kmeans_subsample_stability.csv",
        "text_agreement.csv",
        "memberships.csv",
        "review_papers.csv",
        "labelled_profiles.csv",
        "analyst_labels.csv",
        "topic_terms.csv",
        "citation_sweep.csv",
        "citation_profiles_r1.csv",
        "citation_membership_r1.csv",
        "citation_terms_r1.csv",
        "citation_text_agreement.csv",
        "graph_diagnostics.json",
        "projection_diagnostics.json",
        "coordinates.csv",
        "hierarchy_membership_15.csv",
        "sensitivity.csv",
        "cluster_review_cards.md",
    ]:
        shutil.copy2(OLD / name, OUT / name)
    shutil.copytree(OLD / "overlaps", OUT / "overlaps", dirs_exist_ok=True)
    shutil.copy2(OLD / "figures/hierarchy.png", OUT / "figures/hierarchy.png")
    shutil.copy2(
        OLD / "figures/selection_diagnostics.png",
        OUT / "figures/selection_diagnostics.png",
    )
    # Export the supporting paper identities for each displayed main-map phrase.
    main = phrases.query(
        "solution == @main_solution and ranking == 'tfidf_contrast' and phrase_rank <= 3",
        local_dict={"main_solution": f"kmeans_k{MAIN_K}"},
    )
    term_index = {term: j for j, term in enumerate(vocab["terms"])}
    support = []
    for _, row in main.iterrows():
        present = np.asarray(
            (vocab["counts"][:, term_index[row.term]] > 0).toarray()
        ).ravel()
        for i in np.flatnonzero(present):
            support.append(
                dict(
                    cluster=row.cluster,
                    term=row.term,
                    paper_id=papers.iloc[i].paper_id,
                    inside=bool(partitions[f"kmeans_k{MAIN_K}"][i] == row.cluster),
                    title=papers.iloc[i].title,
                )
            )
    pd.DataFrame(support).to_csv(
        OUT / "main_expression_support_papers.csv", index=False
    )
    return phrases, profiles


def match_clusters(partitions, profiles):
    rows = []
    for k in range(2, 7):
        a, b = partitions[f"kmeans_k{k}"], partitions[f"leiden_k{k}"]
        counts = pd.crosstab(a, b)
        ii, jj = linear_sum_assignment(-counts.to_numpy())
        for i, j in zip(ii, jj):
            left, right = int(counts.index[i]), int(counts.columns[j])
            common = int(counts.iloc[i, j])
            na = int((a == left).sum())
            nb = int((b == right).sum())
            pa = profiles[
                (profiles.solution == f"kmeans_k{k}") & (profiles.cluster == left)
            ].iloc[0]
            pb = profiles[
                (profiles.solution == f"leiden_k{k}") & (profiles.cluster == right)
            ].iloc[0]
            rows.append(
                dict(
                    k=k,
                    kmeans_cluster=left,
                    leiden_cluster=right,
                    kmeans_n=na,
                    leiden_n=nb,
                    common=common,
                    jaccard=common / (na + nb - common),
                    kmeans_silhouette=pa.mean_silhouette,
                    leiden_silhouette=pb.mean_silhouette,
                    kmeans_negative=pa.negative_fraction,
                    leiden_negative=pb.negative_fraction,
                )
            )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "matched_cluster_diagnostics.csv", index=False)
    return result


def figures(partitions, matched):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    coord = pd.read_csv(OUT / "coordinates.csv")
    colours = ["#2878A5", "#D98E22", "#8A559B"]

    def save(fig, name):
        for ext in ("png", "svg"):
            fig.savefig(OUT / "figures" / f"{name}.{ext}", dpi=220, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.25), layout="constrained")
    mapping = dict(
        zip(
            matched.query("k == @MAIN_K").leiden_cluster,
            matched.query("k == @MAIN_K").kmeans_cluster,
        )
    )
    for ax, method in zip(axes, ["kmeans", "leiden"]):
        labels = partitions[f"{method}_k{MAIN_K}"]
        for c in sorted(set(labels)):
            colour = colours[(c if method == "kmeans" else mapping[c]) - 1]
            mask = labels == c
            ax.scatter(
                coord.loc[mask, "UMAP1"],
                coord.loc[mask, "UMAP2"],
                s=15,
                color=colour,
                alpha=0.8,
                label=f"C{c} (n={sum(mask)})",
            )
        ax.set(
            title="K-means" if method == "kmeans" else "Text Leiden",
            xlabel="UMAP 1",
            ylabel="UMAP 2",
        )
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    save(fig, "umap_comparison")
    for method in ("kmeans", "leiden"):
        fig, axes = plt.subplots(1, 4, figsize=(12, 3), layout="constrained")
        for ax, k in zip(axes, range(3, 7)):
            labels = partitions[f"{method}_k{k}"]
            for c in sorted(set(labels)):
                mask = labels == c
                ax.scatter(
                    coord.loc[mask, "UMAP1"],
                    coord.loc[mask, "UMAP2"],
                    s=10,
                    label=f"C{c}",
                )
            ax.set(title=f"{k} clusters", xlabel="UMAP 1", ylabel="UMAP 2")
            ax.legend(fontsize=6, frameon=False, ncol=2, loc="lower right")
        save(fig, f"umap_{method}_counts")


def table_bundle(phrases, profiles, matched):
    tables = {}

    def put(key, title, note, headers, rows, widths, left=(0,)):
        tables[key] = dict(
            title=title,
            note=note,
            headers=headers,
            rows=rows,
            widths=widths,
            left=list(left),
        )

    put(
        "flow",
        "Table 1 Corpus construction",
        "Counts refer to PDFs until duplicate versions are consolidated.",
        ["Stage", "Count", "Change"],
        [
            ["Supplied PDFs", 320, "Starting folder"],
            ["Byte-unique PDFs", 319, "1 exact duplicate"],
            ["Usable text and vectors", 318, "1 empty extraction"],
            ["Selected PDFs", 294, "24 scope exclusions"],
            ["Distinct works", 292, "2 duplicate-version pairs"],
        ],
        [8, 2, 6],
        (0, 2),
    )
    st = json.loads((OUT / "corpus_statistics.json").read_text())
    pr = json.loads((OUT / "projection_diagnostics.json").read_text())
    put(
        "preparation",
        "Table 3 Representation diagnostics",
        "Section flags overlap; they do not establish extraction errors. Correlations are Spearman coefficients.",
        ["Diagnostic", "Observed value"],
        [
            [
                "Segments per work: median [range]",
                f"{st['segments_median']:g} [{st['segments_range'][0]}–{st['segments_range'][1]}]",
            ],
            [
                "Total segments / largest input",
                f"{st['segments']:,} / {st['largest_segment_tokens']:,} tokens",
            ],
            [
                "No detected introduction / uncertain heading level / no boundaries",
                " / ".join(
                    str(st["section_flags"][k])
                    for k in [
                        "introduction_not_detected",
                        "unnumbered_heading_levels_need_review",
                        "no_section_boundaries_detected",
                    ]
                )
                + " works",
            ],
            [
                "Extraction wall time / median per extracted PDF",
                f"{st['extraction_wall_seconds']/60:.1f} min / {st['extraction_median_seconds']:.2f} s",
            ],
            [
                "PC1 / PC2 explained variance",
                " / ".join(f"{v:.1%}" for v in pr["PCA_explained_variance"]),
            ],
            [
                "Segment count correlation with PC1 / PC2",
                " / ".join(
                    f"{pr[f'PC{i}_segment_count_spearman']:.3f}" for i in (1, 2)
                ),
            ],
            [
                "Content-token count correlation with PC1 / PC2",
                " / ".join(
                    f"{pr[f'PC{i}_content_tokens_spearman']:.3f}" for i in (1, 2)
                ),
            ],
            [
                "PCA / UMAP trustworthiness at 15 neighbours",
                f"{pr['PCA_trustworthiness_15']:.3f} / {pr['UMAP_trustworthiness_15']:.3f}",
            ],
        ],
        [11.5, 4.5],
    )
    put(
        "algorithms",
        "Table 2 Clustering design",
        "Text methods use the original 2,560-dimensional vectors. Projections do not determine membership.",
        ["Method", "Representation and settings", "Role"],
        [
            [
                "K-means",
                "Unit paper vectors; k=2–15; 5 seeds × 10 initialisations",
                "Main partition candidates",
            ],
            [
                "Average linkage",
                "Cosine distance; merge by average intergroup distance; inspect 15-group cut",
                "Hierarchy and unusual-paper inspection",
            ],
            [
                "Text Leiden",
                "Union 15-nearest-neighbour graph; cosine weights; γ=0.05–2.00, step 0.025; 3 seeds",
                "Alternative text partitions",
            ],
            [
                "Citation Leiden",
                "Binary undirected internal-citation graph; γ=0.05–2.50, step 0.05; 3 seeds",
                "Relational comparison; γ=1 baseline",
            ],
        ],
        [2.7, 9.6, 3.7],
        (0, 1, 2),
    )
    select = pd.read_csv(OUT / "main_selection_table.csv")
    ls = pd.read_csv(OUT / "leiden_subsample_stability.csv").groupby("k").ARI.mean()
    select["leiden_subsample_ARI"] = select.k.map(ls)
    select.to_csv(OUT / "selection_extended.csv", index=False)
    put(
        "selection",
        "Table 4 Comparison of candidate cluster counts",
        "Sil. = cosine silhouette; sub. = mean ARI across five 80% subsets; cross = K-means–Leiden ARI. Higher values indicate separation or agreement, not accuracy. Full seed and resolution results are in Appendix A.",
        [
            "k",
            "K-means\nSil.",
            "Leiden\nSil.",
            "K-means\nSub.",
            "Leiden\nSub.",
            "Cross\nARI",
        ],
        [
            [
                int(r.k),
                *[
                    f"{r[x]:.3f}"
                    for x in [
                        "kmeans_silhouette",
                        "leiden_silhouette",
                        "kmeans_subsample_ARI",
                        "leiden_subsample_ARI",
                        "cross_method_ARI",
                    ]
                ],
            ]
            for _, r in select.iterrows()
        ],
        [1, 3, 3, 3, 3, 3],
    )
    ns = pd.read_csv(OUT / "neighbour_sensitivity.csv")
    neighbour_rows = []
    for k in range(3, 7):
        values = []
        for n in (10, 25):
            r = ns[(ns.k == k) & (ns.neighbours == n)].iloc[0]
            values.append(f"{int(r.resulting_k)} / {r.ARI:.3f}")
        neighbour_rows.append([k, *values])
    put(
        "neighbours",
        "Table 5 Sensitivity to text graph construction",
        "Resolution is held at the selected 15-neighbour setting. Each entry gives resulting clusters / ARI against that setting; count is not forced.",
        ["Reference k", "10 neighbours", "25 neighbours"],
        neighbour_rows,
        [3, 6.5, 6.5],
    )
    main = phrases.query(
        "solution == 'kmeans_k3' and ranking == 'tfidf_contrast' and phrase_rank <= 3"
    )
    put(
        "expressions",
        "Table 6 Characteristic expressions before thematic naming",
        "Top three eligible bigrams per cluster, ordered by mean per-paper TF-IDF contrast. Counts are documents containing the expression at least once; they are not raw occurrence counts.",
        ["Cluster", "Expression", "Inside", "Outside"],
        [
            [
                f"C{int(r.cluster)}",
                r.term,
                f"{int(r.inside_count)}/{int(r.inside_n)}",
                f"{int(r.outside_count)}/{int(r.outside_n)}",
            ]
            for _, r in main.iterrows()
        ],
        [2, 7, 3.5, 3.5],
        (0, 1),
    )
    mainp = profiles.query("solution == 'kmeans_k3'")
    put(
        "themes",
        "Table 7 Themes supported by lexical and paper evidence",
        "Names are qualitative interpretations. Representative papers are selected by average within-cluster cosine similarity; full membership and boundary examples remain available.",
        ["Cluster and works", "Proposed theme", "Two central papers"],
        [
            [
                f"C{int(r.cluster)}\nn={int(r.papers)}",
                r.proposed_label,
                "\n".join(r.representatives.splitlines()[:2]),
            ]
            for _, r in mainp.iterrows()
        ],
        [2.3, 4.2, 9.5],
        (0, 1, 2),
    )
    put(
        "agreement",
        "Table 8 Cluster-level agreement and separation",
        "Leiden groups are matched to K-means by maximum overlap. Jaccard = shared / union. Negative silhouettes indicate a closer alternative cluster on average, not verified misclassification.",
        [
            "Matched\ncluster",
            "K-means /\nLeiden n",
            "Shared",
            "Jaccard",
            "Silhouette\nKM / L",
            "Negative %\nKM / L",
        ],
        [
            [
                f"C{int(r.kmeans_cluster)}",
                f"{int(r.kmeans_n)} / {int(r.leiden_n)}",
                int(r.common),
                f"{r.jaccard:.3f}",
                f"{r.kmeans_silhouette:.3f} / {r.leiden_silhouette:.3f}",
                f"{100*r.kmeans_negative:.1f} / {100*r.leiden_negative:.1f}",
            ]
            for _, r in matched.query("k == 3").iterrows()
        ],
        [2, 2.8, 1.6, 2, 3.8, 3.8],
    )
    citations = pd.read_csv(OUT / "citation_profiles_r1.csv")
    ct = pd.read_csv(OUT / "citation_terms_r1.csv")
    overlap = pd.read_csv(OUT / "overlaps/citation_r1__kmeans_k3.csv", index_col=0)
    put(
        "citations",
        "Table 9 Citation communities and correspondence with the text map",
        "Citation Leiden at γ=1; only 289 linked works are assigned. Expressions are the first two ranked bigrams in the saved contrast ranking. C1–C3 columns refer to the established text clusters, not citation IDs.",
        [
            "Citation\ngroup",
            "n",
            "Characteristic expressions",
            "Text\nC1",
            "Text\nC2",
            "Text\nC3",
        ],
        [
            [
                r.Community,
                int(r.Works),
                "; ".join(
                    ct.query(
                        "cluster == @c and ranking == 'tfidf_contrast'",
                        local_dict={"c": int(r.Community[1:])},
                    )
                    .loc[lambda z: z.term.str.contains(" ")]
                    .head(2)
                    .term
                ),
                *[int(v) for v in overlap.loc[r.Community]],
            ]
            for _, r in citations.iterrows()
        ],
        [2, 1.2, 7.4, 1.8, 1.8, 1.8],
        (0, 2),
    )
    # Appendix tables contain alternative counts, complete rankings and sampled
    # counterexamples; the main text therefore need not enumerate every result.
    put(
        "seed_summary",
        "Table A1 Seed stability and selected resolutions",
        "Seed stability is mean pairwise ARI. Text Leiden resolution selection used this score, so it is not held-out validation.",
        ["k", "K-means seed ARI", "Leiden seed ARI", "Leiden γ", "Matched works"],
        [
            [
                int(r.k),
                f"{r.kmeans_seed_ARI:.3f}",
                f"{r.leiden_seed_ARI:.3f}",
                r.leiden_resolution,
                f"{int(r.matched_works)}/292",
            ]
            for _, r in select.iterrows()
        ],
        [1, 4, 4, 3, 4],
    )
    full = pd.read_csv(OUT / "text_metrics.csv")
    put(
        "full_sweep",
        "Table A2 Full K-means count sweep",
        "Every tested count is shown; detailed thematic review was limited to two through six.",
        ["k", "Silhouette", "Seed ARI", "Negative %", "Smallest / largest"],
        [
            [
                int(r.k),
                f"{r.cosine_silhouette:.3f}",
                f"{r.seed_ARI:.3f}",
                f"{100*r.negative_fraction:.1f}",
                f"{int(r.smallest)} / {int(r.largest)}",
            ]
            for _, r in full.query("method == 'K-means'").iterrows()
        ],
        [1, 3.5, 3.5, 3.5, 4.5],
    )
    for k in range(2, 7):
        z = profiles.query("solution == @s", local_dict={"s": f"kmeans_k{k}"})
        put(
            f"profiles{k}",
            f"Table B{k-3 if k>=4 else 'supplement'+str(k)} K-means themes at {k} clusters",
            "Labels describe this particular partition and are provisional; fits at different counts are not nested.",
            ["C", "n", "Proposed theme", "Mean sil.", "Negative %"],
            [
                [
                    int(r.cluster),
                    int(r.papers),
                    r.proposed_label,
                    f"{r.mean_silhouette:.3f}",
                    f"{100*r.negative_fraction:.1f}",
                ]
                for _, r in z.iterrows()
            ],
            [1, 1.5, 9, 2.25, 2.25],
            (0, 2),
        )
    for c in range(1, MAIN_K + 1):
        z = phrases.query("solution == 'kmeans_k3' and cluster == @c")
        for method, tag in [("tfidf_contrast", "contrast"), ("c_tf_idf", "pooled")]:
            p = z[z.ranking.eq(method)]
            put(
                f"terms{c}_{tag}",
                f"Table C{2*c-(1 if tag=='contrast' else 0)} C{c} expressions ranked by {'TF-IDF contrast' if tag=='contrast' else 'class-based TF-IDF'}",
                "Ten eligible bigrams; counts show paper coverage. Full unigram and bigram rankings for both text methods and all reviewed counts are retained as CSV.",
                ["Rank", "Expression", "Inside", "Outside", "Score"],
                [
                    [
                        int(r.phrase_rank),
                        r.term,
                        f"{int(r.inside_count)}/{int(r.inside_n)}",
                        f"{int(r.outside_count)}/{int(r.outside_n)}",
                        f"{r.score:.4f}",
                    ]
                    for _, r in p.iterrows()
                ],
                [1.3, 6.2, 3, 3, 2.5],
                (0, 1),
            )
        ex = pd.read_csv(OUT / "review_papers.csv").query(
            "solution == 'kmeans_k3' and cluster == @c"
        )
        put(
            f"examples{c}",
            f"Table D{c} C{c} representative and counterexample papers",
            "Role selection is deterministic; one paper can have multiple roles. Excerpts with page/block references are saved in review_papers.csv.",
            ["Paper", "Title", "Selection", "Sil."],
            [
                [r.paper_id, r.title, r.selection, f"{r.cosine_silhouette:.3f}"]
                for _, r in ex.iterrows()
            ],
            [1.6, 9.5, 3.2, 1.7],
            (0, 1, 2),
        )
    for method in ("kmeans", "leiden"):
        for k in (3, 4, 5):
            z = pd.read_csv(
                OUT / f"overlaps/{method}_k{k}__{method}_k{k+1}.csv", index_col=0
            )
            put(
                f"transition_{method}_{k}",
                f"Table E{k-2+(3 if method=='leiden' else 0)} {method} membership changes from {k} to {k+1}",
                "Rows are the earlier partition; columns are the later partition. Off-diagonal movement is not an error because cluster numbers are arbitrary.",
                [f"{k} → {k+1}", *[f"C{x}" for x in z.columns]],
                [[f"C{i}", *[int(v) for v in row]] for i, row in z.iterrows()],
                [3] + [13 / (k + 1)] * (k + 1),
            )
    sen = pd.read_csv(OUT / "sensitivity.csv").query("check == 'citation manual links'")
    put(
        "citation_sensitivity",
        "Table F1 Effect of omitting manually recovered citation links",
        "Comparison is restricted to the 285 works connected in both graphs.",
        ["γ", "Reviewed count", "API-only count", "ARI"],
        [
            [r.setting, int(r.reference_k), int(r.alternative_k), f"{r.ARI:.3f}"]
            for _, r in sen.iterrows()
        ],
        [2, 5, 5, 4],
    )
    put(
        "citation_papers",
        "Table F2 Central papers in citation communities",
        "Representatives are selected by within-community degree, not text similarity.",
        ["Group", "n", "Two central papers"],
        [
            [
                r.Community,
                int(r.Works),
                "\n".join(r["Representative papers"].splitlines()[:2]),
            ]
            for _, r in citations.iterrows()
        ],
        [2, 1.5, 12.5],
        (0, 2),
    )
    hierarchy = pd.read_csv(OUT / "hierarchy_membership_15.csv")
    counts = hierarchy.groupby("cluster").size()
    put(
        "hierarchy_counts",
        "Table G1 Hierarchical cut sizes",
        "Average linkage with cosine distance; this partition is used for inspection, not as the main thematic map.",
        ["Cluster", "Works"],
        [[int(i), int(n)] for i, n in counts.items()],
        [8, 8],
    )
    save_json(OUT / "tables.json", tables)
    return select, tables


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    original = {str(p.relative_to(ROOT)): sha(p) for p in OLD.rglob("*") if p.is_file()}
    papers, x, passages, *_ = canonical_corpus(ROOT)
    assert papers.Key.tolist() == pd.read_csv(OLD / "corpus.csv").Key.tolist()
    with np.load(OLD / "text_partitions.npz") as f:
        partitions = {key: f[key].copy() for key in f.files}
    metrics = pd.read_csv(OLD / "text_metrics.csv")
    print("Checking graph neighbourhoods and Leiden subsamples", flush=True)
    graph_checks(x, metrics, partitions)
    print("Building expression and paper evidence", flush=True)
    phrases, profiles = lexical_evidence(papers, partitions)
    cautions = []
    for expression in ("yes yes", "fe yes"):
        pattern = r"\b" + r"\W+".join(expression.split()) + r"\b"
        eligible = (
            passages[passages.text.str.contains(pattern, case=False, regex=True)]
            .drop_duplicates("row")
            .head(5)
        )
        for _, p in eligible.iterrows():
            hit = re.search(pattern, p.text, re.I)
            start = max(0, hit.start() - 160)
            paper = papers.iloc[int(p.row)]
            cautions.append(
                dict(
                    expression=expression,
                    paper_id=paper.paper_id,
                    title=paper.title,
                    page=p.page,
                    block_id=p.block_id,
                    source_start=p.source_start,
                    excerpt=p.text[start : min(len(p.text), hit.end() + 200)],
                )
            )
    pd.DataFrame(cautions).to_csv(OUT / "lexical_caution_examples.csv", index=False)
    matched = match_clusters(partitions, profiles)
    selection, tables = table_bundle(phrases, profiles, matched)
    figures(partitions, matched)
    decision = {
        "main_method": "K-means",
        "main_k": MAIN_K,
        "status": "analyst-selected descriptive map",
        "basis": "Highest tested K-means silhouette and highest K-means 80% subsample stability; strong cross-method agreement; no prior six-cluster preference retained.",
        "alternatives": "Four and five show secondary subdivisions; six is exploratory, not the main result.",
        "selection_is_post_hoc": True,
        "names_entered_clustering": False,
    }
    save_json(OUT / "decision_log.json", decision)
    claims = [
        (
            "M08",
            "Some leading pooled or citation expressions are table vocabulary",
            "expression_rankings_all_solutions.csv; citation_terms_r1.csv; lexical_caution_examples.csv",
            "Examples preserve source page and block; not a universal artefact detector",
        ),
        (
            "M01",
            "Corpus flow 320 → 319 → 318 → 294 → 292",
            "corpus.csv; scope_exclusions.csv; pdf_to_work.csv",
            "Original extraction manifest and original evidence verified separately",
        ),
        (
            "M02",
            "2149 segments; 7 median; 1–28 range; maximum32748 tokens",
            "segments.csv; corpus_statistics.json",
            "Count/group by paper_id",
        ),
        (
            "M03",
            "Section flags 103 / 43 / 2 and extraction timing",
            "corpus_statistics.json",
            "Saved preparation and timing provenance in original run",
        ),
        (
            "M04",
            "PCA/UMAP diagnostics and length correlations",
            "projection_diagnostics.json",
            "Computed on same 292 works",
        ),
        (
            "M05",
            "Chosen count and separation/stability evidence",
            "selection_extended.csv; text_metrics.csv",
            "All candidates retained; no significance claim",
        ),
        (
            "M06",
            "Neighbour robustness for three through six",
            "neighbour_sensitivity.csv; additional_experiment_runs.json",
            "Hold resolution fixed",
        ),
        (
            "M07",
            "Leiden subset robustness",
            "leiden_subsample_stability.csv; additional_experiment_runs.json",
            "Shared works; no forced count",
        ),
        (
            "R01",
            "Main characteristic expressions",
            "expression_rankings_all_solutions.csv; main_expression_support_papers.csv",
            "kmeans_k3; contrast; phrase_rank<=3",
        ),
        (
            "R02",
            "Proposed theme names and representative papers",
            "labelled_profiles.csv; review_papers.csv",
            "kmeans_k3; names are interpretations",
        ),
        (
            "R03",
            "Cluster sizes, overlap and ambiguous memberships",
            "matched_cluster_diagnostics.csv; memberships.csv",
            "k=3; compare matched groups",
        ),
        (
            "R04",
            "Finer partitions and membership changes",
            "overlaps/; labelled_profiles.csv",
            "k=4,5,6; not nested",
        ),
        (
            "R05",
            "Citation group terms and text correspondence",
            "citation_terms_r1.csv; citation_profiles_r1.csv; overlaps/citation_r1__kmeans_k3.csv",
            "289 nonisolated works",
        ),
        (
            "R06",
            "Citation coverage sensitivity",
            "sensitivity.csv; graph_diagnostics.json",
            "Manual links excluded at fixed resolutions",
        ),
        (
            "R07",
            "Hierarchy leaves a large group and singletons",
            "hierarchy_membership_15.csv",
            "Group-size counts",
        ),
        (
            "R08",
            "Stable membership of one 29-paper group",
            "memberships.csv; analyst_labels.csv",
            "Equality of paper IDs across named solutions; not inferred from size alone",
        ),
    ]
    pd.DataFrame(
        claims, columns=["claim_id", "claim", "evidence", "operation_or_limit"]
    ).to_csv(OUT / "claim_evidence_index.csv", index=False)
    for path, digest in original.items():
        assert sha(ROOT / path) == digest, f"Earlier evidence changed: {path}"
    save_json(
        OUT / "run_manifest_v2.json",
        dict(
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            original_evidence_sha256=original,
            source_code_sha256=sha(Path(__file__)),
            main_k=MAIN_K,
            reused="Validated original vectors, memberships, text and coordinates",
            new="Phrase-only evidence exports; graph sensitivity; Leiden subset checks; report tables and figures",
        ),
    )
    print(selection.round(3).to_string(index=False), flush=True)
    print(
        pd.read_csv(OUT / "neighbour_sensitivity.csv").round(3).to_string(index=False),
        flush=True,
    )
    print(f"Saved {len(tables)} editable table specifications to {OUT}", flush=True)


if __name__ == "__main__":
    run()
