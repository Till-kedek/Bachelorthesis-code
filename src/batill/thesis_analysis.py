"""Offline thesis experiments on canonical works; no extraction, API or model calls.

Run from the repository with ``python -m batill.thesis_analysis`` or use the
Thesis_analysis notebook. Existing exploratory results are never overwritten.
Every fit uses original 2,560-dimensional vectors; projections are display only.
"""

from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from itertools import combinations
from pathlib import Path
import argparse
import hashlib
import json
import platform

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, leaves_list
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    silhouette_samples,
)
from sklearn.metrics.pairwise import cosine_distances
from threadpoolctl import threadpool_limits

from .topic_analysis import (
    load_topic_corpus,
    fit_partitions,
    build_vocabulary,
    topic_terms,
    describe_partition,
    compare_solutions,
)
from .selection import filter_analysis_corpus
from .citation_clustering import (
    load_citation_corpus,
    citation_network,
    citation_leiden_sweep,
    citation_cluster_tables,
    partition_comparison,
)
from .graph_clustering import similarity_graph, leiden_partition
from .embedding import load_plans
from .storage import read_json, fingerprint


def save_json(path, value):
    def convert(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(type(obj).__name__)

    Path(path).write_text(
        json.dumps(value, indent=2, default=convert, allow_nan=False) + "\n"
    )


def canonical_corpus(root):
    """Apply scope decisions, validate citation mapping, then retain one PDF per work."""
    papers, vectors, passages, validation = load_topic_corpus(
        root / "outputs/corpus_sections"
    )
    papers["Key"] = papers.pdf_sha256
    papers["Paper ID"] = papers.paper_id
    papers, vectors, exclusions, selection = filter_analysis_corpus(
        papers, vectors, root / "configs/analysis_exclusions.json"
    )
    graph_dir = (
        root
        / "outputs/citation_graph_openalex"
        / selection["selection_id"][:16]
        / "reviewed"
    )
    nodes, edges, mapping = load_citation_corpus(
        graph_dir, papers, selection["selection_id"]
    )
    indices = nodes.embedding_row.to_numpy()
    papers, vectors = (
        papers.iloc[indices].copy().reset_index(drop=True),
        vectors[indices],
    )
    old_to_new = dict(zip(papers.row, range(len(papers))))
    passages = passages.loc[passages.row.isin(old_to_new)].copy()
    passages["row"] = passages.row.map(old_to_new)
    papers["row"] = np.arange(len(papers))
    papers["embedding_title"] = papers.title
    papers["title"] = nodes.query_title.to_numpy()  # Display only; vectors unchanged.
    assert papers.Key.tolist() == nodes.Key.tolist()
    assert papers.paper_id.tolist() == nodes.paper_id.tolist()
    return (
        papers,
        vectors,
        passages,
        nodes,
        edges,
        mapping,
        exclusions,
        selection,
        validation,
        graph_dir,
    )


def subsample_stability(vectors, ks=range(2, 7)):
    """Fit five 80% subsets; compare every pair ONLY on shared retained works."""
    rows = []
    for k in ks:
        runs = []
        for seed in (101, 202, 303, 404, 505):
            indices = np.sort(
                np.random.default_rng(seed).choice(
                    len(vectors), int(0.8 * len(vectors)), replace=False
                )
            )
            with threadpool_limits(limits=1):
                labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(
                    vectors[indices]
                )
            runs.append(pd.Series(labels, index=indices))
        for a, b in combinations(range(len(runs)), 2):
            shared = runs[a].index.intersection(runs[b].index)
            rows.append(
                dict(
                    k=k,
                    left=a,
                    right=b,
                    shared=len(shared),
                    ARI=adjusted_rand_score(runs[a][shared], runs[b][shared]),
                )
            )
    return pd.DataFrame(rows)


def run_analysis(root, output=None):
    root = Path(root).resolve()
    output = Path(output or root / "reports/thesis/evidence")
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    print("Validating cached vectors and selecting canonical works", flush=True)
    (
        papers,
        x,
        passages,
        nodes,
        edges,
        mapping,
        exclusions,
        selection,
        validation,
        graph_dir,
    ) = canonical_corpus(root)
    papers.drop(columns="text").to_csv(output / "corpus.csv", index=False)
    exclusions.to_csv(output / "scope_exclusions.csv", index=False)
    mapping.to_csv(output / "pdf_to_work.csv", index=False)
    plans = {
        p["pdf_sha256"]: p
        for p in load_plans(root / "outputs/corpus_sections/prepared")
    }
    segment_rows = []
    flags = Counter()
    for _, p in papers.iterrows():
        plan = plans[p.Key]
        flags.update(
            set(f for s in plan["segments"] for f in s.get("section_review_flags", []))
        )
        for j, s in enumerate(plan["segments"]):
            segment_rows.append(
                dict(
                    paper_id=p.paper_id,
                    segment=j + 1,
                    section=s["section_name"],
                    token_count=s["token_count"],
                    content_tokens=s["content_tokens"],
                    group=s["section_group"],
                )
            )
    segments = pd.DataFrame(segment_rows)
    segments.to_csv(output / "segments.csv", index=False)
    timing = read_json(
        root / "outputs/corpus/extraction/timings/20260929T092613213046Z.json"
    )
    corpus_stats = {
        "unique_works": len(papers),
        "dimensions": x.shape[1],
        "segments": len(segments),
        "segments_median": float(papers.segments.median()),
        "segments_range": [int(papers.segments.min()), int(papers.segments.max())],
        "largest_segment_tokens": int(segments.token_count.max()),
        "section_flags": dict(flags),
        "extraction_wall_seconds": timing["wall_seconds"],
        "extraction_median_seconds": float(
            np.median(
                [
                    p["elapsed_seconds"]
                    for p in timing["papers"]
                    if p["status"] == "extracted"
                ]
            )
        ),
    }
    save_json(output / "corpus_statistics.json", corpus_stats)
    resolutions = np.round(np.arange(0.05, 2.001, 0.025), 3).tolist()
    solutions, metrics, sweep, graph, seed_labels = fit_partitions(
        x, ks=range(2, 16), resolutions=resolutions, prefer_reference=False
    )
    metrics.to_csv(output / "text_metrics.csv", index=False)
    sweep.to_csv(output / "text_leiden_sweep.csv", index=False)
    save_json(output / "text_seed_memberships.json", seed_labels)
    sub = subsample_stability(x)
    sub.to_csv(output / "kmeans_subsample_stability.csv", index=False)
    np.savez_compressed(
        output / "text_partitions.npz", **{k: v["labels"] for k, v in solutions.items()}
    )
    compare_solutions(
        {k: v for k, v in solutions.items() if len(set(v["labels"])) <= 6},
        output / "overlaps",
    ).to_csv(output / "text_agreement.csv", index=False)
    print("Computing citation communities and coverage sensitivity", flush=True)
    network = citation_network(nodes, edges)
    citation_res = np.unique(np.r_[0.05, np.arange(0.1, 2.501, 0.05)]).round(3).tolist()
    cs, cr = citation_leiden_sweep(network, citation_res)
    cs.to_csv(output / "citation_sweep.csv", index=False)
    save_json(
        output / "citation_seed_memberships.json",
        {str(r): v["runs"] for r, v in cr.items()},
    )
    summary, cm = citation_cluster_tables(nodes, network, cr[1.0])
    summary.to_csv(output / "citation_profiles_r1.csv", index=False)
    cm.to_csv(output / "citation_membership_r1.csv", index=False)
    api_network = citation_network(nodes, edges, include_manual=False)
    api_sweep, api_runs = citation_leiden_sweep(api_network, [0.5, 1.0, 1.5])
    api_sweep.to_csv(output / "citation_api_only_sweep.csv", index=False)
    comparison = []
    sensitivity = []
    for r in (0.5, 1.0, 1.5):
        keep = (cr[r]["labels"] > 0) & (api_runs[r]["labels"] > 0)
        sensitivity.append(
            {
                "check": "citation manual links",
                "setting": r,
                "compared": int(keep.sum()),
                "ARI": adjusted_rand_score(
                    cr[r]["labels"][keep], api_runs[r]["labels"][keep]
                ),
                "reference_k": len(set(cr[r]["labels"]) - {0}),
                "alternative_k": len(set(api_runs[r]["labels"]) - {0}),
            }
        )
    for name, fit in solutions.items():
        if len(set(fit["labels"])) > 6:
            continue
        m, overlap = partition_comparison(cr[1.0]["labels"], fit["labels"])
        comparison.append({"text_solution": name, "citation_resolution": 1.0, **m})
        overlap.to_csv(output / "overlaps" / f"citation_r1__{name}.csv")
    pd.DataFrame(comparison).to_csv(output / "citation_text_agreement.csv", index=False)
    for neighbours in (10, 25):
        alt = similarity_graph(x, neighbours=neighbours)
        for k in (3, 6):
            fit = solutions.get(f"leiden_k{k}")
            if fit is None:
                continue
            runs = [
                leiden_partition(alt, fit["resolution"], seed) for seed in (42, 43, 44)
            ]
            best = max(runs, key=lambda a: a["quality"])
            sensitivity.append(
                {
                    "check": "text neighbours",
                    "setting": neighbours,
                    "compared": len(x),
                    "ARI": adjusted_rand_score(fit["labels"], best["labels"]),
                    "reference_k": k,
                    "alternative_k": len(set(best["labels"])),
                }
            )
    pd.DataFrame(sensitivity).to_csv(output / "sensitivity.csv", index=False)
    save_json(
        output / "graph_diagnostics.json",
        {
            "text": graph["diagnostics"],
            "citation": network["diagnostics"],
            "citation_components": sorted(
                network["component_sizes"].tolist(), reverse=True
            ),
            "api_only": api_network["diagnostics"],
        },
    )
    print("Fitting display projections and hierarchy", flush=True)
    from umap import UMAP

    with threadpool_limits(limits=1):
        pca = PCA(n_components=2, svd_solver="full")
        pc = pca.fit_transform(x)
        umap = UMAP(
            n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42, n_jobs=1
        ).fit_transform(x)
    coordinates = papers[["paper_id", "Key"]].copy()
    coordinates[["PCA1", "PCA2"]] = pc
    coordinates[["UMAP1", "UMAP2"]] = umap
    coordinates.to_csv(output / "coordinates.csv", index=False)
    proj = {
        "PCA_explained_variance": pca.explained_variance_ratio_.tolist(),
        "PCA_trustworthiness_15": trustworthiness(
            x, pc, n_neighbors=15, metric="cosine"
        ),
        "UMAP_trustworthiness_15": trustworthiness(
            x, umap, n_neighbors=15, metric="cosine"
        ),
    }
    totals = (
        segments.groupby("paper_id")
        .content_tokens.sum()
        .reindex(papers.paper_id)
        .to_numpy()
    )
    for i in (0, 1):
        proj[f"PC{i+1}_segment_count_spearman"] = float(
            spearmanr(pc[:, i], papers.segments).statistic
        )
        proj[f"PC{i+1}_content_tokens_spearman"] = float(
            spearmanr(pc[:, i], totals).statistic
        )
    save_json(output / "projection_diagnostics.json", proj)
    distance = cosine_distances(x)
    distance = np.maximum((distance + distance.T) / 2, 0)
    np.fill_diagonal(distance, 0)
    tree = linkage(squareform(distance), method="average", optimal_ordering=True)
    np.save(output / "hierarchy_linkage.npy", tree)
    hl = fcluster(tree, t=15, criterion="maxclust")
    hierarchy_members = papers.drop(columns="text").assign(cluster=hl)
    hierarchy_members.to_csv(output / "hierarchy_membership_15.csv", index=False)
    print("Extracting transparent topic evidence", flush=True)
    vocab = build_vocabulary(papers)
    profiles = []
    all_terms = []
    memberships = []
    examples = []
    for name, fit in solutions.items():
        if len(set(fit["labels"])) > 6:
            continue
        terms = topic_terms(vocab, fit["labels"])
        prof, mem, ex = describe_partition(papers, x, passages, fit, terms)
        for frame, container in [
            (terms, all_terms),
            (prof, profiles),
            (mem, memberships),
            (ex, examples),
        ]:
            frame.insert(0, "solution", name)
            container.append(frame)
    for frames, name in [
        (profiles, "topic_profiles"),
        (all_terms, "topic_terms"),
        (memberships, "memberships"),
        (examples, "review_papers"),
    ]:
        pd.concat(frames, ignore_index=True).to_csv(output / f"{name}.csv", index=False)
    active = network["active"]
    cp = papers.iloc[active].copy().reset_index(drop=True)
    citation_terms = topic_terms(build_vocabulary(cp), cr[1.0]["labels"][active])
    citation_terms.to_csv(output / "citation_terms_r1.csv", index=False)
    # Data-only summary; names are reviewed and frozen separately by membership ID.
    save_json(
        output / "vocabulary.json",
        {
            "terms": len(vocab["terms"]),
            "min_df": 3,
            "max_df": 0.9,
            "ranking_support": "max(2, ceil(0.10 * cluster_size))",
            "analyzer": "unigrams and contiguous bigrams",
        },
    )
    make_figures(
        figures,
        metrics,
        sweep,
        cs,
        sub,
        umap,
        solutions,
        cr[1.0]["labels"],
        tree,
        distance,
    )
    inputs = [
        root / "configs/analysis_exclusions.json",
        root / "outputs/corpus_sections/embeddings/manifest.json",
        root / "outputs/corpus_sections/embeddings/embeddings.npz",
        graph_dir / "nodes.csv",
        graph_dir / "edges.csv",
        graph_dir / "pdf_to_work.csv",
    ]
    code = list((root / "src/batill").glob("*.py"))
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "packages": {
            p: version(p)
            for p in [
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "umap-learn",
                "igraph",
                "leidenalg",
                "matplotlib",
            ]
        },
        "selection": selection,
        "canonical_work_order_sha256": fingerprint(papers.Key.tolist()),
        "inputs": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in inputs
        },
        "code": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in code
        },
        "validation": validation,
        "settings": {
            "kmeans_k": list(range(2, 16)),
            "kmeans_seeds": [11, 22, 33, 44, 55],
            "kmeans_n_init": 10,
            "leiden_seeds": [42, 43, 44],
            "text_neighbours": 15,
            "text_resolution_selection": "highest mean seed ARI among exact-count candidates, then lower resolution",
            "citation_main_resolution": 1.0,
            "citation_resolution_selection": "conventional baseline, not optimized against text",
            "UMAP": {
                "n_neighbors": 15,
                "min_dist": 0.1,
                "metric": "cosine",
                "random_state": 42,
            },
            "term_min_df": 3,
        },
    }
    save_json(output / "run_manifest.json", manifest)
    print(f"Completed: {output}", flush=True)
    return output


def make_figures(
    folder, metrics, sweep, cs, sub, umap, solutions, citation, tree, distance
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.optimize import linear_sum_assignment

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 220,
        }
    )

    def save(fig, name):
        fig.savefig(folder / f"{name}.png", bbox_inches="tight")
        fig.savefig(folder / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
    for method, g in metrics.groupby("method"):
        axes[0].plot(g.k, g.cosine_silhouette, "o-", label=method)
        axes[1].plot(g.k, g.seed_ARI, "o-", label=method)
    s = sub.groupby("k").ARI.agg(["mean", "min", "max"])
    axes[1].plot(s.index, s["mean"], "s--", label="K-means 80% subsets")
    axes[0].set(
        xlabel="Number of clusters", ylabel="Mean cosine silhouette", xlim=(1.5, 15.5)
    )
    axes[1].set(
        xlabel="Number of clusters", ylabel="Adjusted Rand index", ylim=(0, 1.04)
    )
    axes[1].legend(fontsize=8)
    axes[2].plot(sweep.resolution, sweep.k, label="Text Leiden")
    axes[2].plot(cs.resolution, cs.communities, label="Citation Leiden")
    axes[2].set(xlabel="Resolution", ylabel="Communities", ylim=(0, 17))
    axes[2].legend(fontsize=8)
    save(fig, "selection_diagnostics")
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.6), layout="constrained")
    left = solutions["kmeans_k6"]["labels"]
    cmap = plt.get_cmap("tab10")
    for ax, (title, labels) in zip(
        axes,
        [
            ("K-means · six clusters", left),
            ("Text Leiden · six clusters", solutions["leiden_k6"]["labels"]),
            ("Citation Leiden · resolution 1", citation),
        ],
    ):
        tab = pd.crosstab(labels, left)
        tab = tab.loc[tab.index > 0]
        r, c = linear_sum_assignment(-tab.to_numpy())
        pairs = {
            tab.index[i]: tab.columns[j] for i, j in zip(r, c) if tab.iloc[i, j] > 0
        }
        colours = {value: cmap(partner - 1) for value, partner in pairs.items()}
        unpaired = sorted(set(labels) - {0} - set(pairs))
        for index, value in enumerate(unpaired):
            colours[value] = cmap([6, 8, 9][index])
        for value in sorted(set(labels)):
            mask = labels == value
            color = "#aaaaaa" if value == 0 else colours[value]
            ax.scatter(
                umap[mask, 0],
                umap[mask, 1],
                s=16,
                c=[color],
                alpha=0.85,
                marker="x" if value == 0 else "o",
                label=str(value),
            )
        ax.set(title=title, xlabel="UMAP 1", ylabel="UMAP 2")
        ax.tick_params(labelsize=12)
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)
        ax.title.set_fontsize(11)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=4,
            fontsize=8,
            frameon=False,
        )
    save(fig, "umap_comparison")
    fig, ax = plt.subplots(figsize=(6, 3.3), layout="constrained")
    tab = pd.crosstab(
        pd.Series(citation[citation > 0], name="Citation community"),
        pd.Series(left[citation > 0], name="K-means cluster"),
    )
    im = ax.imshow(tab, cmap="Blues", aspect="auto")
    for i in range(len(tab)):
        for j in range(len(tab.columns)):
            ax.text(
                j,
                i,
                str(tab.iloc[i, j]),
                ha="center",
                va="center",
                color=(
                    "white" if tab.iloc[i, j] > tab.to_numpy().max() * 0.55 else "black"
                ),
            )
    ax.set(
        xticks=range(len(tab.columns)),
        xticklabels=tab.columns,
        yticks=range(len(tab)),
        yticklabels=tab.index,
        xlabel="K-means cluster",
        ylabel="Citation community",
    )
    ax.tick_params(labelsize=12)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    fig.colorbar(im, ax=ax, label="Works")
    save(fig, "citation_overlap")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    dendrogram(
        tree,
        truncate_mode="lastp",
        p=15,
        show_leaf_counts=True,
        leaf_label_func=lambda node: (
            f"n={int(tree[node - len(distance), 3])}"
            if node >= len(distance)
            else "n=1"
        ),
        leaf_rotation=90,
        ax=axes[0],
        color_threshold=0,
    )
    axes[0].set(xlabel="Fifteen displayed subtrees", ylabel="Average cosine distance")
    order = leaves_list(tree)
    im = axes[1].imshow(
        1 - distance[np.ix_(order, order)],
        vmin=0,
        vmax=1,
        cmap="viridis",
        rasterized=True,
    )
    axes[1].set(
        xlabel="Works in hierarchical order", ylabel="Works in hierarchical order"
    )
    fig.colorbar(im, ax=axes[1], label="Cosine similarity")
    save(fig, "hierarchy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_analysis(args.root, args.output)
