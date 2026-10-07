"""Create the readable runner; execution is a separate explicit step."""

from pathlib import Path
import nbformat as nbf

root = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
nb.cells = [
    md(
        "# Thesis analysis on the final selection\n\nThis notebook reproduces the numerical evidence used in `reports/thesis/thesis_sections.docx`. It uses **292 unique works**, applies the 24 recorded scope exclusions and collapses the two reviewed duplicate pairs. The earlier clustering notebooks are preserved as exploratory work.\n\nNo PDF extraction, model inference, API key or network connection is required. All clustering is fitted in the original 2,560-dimensional embedding space. PCA and UMAP are used only for display."
    ),
    code(
        "from pathlib import Path\nimport sys\nimport json\nimport pandas as pd\nfrom IPython.display import display, Image, Markdown\nROOT = Path.cwd()\nif not (ROOT / 'src/batill').exists():\n    raise RuntimeError('Open this notebook with batill as the working directory')\nOUT = ROOT / 'reports/thesis/evidence'\nfrom batill.thesis_analysis import run_analysis\n"
    ),
    md(
        "## 1 Reproduce the experiments\n\nSet `RECOMPUTE = True` to regenerate the evidence folder. Cached results are displayed when it is false. The experiment fits K-means for 2–15 clusters and sweeps the Leiden resolution parameter. Five K-means seeds, three Leiden seeds and five 80% subsamples are saved. The canonical work order, input hashes, code hashes and package versions are recorded in `run_manifest.json`.\n\nLeiden reference resolutions from older corpora are deliberately not reused. For each attainable count, the displayed text partition uses the most stable tested resolution, with lower resolution breaking ties. This is exploratory selection; its stability is not a held-out performance estimate."
    ),
    code(
        "RECOMPUTE = False\nif RECOMPUTE:\n    run_analysis(ROOT)\nif not (OUT / 'run_manifest.json').exists():\n    raise FileNotFoundError('Set RECOMPUTE=True for the first run')\nmanifest = json.loads((OUT / 'run_manifest.json').read_text())\ndisplay(pd.Series(json.loads((OUT / 'corpus_statistics.json').read_text()), name='Value').to_frame())"
    ),
    md(
        "## 2 Cluster counts and stability\n\nSilhouette measures separation using cosine distances. Adjusted Rand index (ARI) measures membership agreement after correction for chance. Seed agreement, subsample agreement and agreement between methods answer different questions; none is an accuracy score. The complete sweeps are retained, including settings that give more than six clusters."
    ),
    code(
        "metrics = pd.read_csv(OUT / 'text_metrics.csv')\ndisplay(metrics.query('k <= 6').round(3))\ndisplay(pd.read_csv(OUT / 'kmeans_subsample_stability.csv').groupby('k').ARI.agg(['mean', 'min', 'max']).round(3))\ndisplay(Image(filename=str(OUT / 'figures/selection_diagnostics.png')))"
    ),
    md(
        "## 3 Text and citation agreement\n\nThe citation graph retains 292 works, but isolates receive label zero and are excluded from citation agreement calculations. Resolution one is a conventional citation baseline, chosen independently of text agreement. Citation community numbers and text cluster numbers are arbitrary identifiers. The UMAP coordinates are identical in all panels and colours are paired by overlap only."
    ),
    code(
        "display(pd.read_csv(OUT / 'citation_sweep.csv').query('resolution == 1').round(3))\ndisplay(pd.read_csv(OUT / 'citation_text_agreement.csv').round(3))\ndisplay(pd.read_csv(OUT / 'citation_profiles_r1.csv'))\ndisplay(Image(filename=str(OUT / 'figures/umap_comparison.png')))\ndisplay(Image(filename=str(OUT / 'figures/citation_overlap.png')))"
    ),
    md(
        "## 4 Topic evidence and provisional names\n\nThe two transparent term rankings are class-based TF-IDF and mean per-paper TF-IDF contrast. A term must occur in at least three papers globally and at least 10% of a cluster (minimum two papers) to appear in a displayed ranking. Names are qualitative interpretations, recorded against the exact membership hash in `analyst_labels.csv`. They are not automatically generated ground truth.\n\nUse the full `cluster_review_cards.md` and `review_papers.csv` to inspect central, boundary and seeded random examples. Source excerpts include PDF page and block information. The names should be reviewed again if the corpus or memberships change."
    ),
    code(
        "profiles = pd.read_csv(OUT / 'labelled_profiles.csv')\ndisplay(profiles.loc[profiles.solution.isin(['kmeans_k3', 'kmeans_k5', 'kmeans_k6']), ['solution','cluster','papers','proposed_label','contrast_terms','representatives']])\nSOLUTION = 'kmeans_k5'\nCLUSTER = 1\nterms = pd.read_csv(OUT / 'topic_terms.csv')\ndisplay(terms[(terms.solution == SOLUTION) & (terms.cluster == CLUSTER)].head(30))\nexamples = pd.read_csv(OUT / 'review_papers.csv')\ndisplay(examples[(examples.solution == SOLUTION) & (examples.cluster == CLUSTER)][['paper_id','title','selection','cosine_silhouette','page','excerpt']])"
    ),
    md(
        "## 5 Sensitivity and limitations\n\nThe API-only citation graph omits the 65 manually verified links. Its comparison uses only works connected in both graph versions. The text graph checks use 10 and 25 neighbours at the original resolution without forcing the original count. These are descriptive sensitivity checks, not confidence intervals.\n\nSection flags indicate boundaries that need inspection. Small correlation with segment count does not prove all document-length effects are removed. Hierarchical small clusters flag review candidates; they do not justify automatic exclusion."
    ),
    code(
        "display(pd.read_csv(OUT / 'sensitivity.csv').round(3))\ndisplay(pd.Series(json.loads((OUT / 'projection_diagnostics.json').read_text()), name='Value'))\ndisplay(Image(filename=str(OUT / 'figures/hierarchy.png')))\ndisplay(pd.read_csv(OUT / 'hierarchy_membership_15.csv').groupby('cluster').size().rename('works').to_frame())"
    ),
    md(
        "## 6 Reading and reproducing the report\n\nThe editable chapter text is `reports/thesis/thesis_sections.md`; the Word document follows the supplied TUM template. `reports/thesis/README.md` maps each claim and figure to its evidence and explains regeneration. `scripts/thesis_review.py` freezes the current analyst names after the numerical analysis; changing assignments requires rechecking those names. No names enter the clustering fit.\n\nThe main interpretation is a three-group overview with a five-group thematic breakdown. Six groups offer an additional employment distinction but show weaker stability. This is a choice of useful granularity, not proof that the literature has a unique true number of topics."
    ),
]
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3 (ml)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python"},
}
nbf.write(nb, root / "Thesis_analysis.ipynb")
