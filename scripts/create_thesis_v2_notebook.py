"""Create a separate notebook following methods, selection, then thematic results."""

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
nb = nbf.v4.new_notebook()
nb.cells = [
    md(
        "# NLP and ML analysis of private equity literature\n\nVersion two separates the methodological evaluation and choice of partitions from the substantive cluster results. It reuses the validated 292-work corpus and keeps the earlier report and notebooks unchanged. The supporting Word files are in `reports/thesis_v2`. No API call, extraction or embedding inference is needed."
    ),
    code(
        "from pathlib import Path\nimport sys, json\nimport pandas as pd\nfrom IPython.display import display, Image\nROOT = Path.cwd()\nassert (ROOT / 'src/batill').is_dir()\nsys.path.insert(0, str(ROOT / 'scripts'))\nOUT = ROOT / 'reports/thesis_v2/evidence'\npd.set_option('display.max_colwidth', 120)"
    ),
    md(
        "## 1 Reproduce and inspect the methodology\n\nThe original partitions are frozen in `reports/thesis/evidence`. Set the switch below to recompute the added experiments: Leiden on five 80% subsets and graph neighbourhood checks for 3–6 clusters. The routine also rebuilds expression evidence and report tables, writing only to the version-two folder. The original full-fitting code remains in `batill.thesis_analysis`."
    ),
    code(
        "RECOMPUTE_ADDED_EXPERIMENTS = False\nif RECOMPUTE_ADDED_EXPERIMENTS:\n    from thesis_v2_evidence import run\n    run()\nfrom verify_thesis_v2 import verify\nverification = verify()\ntables = json.loads((OUT / 'tables.json').read_text())\ndef show_table(name):\n    t = tables[name]\n    print(t['title'])\n    print(t['note'])\n    display(pd.DataFrame(t['rows'], columns=t['headers']))\nshow_table('flow')\nshow_table('algorithms')\nshow_table('preparation')"
    ),
    md(
        "## 2 Evaluate and select a partition\n\nSelection precedes thematic naming. Compare separation, initialisation stability, subset stability and cross-method agreement. Three clusters have the highest K-means silhouette and the highest subset agreement for both text methods. Four remains a credible finer view; six has no priority because it was an earlier candidate. These are exploratory choices, not a proven true count."
    ),
    code(
        "show_table('selection')\nshow_table('seed_summary')\nshow_table('neighbours')\ndisplay(pd.read_csv(OUT / 'leiden_subset_counts.csv').pivot(index='k', columns='subset_seed', values='resulting_k'))\ndisplay(Image(filename=str(OUT / 'figures/selection_diagnostics.png')))"
    ),
    md(
        "## 3 Establish themes from expressions and papers\n\nFirst inspect clusters as numerical groups. The expression table uses the three highest-ranked eligible bigrams under mean per-paper TF-IDF contrast. Counts are paper-level coverage inside and outside each group. Names are then interpreted from terms and central titles; they do not enter the clustering algorithm."
    ),
    code(
        "show_table('expressions')\nshow_table('themes')\nshow_table('agreement')\ndisplay(Image(filename=str(OUT / 'figures/umap_comparison.png')))"
    ),
    md(
        "## 4 Inspect supporting and contradictory evidence\n\nChoose a text solution and cluster. Both lexical rankings, their coverage, central papers, boundary cases and seeded random examples remain visible. Excerpts selected around terms can contain extraction or table noise and are not independent annotation."
    ),
    code(
        "SOLUTION = 'kmeans_k3'\nCLUSTER = 2\nphrases = pd.read_csv(OUT / 'expression_rankings_all_solutions.csv')\ndisplay(phrases[(phrases.solution == SOLUTION) & (phrases.cluster == CLUSTER)])\nexamples = pd.read_csv(OUT / 'review_papers.csv')\ndisplay(examples[(examples.solution == SOLUTION) & (examples.cluster == CLUSTER)][['paper_id','title','selection','cosine_silhouette','page','block_id','excerpt']])\nsupport = pd.read_csv(OUT / 'main_expression_support_papers.csv')\ndisplay(support[(support.cluster == CLUSTER) & support.inside])"
    ),
    md(
        "## 5 Examine finer resolutions\n\nCounts are separately fitted: changing k can move papers among several groups. Read transition matrices alongside names. Three is the primary map; four to six are secondary results, with all corresponding evidence preserved."
    ),
    code(
        "for k in (4, 5, 6):\n    show_table(f'profiles{k}')\nfor method in ('kmeans', 'leiden'):\n    for k in (3, 4, 5):\n        show_table(f'transition_{method}_{k}')\n    display(Image(filename=str(OUT / f'figures/umap_{method}_counts.png')))"
    ),
    md(
        "## 6 Compare citation communities\n\nCitation resolution one is chosen independently of text agreement. The six citation communities include a disconnected three-work component; three isolates remain unassigned. Terms are shown before thematic interpretation, and the text overlap relates citation groups to the already established text themes. Coverage sensitivity prevents treating the citation count as confirmation of a text count."
    ),
    code(
        "show_table('citations')\nshow_table('citation_papers')\nshow_table('citation_sensitivity')\ndisplay(pd.read_csv(OUT / 'citation_text_agreement.csv').round(3))\nshow_table('hierarchy_counts')"
    ),
    md(
        "## 7 Trace claims and reproduce the documents\n\nThe Word chapters contain nine concise evidence tables; the separate editable appendix retains alternatives and paper-level review material. CSV files preserve more detail than the appendix can display. `README.md` maps files to report sections. Direct Word edits and Markdown edits are separate: transfer any direct edits before rebuilding."
    ),
    code(
        "display(pd.read_csv(OUT / 'claim_evidence_index.csv'))\nprint('Build Word files: python scripts/build_thesis_v2.py')\nprint('Verify evidence: python scripts/verify_thesis_v2.py')"
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
nbf.write(nb, ROOT / "Thesis_analysis_v2.ipynb")
