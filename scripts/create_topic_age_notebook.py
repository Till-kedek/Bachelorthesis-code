"""Regenerate the topic-age notebook (clears its executed display outputs)."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source):
    cells.append(nbf.v4.new_code_cell(source.strip()))


md('''# Publication ages of the fixed private-equity topic clusters

This notebook takes **every paper, original cluster ID and recorded topic name** from the saved outputs of `Topic analysis PDFs.ipynb`, `Topic analysis Abstracts.ipynb`, and `Topic analysis Citations.ipynb`. It compares all five solutions separately: PDF K-means / embedding Leiden (five groups each), abstract K-means / embedding Leiden (six each), and direct-citation Leiden (six communities). **No clustering or naming is rerun.** Cluster numbers have no correspondence across methods.

Two questions are examined:

1. Which named clusters contain older or younger publications?
2. How does each cluster's **share of the publications in this corpus** change across periods?

Ages are `2026 − publication year`, measured in calendar years, not exact elapsed birthdays. The reference year is fixed and does not change when this notebook is rerun. Tables retain every paper and its date provenance. All analysis runs offline from eight checksum-pinned CSV files; no API keys, PDFs, embeddings or external lookups are needed.

**Run:** select Python (batill) here and run all cells. For a supervisor handoff, use `outputs/topic-age-reproduction.zip` and the reproduction instructions at the end. The computations are in `src/batill/topic_age_analysis.py`, included with the notebook.''')
md('''## 1. Load and validate the frozen inputs

`configs/topic-age-inputs.json` names the exact source exports and their SHA-256 checksums. The loader refuses changed inputs, unmatched names, duplicate identities, missing dates, or missing joins. It verifies the full corpus and cluster counts and proves that joining years preserves every original `(solution, paper, cluster, name)` tuple.

- **PDFs:** 292 canonical PDFs in each method, after the two duplicate copies were removed before text clustering. Join each retained PDF hash through the reviewed PDF-to-work map to the saved OpenAlex work year. Use the refreshed five-cluster memberships and names.
- **Abstracts:** 3,819 Scopus records in each method, using the publication year already saved with their named membership.
- **Citations:** 292 canonical works, using saved OpenAlex work years. Three isolates remain unassigned. Their ages are exported and audited, but they do not become a named topic or enter the assigned-cluster share denominator.

OpenAlex dates the indexed work and may differ from the available PDF version or bibliographic query year. Section 6 repeats the age and period comparisons using the saved alternative query year. No year is guessed from a filename or silently replaced. Names remain the source notebooks' provisional analyst proposals.''')
code('''from pathlib import Path
import json
import sys
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents)
            if (p / 'configs/topic-age-inputs.json').is_file())
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
from batill.topic_age_analysis import (
    load_paper_ages, analyze, write_tables, make_figures, write_manifest,
)

CONFIG = json.loads((ROOT / 'configs/topic-age-inputs.json').read_text())
OUT = ROOT / 'reports/topic_age_analysis'
REFERENCE_YEAR = CONFIG['reference_year']  # Frozen at 2026; never datetime.now().year.
papers, source_profiles = load_paper_ages(ROOT, CONFIG)
tables = analyze(papers, CONFIG)
write_tables(tables, OUT)
pd.set_option('display.max_colwidth', 90)
print(f'Fixed reference year: {REFERENCE_YEAR}; paper-membership rows: {len(papers):,}')
display(tables['coverage'])
print('All source memberships and names preserved. All publication years are valid.')''')
md('''## 2. Paper-level publication years and ages

`paper_ages.csv` is the complete list for all clusters and methods. A paper occurs once per solution, so **do not pool rows across solutions**. The PDF and citation analyses use the same 292 canonical works, with different partitions; the abstract corpus has different coverage. Inspect an original cluster below; change the three selectors to view another group. The preview is bounded to 30 rows; the exported table is complete.''')
code('''INSPECT_ANALYSIS = 'pdfs'  # pdfs / abstracts / citations
INSPECT_SOLUTION = 'kmeans_k5'  # See coverage table above.
INSPECT_CLUSTER = 3  # Original IDs; citation 0 means unassigned.
selection = papers.loc[papers.analysis.eq(INSPECT_ANALYSIS)
                       & papers.solution.eq(INSPECT_SOLUTION)
                       & papers.cluster.eq(INSPECT_CLUSTER)].sort_values(['publication_year', 'paper_id'])
if selection.empty:
    raise ValueError('Choose a source analysis, solution and original cluster ID shown in the summary.')
display(Markdown(f"**{selection.topic_name.iloc[0]}** — {len(selection)} papers"))
display(selection[['paper_id', 'title', 'doi', 'publication_year', 'age_years',
                   'year_source', 'alternative_year', 'canonical_paper_id']].head(30))
print('Complete paper list:', OUT / 'paper_ages.csv')''')
md('''## 3. Which clusters are older or younger?

The table ranks named clusters from older to younger **within each solution**. Median age summarizes a typical paper; mean age, quartiles and the full year range show skew and overlap. All papers have equal weight in their source analysis unit. `small_cluster` flags groups with fewer than 10 papers; their results can be driven by a few documents.

The boxplots below show distributions, not uncertainty intervals. Overlap between clusters is expected. A younger median alone does not establish that a topic replaced an older one.''')
code('''summary = tables['cluster_age_summary']
for (analysis, solution), group in summary.loc[summary.assigned].groupby(['analysis', 'solution'], sort=True):
    display(Markdown(f'### {analysis.title()} — {solution}'))
    ranked = group.sort_values(['median_age', 'mean_age', 'cluster'], ascending=[False, False, True])
    with pd.option_context('display.max_colwidth', None):
        display(ranked[['display_cluster', 'topic_name', 'papers', 'median_year', 'median_age',
                        'mean_age', 'q25_age', 'q75_age', 'first_year', 'last_year', 'small_cluster']].round(2))''')
md('''## 4. Publication shares through time

For each period, `share = papers in this fixed cluster / all assigned papers in this solution and period × 100`. Shares sum to 100% within a nonempty period. This distinguishes topic composition from growth in the total number of publications. Zero cluster counts are included; a period with no assigned papers has an undefined share (blank), not 0%.

Periods are fixed in advance: through 1999, 2000–2009, 2010–2019, and 2020–2026. They have unequal durations. **2026 and the final decade are incomplete**, so raw counts cannot be interpreted as comparable publication rates. The heatmaps display the denominators and each cell's count. `annual_shares.csv` also contains every calendar year through the fixed reference year, including zero-count years, for inspection without aggregation.''')
code('''figures = make_figures(tables, CONFIG, OUT / 'figures')
for filename, figure in figures:
    display(figure)
    plt.close(figure)
print('Standalone PNG and PDF figures:', OUT / 'figures')''')
md('''## 5. How much did topic shares change?

The following descriptive contrast compares **2000–2009 with 2020–2026** (fixed in the configuration). The intervening decade remains visible above. A change of +10 percentage points means the topic occupies a 10-point larger fraction of this corpus's dated, assigned papers in the later period. This is not a ten-percent growth rate. The contrast does not fit new clusters or select a breakpoint to maximize a difference; other period choices may give different conclusions.''')
code('''changes = tables['period_changes']
for (analysis, solution), group in changes.groupby(['analysis', 'solution'], sort=True):
    display(Markdown(f'### {analysis.title()} — {solution}'))
    with pd.option_context('display.max_colwidth', None):
        display(group[['display_cluster', 'topic_name', 'papers_early', 'period_assigned_papers_early',
                       'share_pct_early', 'papers_late', 'period_assigned_papers_late',
                       'share_pct_late', 'change_percentage_points']].round(2))''')
md('''## 6. Does the choice of publication date matter?

Primary dates are OpenAlex work years for PDFs/citations and Scopus years for abstracts. The alternative is the **saved bibliographic/PDF query year** for PDFs/citations; it is not necessarily the publication year of a final journal version. Abstracts have no independent alternative in this workflow and keep their Scopus dates.

The sensitivity table holds every cluster and name fixed and repeats both median dates and the early/late share contrast. `share_change_sign_agrees` compares the directions of the two share changes, not their statistical significance. `alternative_period_shares.csv` provides all alternative counts and denominators. The paper-level audit keeps both dates and their signed difference. Differences between the two databases used for the two corpora are not resolved by this check.''')
code('''sensitivity = tables['year_sensitivity']
with pd.option_context('display.max_rows', None, 'display.max_colwidth', 65):
    display(sensitivity.loc[sensitivity.analysis.ne('abstracts') & sensitivity.assigned,
        ['analysis', 'solution', 'display_cluster', 'topic_name', 'median_year_primary',
         'median_year_alternative', 'change_percentage_points_primary',
         'change_percentage_points_alternative', 'share_change_sign_agrees']].round(2))
# One row per analysis unit; do not double-count the two PDF solutions in this audit.
date_differences = papers.loc[papers.year_difference.ne(0)].drop_duplicates(['analysis', 'paper_id'])
display(date_differences[['analysis', 'paper_id', 'title', 'publication_year',
                          'alternative_year', 'year_difference']].head(20))
print('Full date differences are retained in paper_ages.csv.')''')
md('''## 7. Evidence for interpreting a shift

The statements below are generated from the exported tables and refer only to these corpora. They identify the largest observed gains/losses under the predeclared period contrast, without testing significance. Review the complete profiles and date sensitivity before choosing a thesis interpretation.

**Limits on the claim:** these are selected corpora, not a census or random sample of private-equity research. Coverage, publication lags, document types and the incomplete final period can change the observed composition. In particular, some abstract clusters include industry/deal news, so their temporal patterns cannot all be described as academic research. Citation communities also reflect citation opportunities and age. The partitions are fixed over the whole observation window; this analysis does not estimate when topics first emerged. Agreement between methods using the same papers is not independent replication. Use wording such as “the composition of the analyzed literature shifted toward …” when supported by both the period shares and the cluster evidence; age rankings alone do not demonstrate a field-wide or causal shift.''')
code('''for (analysis, solution), group in changes.groupby(['analysis', 'solution'], sort=True):
    ordered = group.dropna(subset=['change_percentage_points']).sort_values(
        ['change_percentage_points', 'cluster'], ascending=[False, True])
    if ordered.empty:
        continue
    gain, loss = ordered.iloc[0], ordered.iloc[-1]
    def description(row):
        return (f"**{row.topic_name}** ({row.display_cluster}): "
                f"{row.share_pct_early:.1f}% to {row.share_pct_late:.1f}% "
                f"({row.change_percentage_points:+.1f} percentage points)")
    display(Markdown(f"**{analysis.title()} / {solution}**, {CONFIG['comparison_periods'][0]} → "
                     f"{CONFIG['comparison_periods'][1]}: largest observed increase, {description(gain)}; "
                     f"largest observed decrease, {description(loss)}."))
# Preserve the source naming caveats beside the temporal evidence.
with pd.option_context('display.max_colwidth', 100):
    display(source_profiles[['analysis', 'solution', 'cluster', 'proposed_label', 'review_caveat']])''')
md('''## 8. Reproduction checks and supervisor handoff

The checks below reconcile all paper counts, verify nonempty-period shares sum to 100%, and ensure that exactly the three citation isolates remain unassigned. The manifest records input hashes, code/config hashes, the fixed reference year, package versions and output hashes. No random operation is used.

**Send `outputs/topic-age-reproduction.zip`**, or this entire project including its input folders. The compact archive contains this executed notebook, all eight frozen CSV inputs at their original relative paths, code, configuration, a pinned dependency lock, reference tables and figures. It is sufficient for this age analysis independently of the other notebooks. The larger topic-analysis archive reproduces the original clustering interpretations separately.

After unpacking the compact archive, run inside `topic-age-reproduction/`:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r configs/topic-age-requirements.lock
.venv/bin/python scripts/reproduce_topic_ages.py --check
```

The checker executes this notebook in a fresh Python process, blocks network connections, and compares every generated CSV **byte for byte** with the supplied reference hashes. It checks outputs after recomputation, so it can also regenerate deleted tables. Source input changes fail before computation. The tested environment is Python 3.12.7 on macOS Apple Silicon. Figures and notebook displays are excluded from byte comparisons because rendering can vary across systems; their numerical data are verified. Dependency installation needs internet access; rerunning the analysis does not.

For interactive use, register the environment with `.venv/bin/python -m ipykernel install --user --name batill-topic-ages`, then select that kernel in your notebook editor. To package a reviewed update, rerun the checker, then `scripts/build_topic_age_bundle.py`. `--record-reference` deliberately replaces the expected table hashes and is only for reviewed changes, never needed for normal reproduction. Editing the input configuration defines a new analysis and requires a new reviewed reference.''')
code('''for (analysis, solution), group in papers.groupby(['analysis', 'solution']):
    expected = CONFIG['sources'][analysis]['expected_papers_per_solution']
    assert len(group) == expected and group.paper_id.is_unique
    assert group.publication_year.notna().all()
    assert group.age_years.eq(REFERENCE_YEAR - group.publication_year).all()
assert papers.loc[~papers.assigned, 'analysis'].eq('citations').all()
assert (~papers.assigned).sum() == 3
for filename in ['period_shares', 'annual_shares', 'alternative_period_shares']:
    table = tables[filename]
    for _, group in table.groupby(['analysis', 'solution', 'period']):
        denominator = group.period_assigned_papers.iloc[0]
        assert group.papers.sum() == denominator
        if denominator:
            assert abs(group.share_pct.sum() - 100) < 1e-9
        else:
            assert group.share_pct.isna().all()
manifest = write_manifest(ROOT, CONFIG, OUT)
print(f"PASS: fixed memberships, dates, counts and shares checked; {len(manifest['artifacts_sha256'])} tables saved.")
print('Output directory:', OUT)
print('Run scripts/reproduce_topic_ages.py --check for comparison against the frozen reference.')''')

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python (batill)', 'language': 'python', 'name': 'batill'},
    'language_info': {'name': 'python', 'version': '3.12.7'},
})
for index, cell in enumerate(notebook.cells):
    cell.id = f'topic-ages-{index:02d}'
nbf.validate(notebook)
nbf.write(notebook, ROOT / 'Topic analysis Ages.ipynb')
print(f'Wrote {len(cells)} cells to Topic analysis Ages.ipynb')
