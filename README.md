# Private-equity research paper embeddings

A reproducible pipeline for turning a folder of research PDFs into a matrix of paper
embeddings for thematic mapping. The implementation separates PDF extraction,
input preparation and model inference. A notebook exposes each stage for inspection;
the same functions are available through a command-line interface.

The representation uses the paper's retained prose and document structure. Detected
formulas are omitted rather than transcribed into LaTeX. The default embedding model
is **Qwen3-Embedding-4B**, with a pinned revision and a 32,768-token context limit.
The retained final experiments compare text and citation communities and save transparent topic evidence.

## Running on this Mac

This computer has a dedicated environment in `.venv/`. Double-click
`start_jupyter.command` to open JupyterLab at `Thesis_analysis_v2.ipynb`, then
select **Python (batill)** if asked for a kernel. From a terminal in this folder:

```bash
source .venv/bin/activate
python -m pytest -q
python scripts/run_local_analysis.py
```

The local runner validates both saved embedding datasets, refits the base text
and citation clustering, compares the new memberships with the saved experiment,
reruns the version-two sensitivity experiments, verifies their evidence, and builds
new Word drafts. Each invocation writes to a dated folder under
`outputs/local_runs/`; the existing thesis reports remain the reference snapshot.
The version-two workflow reuses the original base memberships by design.

The current revised chapters are in `reports/thesis_v2/`; see its README for the
evidence map. Existing embeddings and reviewed citations suffice for local analysis.
Fresh CUDA embedding inference requires a suitable NVIDIA machine; the original
GPU configuration in `configs/pipeline.json` is retained. New PDF extraction or
tokenization may download model resources on first use.

To recreate this environment with Python 3.12:

```bash
python -m venv .venv
.venv/bin/python -m pip install -c configs/macos-analysis-constraints.txt -e '.[extract,embed,notebook,analysis,graph,citations,test,report]' jupyterlab
.venv/bin/python -m ipykernel install --sys-prefix --name batill --display-name 'Python (batill)'
.venv/bin/python -m ipykernel install --sys-prefix --name ml --display-name 'Python (batill, legacy ml)'
```

## Final thesis analysis

### Selected papers represented by title and abstract

Run these new notebooks in order with **Python (batill)**:

1. [Embedding Elite Paper Abstracts.ipynb](<Embedding Elite Paper Abstracts.ipynb>)
   reproduces the 292-work PDF selection, prepares reviewed PDF abstracts, prints
   missing-abstract IDs/titles/counts, and computes or reuses their BGE-M3 vectors.
2. [Clustering Elite Papers Abstracts.ipynb](<Clustering Elite Papers Abstracts.ipynb>)
   runs the current large-abstract clustering method and saves every fitted
   K-means/Leiden partition. Choose a preferred count/resolution after inspection;
   neither a final count nor old topic labels are assigned automatically.
3. [Topic analysis Elite Papers Abstracts.ipynb](<Topic analysis Elite Papers Abstracts.ipynb>)
   loads those exact memberships to inspect expressions, representative and
   borderline papers, full abstracts, and their PDF source spans.

The topic notebook's section 5, immediately after the cluster comparison, adds two
content-reviewed papers per cluster for the saved **Leiden 0.7** partition (282
abstracts; cluster sizes 65, 86, 110 and 21). Review prompts are PE acquisitions of
listed firms (C1), LBO/MBO process (C2), fund-performance criteria (C3), and health
care/nursing homes (C4). Each example includes its PE relevance, rationale, source
evidence, full included abstract and diagnostics from the abstract vectors.
The sixteen passages comprise thirteen abstract excerpts and three supplementary
PDF passages; the latter explain transaction context and do not enter clustering.
P229's approved external abstract is explicitly distinguished from PDF extraction.
`configs/elite_abstract_cluster_examples.json` records the judgments and binds them
to the exact texts, sources, vector input and partition catalog. Running the section
exports the eight examples, sixteen passages and all-paper diagnostics to
`reports/elite_abstract_cluster_examples/<review-checksum>/`.

The approved corpus now contains **282 of 292 works**: 278 reviewed PDF
abstracts and **four external full abstracts** (P047, P229, P263, P284).
Short publisher descriptions are excluded: **P066, P070, P180, P299, P318**.
Together with **P159, P168, P169, P204 and P297**, ten works lack usable full
abstracts. Notebook 1 reports their titles, IDs and count.

The four approved additions are selected by
`configs/selected_abstracts_supplements.json`; cached source records remain in
`data/selected_abstracts/supplements/`. The retrieval script
`scripts/fetch_selected_abstract_supplements.py` only includes these four full
abstracts. P263 uses the Scopus abstract of the 2009 Chicago Law Review article,
matched by title, authors and publication details; its blank Scopus DOI is not
claimed to match the selected working-paper DOI. AEA supplies the other three.

The previous 287-paper run and results are historical. Its active embedding and
partition configs and executed notebooks were archived under
`outputs/selected_abstracts/archive/` when the short descriptions were removed.
The current topic notebook loads the saved 282-paper embedding and clustering
catalog and has executed outputs. If those upstream inputs change, rerun the
notebooks in order and review any representative-paper choices invalidated by
changed texts or memberships.

PDF-span decisions and title repairs remain in
`configs/selected_abstracts_review.json`, bound to extraction checksums. Approved
supplements explicitly supersede earlier PDF-only missing/review decisions.
No generated summaries, introductions or title-only records are included.
Four damaged/missing PDF titles use previously reviewed version metadata;
all title and abstract sources are disclosed in the audit.

The encoder matches the completed 3,819-record run: `BAAI/bge-m3`, revision
`5617a9f61b028005a4858fdac845db406aefb181`, title + blank line + abstract,
model-defined CLS pooling, 1,024 float32 dimensions, L2 normalization, no prompt,
and no truncation/chunking. It uses the cached model without API calls. Inputs
exceeding 8,192 tokens stop execution. PDF OCR differences remain a limitation;
matching the representation does not make the two corpus selections identical.

The clustering notebook preserves average-linkage cosine hierarchy, PCA/UMAP
inspection, K-means k=2–15 (five seeds, ten initializations; five 80% subsamples),
and positive union 15-neighbour cosine Leiden graphs (resolutions 0.10–0.95 in
steps of 0.05; seeds 42, 43, 44). Clustering uses the original embeddings; 2-D
projections are for display. Numeric cluster IDs do not map across corpora.

New outputs are isolated under `outputs/selected_abstracts/`. Each prepared
bundle contains `extraction_audit.csv`, `missing_abstracts.csv`,
`extraction_review_needed.csv`, `pending_review.csv`, `supplementary_abstracts.csv`,
ready inputs, and checksums.
`configs/selected_abstracts_input.json` and
`configs/selected_abstracts_partitions.json` hand off explicit authenticated
runs. Changes to included text require re-embedding and reclustering; stale
memberships are rejected. Existing PDF and large-abstract workflows are retained.

### Institution ranking

Open [Institution Analysis.ipynb](<Institution Analysis.ipynb>) with **Python
(batill)** and run all cells to rank author-affiliated institutions across the
existing **292 canonical works**. Each paper counts once per institution, even
when several coauthors share that affiliation. Papers with multiple institutions
count toward each institution; ties share a rank.

The notebook uses saved reviewed OpenAlex work identities and affiliations,
supplemented by source-linked reviews in
`configs/institution_affiliation_supplements.json`. It runs offline, writes
**Institution Analysis.xlsx** in the project root, and saves CSV tables, a chart of institutions with at least 12 papers, and a source-hash manifest under `reports/institution_analysis/`. The
workbook includes the complete ranking, all papers, paper–institution links,
author evidence, and methods. Full paper coverage does not imply exhaustive
manual verification of every secondary affiliation in OpenAlex. The optional
`institutions` dependency installs the Excel writer (`openpyxl`).

The cluster section joins these same 292 papers to the saved five-cluster PDF
embedding Leiden solution (resolution 0.95, seed 44). It exports institution-by-cluster
counts and percentages of each institution's papers, plus the cluster definitions
and paper-level memberships. The graph shows institutions with at least 12 papers;
the tables include all institutions. Each institution's cluster counts sum to its
overall paper count, and its percentages sum to 100%. Memberships and provisional
topic names are preserved; clustering is not rerun.

The notebook generator is `scripts/create_institution_analysis_notebook.py`;
regenerating clears execution outputs and should not overwrite independent
notebook edits.

### Final broad-abstract analysis

The approved screening run `5779f6345a8f5d18` retains **2,056 of 3,819 papers**.
The following clustering/topic notebooks retain that 2,056-paper corpus:

1. [Clustering Abstracts Final.ipynb](<Clustering Abstracts Final.ipynb>) refits the
   full clustering sweeps and saves the new catalog and selected memberships.
2. [Topic_analysis_Abstracts_v2 Final.ipynb](<Topic_analysis_Abstracts_v2 Final.ipynb>)
   currently analyzes **saved K-means k=6**, with two content-reviewed PE example
   papers per cluster in section 5, directly after the unnamed-cluster comparison.
   Existing companion exports for K-means k=5 and Leiden 0.45 remain available.

The twelve choices and twenty-four exact abstract passages are recorded in
`configs/abstract_final_cluster_examples.json`. The section verifies Scopus IDs,
the saved memberships, input hashes, metadata and excerpt offsets, and exports
the selections and diagnostics for all 2,056 papers under
`reports/abstract_final_cluster_examples/<review-checksum>/`. Centrality and
cosine silhouette use the existing abstract vectors. Cluster 4's HCA and Heinz
cases are explicitly boundary examples: they address deal partners and contractual
risk, but do not substantiate a general theme of finding capital partners.

The age notebook now uses the newer **1,531-paper corpus and K-means k=6**, as
described below; its selection is independent of these two notebooks.

`configs/abstract_final_input.json` pins the retained CSV under
`data/abstracts/final/5779f6345a8f5d18/` and the **2,056 × 1,024** vector matrix under
`outputs/abstract_embeddings/final/5779f6345a8f5d18/`. The vectors are exact original
rows; they have not been recomputed. Checksums and the approved per-paper decisions
are validated on load. Final notebooks never fall back to the old corpus.

Clustering reports/catalogs are separated by the new input fingerprint. Selected
assignment CSVs go to `outputs/abstract_final/`; topic evidence goes to
`outputs/topic_analysis_abstracts_final/`. The topic notebook currently displays
**K-means k=6**, with cluster IDs **0–5**. The older companion settings remain
Leiden **0.45** (four clusters) and K-means **k=5**. All memberships come from the
filtered corpus's catalog, not the original unfiltered results.

The original notebooks/data remain available for historical comparison.
**Elite Papers Abstracts**, PDF inputs, and their embedding pipelines are unchanged.
The reproducible one-time preparation script is
`scripts/create_final_abstract_notebooks.py`; it refuses to overwrite prepared
Final notebooks. Rerun the Final notebooks themselves for subsequent analysis.

### Experimental clustering: 1,531 papers

[Clustering Abstracts Test 1531.ipynb](<Clustering Abstracts Test 1531.ipynb>) clones
the Final clustering workflow for an experimental subset. It removes the **525**
protected title matches failing the journal rule or having any detected PE list
mention. It does not extend the last-third rule to title matches. The notebook
retains the same clustering settings and plots, with no corpus-comparison section.
The per-paper decisions, retained CSV, exact embedding rows and assignments are
saved under `outputs/abstract_clustering_experiments/title_journal_or_list/`.
The approved Final inputs and all PDF/Elite Papers Abstracts workflows remain
unchanged. The generator is `scripts/create_title_screen_clustering_notebook.py`.

### Current age analysis: one method per corpus

[Topic_analysis_Ages_v3 test Final.ipynb](<Topic_analysis_Ages_v3 test Final.ipynb>)
uses only **PDF Leiden 0.95**, **abstract K-means k=6 on 1,531 papers**, and
**citation Leiden 0.9**. `configs/topic-age-final-inputs.json` pins the abstract
experiment directory and clustering-manifest checksum. The six-cluster memberships
are loaded from its saved `partitions.npz`, with exact ID/order and cosine-silhouette
validation; no clustering is refitted in the age notebook.

The top eight two-word expressions are recomputed for the selected memberships
using the v2 ranking settings. All six abstract clusters appear in the four year
bins. Each stacked bar shows the distribution of year bins within one cluster;
year-bin shares sum to 100% per cluster, with its total displayed as `n`.
Tables and nine PNG/PDF/SVG figures are saved in
`reports/topic_age_analysis_v3_test_final/`; obsolete method figures are archived.
The preceding five-partition report, notebook and configuration are preserved in
the archive recorded by `previous_report_archive` in the current configuration.
PDF/citation selected memberships and Elite Papers Abstracts remain unchanged.
Run `.venv/bin/python scripts/validate_final_age_selection.py` to verify the
selected memberships, year-bin totals, source checksums and current figure set.

### Step-by-step abstract screening

Open [Abstract Screening.ipynb](<Abstract Screening.ipynb>) with **Python (batill)**
and run all cells. This is the current, transparent screening workflow for the
original 3,819-paper corpus. All matching, decision, sample and export code is
visible in the notebook. Its starting-point section records the upstream basic
cleaning and embedding provenance.

The agreed first rule uses the **full phrase “private equity”**, case-insensitively
and including hyphenated forms; standalone “PE” does not count. It drops **304**
papers with no match in either title or abstract, marks **1,483** title matches
safe under the working rule, and retains **2,032** abstract-only matches for
further evaluation. The stage-one dataset therefore has **3,515** papers.
The second rule applies **only to abstract-only papers**: their journal title
must match at least one editable keyword: `econom...`, `financ...`, `accounting`,
`manag...`, `market...`, `acquisition`, `merger`, `invest...`, `private equity`,
`business`, or `corporate`. Matching is case-insensitive and accepts
acquisition/merger plurals. It drops **1,087** papers and keeps **945**
abstract-only papers for the next steps.

The next rules apply only to these remaining abstract-only papers. First,
**184** are excluded because PE occurs as an item in a detected enumeration.
The documented heuristic requires three short noun-like items, or two inside
parentheses/with an explicit list cue. Ordinary two-subject PE/VC comparisons
do not trigger this rule. PE must itself be a list item with optional financial
modifiers; mention evidence is exported for review. Second, **188** more are
excluded because all PE mentions start in the final third by word count.
The overlap table records **53** papers satisfying both rules, counted only in
the first exclusion step. All **1,483** safe title matches remain, giving
**2,056** retained papers with aligned original embeddings, including **573**
abstract-only papers for further review. These lexical rules do not establish
substantive PE relevance.

The executed notebook displays **20 reproducibly sampled abstract-only papers
remaining after all filters**, with full titles, abstracts, metadata and
highlighted matches. Its final cell is a read-only experiment showing protected
title matches that would fail other rules: **503** journal failures, **28** list
failures (**525** unique papers), and last-third flags separately. Expand each
row for the full abstract and matched evidence. All title matches stay retained.
Future filters belong before sample preparation so the
inspection always reflects the end of the funnel. Each run saves the
complete paper-level audit, exact dropped/safe/review lists, the retained CSV,
stage-one and journal-stage decisions, exclusions for each subsequent rule,
journal keywords, per-mention context and word positions, rule overlaps, aligned original
embedding rows, an embedding row map, the 20 examples in CSV and
HTML, a notebook source snapshot and a checksum manifest under
`reports/abstract_screening/<run-id>/`. `review_notes.csv` is never overwritten;
its notes are not applied as new exclusions at this stage. Original sources remain
unchanged. The approved run is now pinned by the separate Final workflow above;
changing screening rules does not silently change its inputs.
The generator is `scripts/create_abstract_screening_notebook.py`;
regeneration clears notebook outputs.

### Earlier private-equity scope cleaning (inactive)

**Current analysis input: the original 3,819 abstracts and their unchanged
3,819 × 1,024 embedding matrix.** Subject filtering has been deactivated at the
user's request. `Clustering Abstracts.ipynb` and both abstract topic notebooks
explicitly select the original run; restart notebook kernels before rerunning.
The cleaning workbook and derived datasets remain optional archived work.
The following section describes that optional workflow, not the active corpus.

Open [Abstract Cleaning.ipynb](<Abstract Cleaning.ipynb>) with **Python (batill)**
and run all cells before the abstract analyses. This second cleaning stage screens
the original 3,819 cleaned abstracts for private-equity/buyout relevance. It
excludes chemical-industry material, identified trade news and **all explicit
venture-capital mentions, including mixed PE/VC papers**. Startup finance,
private placements, nonresearch document types and weak PE matches are held for
review. Clear PE evidence means a PE/buyout term in the title or at least two
occurrences in the abstract. These are editable lexical rules, not a manual
assessment of every record. Review is excluded from the analysis until accepted.

The initial strict run retains **1,404** records, excludes **1,359**, and withholds
**1,056** for review. Inspect **Abstract Cleaning.xlsx** for full abstracts,
matched evidence and all decisions. Edit `configs/abstract_scope_rules.json` for
global rules or `configs/abstract_scope_overrides.csv` for individual decisions
(`paper_id`, `record_sha256`, `decision`, `reason`), then rerun. The Excel file is
a review export; edits in Excel are not imported. Overrides reject unknown IDs,
duplicate IDs, missing reasons and hashes for changed source records.

Each reproducible bundle lives in `data/abstracts/private_equity/<scope-id>/`:
the schema-preserving `abstracts_clean.csv`, `scope_audit.csv`, `excluded.csv`,
`review.csv`, an override template, Excel workbook, exact embedding subset,
parent-row map and hash manifest. Original text, identifiers and all metadata
remain unchanged. Vectors are copied by EID after validating the original run;
no inference/download is needed. The existing strict embedding loader validates
the subset's metadata, checksums, IDs, counts, token audit and vectors.

The final cell activates `configs/abstract_analysis_input.json`. Run in order:

1. `Abstract Cleaning.ipynb`.
2. `Clustering Abstracts.ipynb` in a fresh kernel.
3. `Topic_analysis_Abstracts_v2.ipynb` or `Topic analysis Abstracts.ipynb`.

All three analysis notebooks resolve this explicit shared input rather than the
newest directory. Old clusters and topic names cannot be reused for changed
memberships. The legacy topic export now records the actual selected K-means and
Leiden counts; a fixed Leiden resolution need not retain six communities after
filtering. Historical PDF/citation/thesis reports remain unchanged. The fixed
five-cluster age-v2/v3 workflow keeps the original abstract input by default;
to update it, set `abstract_input` to `active` in its configuration and explicitly
select five-cluster partitions from the new catalog first.

The implementation is `src/batill/abstract_scope.py`; the notebook generator is
`scripts/create_abstract_cleaning_notebook.py`. Generation clears notebook outputs.
The Excel writer is already installed here; on another environment install
`pip install -e '.[cleaning,notebook,analysis,graph]'`.

### Original abstract corpus and embedding run

The separate abstract analysis starts from the original Scopus export at
`data/bibliography/All_private_equity_bib.csv` (4,583 records, 45 columns).
Open [ngramm emb.ipynb](<ngramm emb.ipynb>) with the **Python (batill)** kernel and run all
cells. The notebook contains the complete cleaning implementation, editable input
and output paths, result previews and integrity checks; it does not import the
cleaning script. The original command-line version remains available:

```bash
.venv/bin/python scripts/clean_abstracts.py
```

The original CSV is preserved. Outputs are in `data/abstracts/`:

- `abstracts_clean.csv`: 3,819 unique cleaned abstracts with Scopus EID,
  original record number, title, authors, year, journal/source title, DOI, URL,
  publication language, document type and whitespace-based abstract word count.
  The `embedding_text` column combines the normalized title, a blank line, and
  the cleaned abstract as one input for vector embedding. An empty title falls
  back to the abstract alone. The original `title` and `abstract` columns remain
  available separately for inspection and later n-gram analysis. The notebook
  prepares aligned `embedding_paper_ids` and `embedding_texts` lists; embedding
  inference is not run yet.
- `cleaning_audit.csv`: one entry per source record, including exclusion reasons,
  the retained EID for repeated text, and the exact removed publisher notice.
- `review.csv`: 86 retained records requiring attention, with explicit flags.
- `cleaning_summary.json`: counts, cleaning rules, publication-year cutoff and
  the source SHA-256 hash.

Cleaning excludes 695 missing-abstract placeholders, one entry consisting only of
a publisher notice, five records marked Retracted, one future-dated publication,
44 repeated DOIs, and 18 repeated exact cleaned abstracts. These are mutually
exclusive audit reasons, assigned in this order: missing/notice-only abstract,
retracted, future year, duplicate DOI, duplicate text. Among eligible records,
retain the first source occurrence per nonempty normalized DOI and unique cleaned
text. DOI matching ignores capitalization and common DOI URL prefixes. Missing
DOIs are allowed; rejected records do not reserve a DOI or text. Text deduplication
does not assert that every repeated abstract describes the same work;
the audit preserves the link to all excluded records. Publisher copyright and
licensing suffixes are removed from 3,036 source records and saved in the audit.
HTML entities, known formatting tags, whitespace and invisible text artifacts
are cleaned while preserving case, punctuation, numbers, scientific comparisons,
stopwords and word order. No stemming, embeddings or n-gram analysis is run yet.

All 36 short abstracts remain, without length warnings. Different abstracts
sharing a title remain flagged, as do errata and invalid publication years.
Errata are retained because they are corrections rather than retractions.
Future means a publication year later than the current calendar year (2026 for
this run); the cutoff is recorded in the summary. Retraction filtering uses the
CSV document-type field, without an external retraction lookup. Review the
remaining flags before defining the final analysis sample. There is no subject
or language restriction at this stage. Publication language is not a reliable label for the abstract's language;
non-English publications can have English abstracts. Keywords, references,
affiliations and other export metadata are omitted from the clean corpus but
remain available in the original CSV. Optional `--input` and `--output-dir`
arguments support other exports with the same Scopus schema.

Sections 7–9 of `ngramm emb.ipynb` use **BAAI/bge-m3** through Sentence Transformers,
replacing the earlier SPECTER2 implementation. Select **Python (batill)**; the
existing `.venv` already contains the required packages. The current configuration
is `RUN_EMBEDDING = True`, `RUN_MODE = "full"`, and batch size 8, processing
all 3,819 cleaned papers. The earlier benchmark used a reproducible sample of
50 distinct papers with seed 42; sample mode remains available.

The model receives the complete `embedding_text` (title, blank line, abstract).
The runner audits all corpus lengths using BGE-M3's tokenizer, including special
tokens, against the 8,192-token limit. It refuses overlong selected inputs;
there is no truncation or chunking. Each run saves its selected inputs,
1,024-dimensional normalized vectors, aligned paper IDs, full-corpus length
audit, batch timings, model revision, input checksum, and package versions under
`outputs/abstract_embeddings/bge_m3/<run>/`. Download/loading, warm-up and the
measured embedding run have separate timings. The full-corpus time estimate is
only a throughput extrapolation. The earlier SPECTER2 diagnostics and separate
environment are preserved, but this notebook no longer uses them.

The 2026-09-30 sample run successfully embedded **50 papers in 13.13 seconds**
on the Mac GPU (MPS, float32, batch size 8). First-run model download/loading took
104.03 seconds and warm-up took 5.24 seconds; total time was
129.53 seconds. All 3,819 title-and-abstract inputs fit (maximum 1,973 tokens).
The saved matrix has shape 50 × 1,024, with finite, normalized vectors and verified
paper-ID alignment. The notebook pins the tested model revision. Results are in
`outputs/abstract_embeddings/bge_m3/20260930T174040_070486Z_sample_29fced1c/`.

The full-corpus run then embedded **all 3,819 papers in 1,185.11 seconds
(19 minutes 45 seconds)**; total runtime was 1,196.76 seconds (19 minutes
57 seconds). The matrix has shape 3,819 × 1,024. Finite values, unit norms,
complete source-text preservation, unique paper IDs, and row alignment were
verified. No inputs were truncated or split. Full results are in
`outputs/abstract_embeddings/bge_m3/20260930T174845_508054Z_full_3c498a9d/`; `embeddings.npy` is aligned with `paper_ids.csv` and
`selected_papers.csv`. The executed notebook contains the run output.

### Clustering the abstracts

Open [Clustering Abstracts.ipynb](<Clustering Abstracts.ipynb>) with **Python
(batill)** and run top to bottom. It loads the active subject-filtered corpus
and aligned vectors from `configs/abstract_analysis_input.json`. Without this
configuration it uses the original 3,819 × 1,024 BGE-M3 run above. The notebook
deliberately does not discover a run by modification time.
The loader checks the source checksum, complete corpus coverage, saved row/Scopus
EID alignment, exact text and metadata, token audit, matrix shape and unit norms.
It rejects samples and incomplete runs; no inference or download is required.

The workflow includes cosine average-linkage hierarchy, neighbours, PCA/UMAP,
embedding-based Leiden and K-means k=2–15 with the original five-seed and
five-subsample stability checks. All clustering and silhouettes use the original
normalized embeddings. Optimal leaf ordering is off by default to reduce the
cost for thousands of records; hierarchy merges are unchanged. Full-data pairwise
diagnostics need several hundred MiB, and repeated K-means fits can take minutes.
Scopus metadata replaces PDF filenames and segment diagnostics throughout.
The user-selected topic interpretation uses **six clusters per method**. The final
text-clustering cell exports `kmeans_results[6]` and the six-community embedding
Leiden fit at **resolution 0.5 (seed 42)**, preserving original labels and Scopus
IDs. This resolution already occurs in the original sweep. The export is under
`outputs/abstract_text_clusters/<input-id>/six_clusters/` and rejects stale fits.
The PDF exclusions and reviewed PDF citation graph do not apply to this corpus;
independent citation comparison requires an aligned abstract-corpus graph.

### Topic interpretation of abstract clusters

The historical results described below use the original 3,819 abstracts. Current
notebooks read the active input and require new clustering exports for that
corpus. Counts, memberships, terms and proposed names will change after filtering.

[Topic analysis Abstracts.ipynb](<Topic analysis Abstracts.ipynb>) is copied from
the PDF topic workflow and adapted to all **3,819 title-and-abstract records**.
It loads the source notebook's exported memberships and does no clustering.
Run the source setup/loading, embedding-Leiden graph/sweep, K-means sweep and
six-cluster export first; after that, the topic notebook runs independently.
K-means IDs stay 0–5 and embedding-Leiden IDs stay L1–L6. There are no PDF exclusions,
PDF identities or citation partitions in this workflow.

The notebook computes c-TF-IDF, per-paper TF-IDF contrast and inside/outside term
coverage. It saves central, boundary and seeded random examples with full cleaned
abstracts, exact excerpt offsets, titles, DOIs, years and journals. It displays
six names separately for each method and compares the two saved partitions.
`configs/abstract_topic_labels.csv` contains the **AI-assisted analyst proposals**,
their rationales/caveats and blank reviewer fields. These are interpretive inputs,
not names generated by a deterministic naming algorithm. Each name is checked
against both membership IDs and source text; changed groups/text cannot silently
inherit an old name. The notebook preserves this proposal file.

Outputs are in `reports/topic_analysis_abstracts/<input-id>/<assignment-checksum>/`:
`named_cluster_profiles.csv`, `named_memberships.csv`, `cluster_review_cards.md`,
`term_scores.csv`, `review_papers.csv`, overlap counts and a run manifest. The saved
six-group results include weak/mixed groups and an industry-deal news group;
names are provisional and do not establish six equally coherent research topics.

For a reproducible handoff, include both notebooks, `src/batill/`, `pyproject.toml`,
the proposal and constraints files, `data/abstracts/abstracts_clean.csv`, the complete
saved abstract embedding run, the partition export and its reports. **`outputs/`
is ignored by Git**, so include those inputs explicitly. Install into Python 3.12
and select that environment as the notebook kernel:

```bash
python -m pip install -c configs/abstract_topic_constraints.txt -e '.[analysis,graph,notebook]'
python -m ipykernel install --user --name batill --display-name 'Python (batill)'
```

The constraints record the core versions used here, not a complete environment
lock or a guarantee of identical new fits across operating systems. The saved
assignment export is the reference for reproducing the interpretation. Running
the topic notebook from that export needs no model download/inference, external
API, source PDF or citation data. Its manifest records the input, assignment,
proposal, code and output checksums plus software versions.

### Publication ages of the fixed topic clusters

Open [Topic analysis Ages.ipynb](<Topic analysis Ages.ipynb>) with **Python
(batill)** and run all cells. It uses the existing named membership exports of
all three topic notebooks: both PDF partitions, both abstract partitions and the
direct-citation partition. It preserves every original assignment and name,
using the refreshed 292-PDF text partitions and retaining the three unassigned citation isolates.
No clustering, embedding inference, PDF reading or external API call is required.

The notebook calculates every paper's age using the **fixed reference year 2026**.
PDF/citation years come from the saved reviewed OpenAlex work metadata; PDF hashes
join through the reviewed canonical-work map. Abstract years come from the saved
Scopus metadata. All 292 canonical PDFs, 3,819 abstract records and 292 citation works
have valid publication years. Missing or invalid dates fail explicitly. A
sensitivity table repeats the age and period-share comparisons with the saved
bibliographic/PDF query years, which differ from OpenAlex for 112 canonical works.

Results in `reports/topic_age_analysis/` include:

- `paper_ages.csv`: every original membership and name with title, identifiers,
  primary/alternative years, provenance and age.
- `cluster_age_summary.csv`: counts, mean/median ages, quartiles and year ranges.
- `period_shares.csv` and `annual_shares.csv`: counts and within-period topic
  shares, with explicit assigned/unassigned denominators and zero-count cells.
- `period_changes.csv`: share changes from 2000–2009 to 2020–2026 in percentage
  points; `year_sensitivity.csv` and `alternative_period_shares.csv` repeat the
  comparison using alternative dates.
- `coverage.csv`, `run_manifest.json`, and ten figures in both PNG and PDF format.

The periods are through 1999, 2000–2009, 2010–2019 and 2020–2026. The final period
is incomplete. Topic shares describe the composition of these selected corpora;
age differences alone do not prove a field-wide shift. The notebook discusses
coverage, citation-age effects, provisional names and industry news in the
abstract corpus. Methods and corpora are kept separate throughout.

**Send `outputs/topic-age-reproduction.zip` to reproduce this notebook.** This
compact archive includes the executed notebook, all eight frozen CSV inputs,
code, pinned dependencies, reference tables and figures. After unpacking, run
inside `topic-age-reproduction/` with Python 3.12:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r configs/topic-age-requirements.lock
.venv/bin/python scripts/reproduce_topic_ages.py --check
```

The checker executes all notebook code cells in a fresh process with network
connections blocked, then requires exact SHA-256 equality for all eight generated
CSV tables. It also verifies frozen inputs and notebook/code/config sources.
Plot rendering and environment metadata are not compared byte for byte. The
tested environment is Python 3.12.7 on macOS Apple Silicon; other operating systems
are not certified. Installation needs internet access; execution runs offline.
For interactive use, register the environment with
`.venv/bin/python -m ipykernel install --user --name batill-topic-ages` and select
it in your notebook editor.

Rebuild the compact archive after a successful check with
`.venv/bin/python scripts/build_topic_age_bundle.py`. The separate larger
`topic-analysis-reproduction.tar.gz` reproduces the upstream topic interpretations;
the compact age bundle does not require it. Implementation lives in
`src/batill/topic_age_analysis.py`. Regenerating the notebook with
`scripts/create_topic_age_notebook.py` clears display outputs. Use
`scripts/reproduce_topic_ages.py --record-reference` only after deliberately
reviewing an analysis change, never to bypass a failed reproduction check.

### Verbal complexity

Open [Verbal_Complexity.ipynb](Verbal_Complexity.ipynb) with the **Python (batill)**
kernel and run all cells. It computes automated Gunning Fog and Flesch Reading Ease
scores for each of the 292 canonical works from the same cleaned Marker prose.
Higher Fog means harder prose; higher Flesch Reading Ease means easier prose.
Every paper uses the publication
year from the saved OpenAlex metadata. This dates the indexed work, which can
differ from the available PDF version. Time bins are through 1999 inclusive, then
calendar decades with the final bin labelled 2020–2026. Scores, OpenAlex year provenance and
block audits, exact scored texts, summaries, and PNG/PDF plots are saved under
`reports/verbal_complexity/`. No API calls, PDF reading, or embedding inference are needed.
`paper_scores.csv` includes both scores and their component counts;
`period_summary.csv` includes statistics for both measures. Separate plots are
saved as `fog_by_period` and `flesch_reading_ease_by_period` in PNG and PDF format.

For a fresh environment, install `.[readability,analysis,notebook]` and run
`python -m nltk.downloader -d .venv/nltk_data cmudict` once. All analysis code is
contained in the notebook. Its source generator is
`scripts/create_verbal_complexity_notebook.py`.
Regenerating the notebook clears its executed outputs.

The current report is [the editable Word chapter insert](reports/thesis/thesis_sections.docx).
[Thesis_analysis.ipynb](Thesis_analysis.ipynb) is the executed, offline runner and
[the evidence guide](reports/thesis/README.md) maps its claims to tables, plots and code.
It uses **292 canonical works**: 318 stored vectors minus 24 scope exclusions,
then two reviewed duplicate-version pairs collapsed. Both text and citation models
are fitted on this canonical selection; citation isolates remain unassigned.
The older clustering notebooks and 318-paper topic report are exploratory records.
The final citation graph comes from reviewed OpenAlex retrieval plus page-supported
manual links. Existing vectors and graph caches suffice; no API key or inference is
needed to reproduce the thesis experiments.

## Repository layout

```text
Pipeline.ipynb              Documented extraction, preparation and embedding stages
Clustering PDFs.ipynb       Current PDF embedding and citation clustering analysis
archive/
  Clustering.ipynb          Outdated clustering notebook, preserved for reference
Citation_graph.ipynb        Semantic Scholar matching, citation retrieval and graph export
configs/
  pipeline.json             New-corpus configuration
src/batill/
  pipeline.py               Public stage functions and output validation
  extraction.py             PDF discovery, extraction, caching and readable exports
  documents.py              Structured blocks, text filtering and title inference
  embedding.py              Input segmentation, inference and vector aggregation
  parallel.py               Independent GPU workers and validated result merging
  clustering.py             Cosine hierarchy and aligned dendrogram/heatmap
  citation_graph.py         Reviewed paper matching and selected-paper citation graph
  semantic_scholar.py       Rate-limited API client with reusable response cache
  attention.py              Optional exact SDPA adapter for grouped-query attention
  storage.py                Hashing and atomic output writes
  __main__.py               Command-line interface
tests/                     Offline synthetic integrity and numerical tests
data/
  pdfs/                     Source PDF corpus
  bibliography/             Original bibliography exports; not required by this pipeline
outputs/                   Generated files, excluded from version control
```

## Installation

Use a Python 3.10+ environment. For the existing development environment:

```bash
conda activate ml
python -m pip install -e '.[extract,embed,notebook,test]'
```

For a new machine, install a PyTorch build appropriate for its hardware before
installing the project. Dependencies are not installed automatically by notebooks.
Select the same environment as the notebook kernel. `-e` makes edits to `src/batill/`
available without reinstalling the package.

Extraction uses Marker 2.0.0 in fast mode with OCR disabled. It needs native PDF text
and a layout model; it does not need Docker, GROBID or a formula-transcription server.
The first run can download layout weights. Embedding weights are downloaded separately
by Hugging Face when inference is first requested; they are not stored in this repository.

## Run the pipeline

Open **Pipeline.ipynb** and run its cells in order. The default configuration is
`configs/pipeline.json`. Inspect extraction and exact model inputs before inference.
`RUN_EMBEDDING = False` is intentional on the development Mac. Set it to `True` only
on the machine where inference should run.

The equivalent commands are:

```bash
python -m batill extract --config configs/pipeline.json
python -m batill prepare --config configs/pipeline.json
# Execute this only on a machine with sufficient inference resources:
python -m batill embed --config configs/pipeline.json
python -m batill validate --config configs/pipeline.json
```

A complete local run is also available:

```bash
python -m batill all --config configs/pipeline.json
```

This last command **does run model inference**. The notebook is preferable when
manual inspection between stages is required. No embedding computation is performed
on import, during extraction, or during tokenizer/input preparation.

## Use a new folder of PDFs

1. Add PDFs to `data/pdfs/`; subfolders and `.PDF` extensions are supported.
2. Choose `configs/pipeline.json` in the notebook or command line.
3. Run extraction, inspect the readable outputs and prepare the selected papers.
4. Run inference on a suitably resourced machine and validate the exported matrix.

The output is `outputs/corpus/`. Previous sample PDFs and generated results have been archived outside the project.
Custom input/output folders can be specified in a configuration file in `configs/`.
Paths are resolved relative to the project root. The pipeline refuses an empty input
folder and checks that PDFs still match the extraction report before preparation.
Bibliographic matching is not automatic: paper IDs are SHA-256 hashes of PDF content,
and titles are extracted or inferred from first-page headings. Inspect title quality.

Preparation normally stops if a selected paper has no retained text. The notebook
explicitly sets `SKIP_EMPTY_DOCUMENTS = True` and displays these exclusions, allowing
the remaining papers to proceed. The Python equivalent is
`pipeline.prepare(config, skip_empty=True)`. Excluded PDFs are preserved; their IDs,
filenames and reasons are recorded in `prepared/preparation_report.json` and the
prepared manifest. Nonempty extractions still need quality inspection.

## Section-based preparation used in the final analysis

From the project root, using the `ml` environment, run:

```bash
python -m batill.prepare_sections --skip-empty
```

This reuses `outputs/corpus/extraction/` and writes a separate corpus to
`outputs/corpus_sections/prepared/`. It loads only the tokenizer, never embedding
weights, and does not run extraction or inference. Existing prepared inputs and
embeddings remain unchanged. `--skip-empty` explicitly records/excludes empty
extractions; omit it to stop on them. `--max-tokens 32768` sets the input cap.

Every paper uses the same section strategy, even if the entire paper fits:

- Front matter, abstract and introduction form the opening group.
- Each subsequent numbered main section forms a group; subsections stay with it.
- Every input includes the paper title and task instruction.
- Oversized groups split at block boundaries, then character boundaries only when
  necessary, preserving source offsets and all retained text.

Inspect `prepared/section_review.csv` and `prepared/readable/` before embedding.
The table lists section names, group IDs, token counts, page ranges and review flags.
Repeated group IDs mean a section needed multiple inputs. Numbered headings take
priority over inconsistent extracted heading levels; unnumbered papers use heading
levels as a fallback. Missing introductions/boundaries are flagged. Papers do not
all share the same structure, so this heuristic needs manual review. It does not
infer missing prose or guarantee that all unnumbered headings are recognized.

The downstream embedding code supports these plans unchanged. Token-weighted
aggregation remains the same to isolate the effect of segmentation; sections do
not receive equal weight. The retained strategy does not guarantee that length bias
disappears. The strategy is recorded in each input's configuration and checksum.
To analyse the new embeddings later, set `OUTPUT_DIR` in `Clustering PDFs.ipynb` to
`ROOT / 'outputs' / 'corpus_sections'` after that corpus has been embedded.

## Use four GPUs on one machine

After input preparation, a machine with four visible CUDA GPUs can run:

```bash
python -m batill embed --config configs/pipeline.json --workers 4
```

In the notebook, set `GPU_WORKERS = 4` and enable `RUN_EMBEDDING` only on that machine.
Each process loads its own model and sees exactly one GPU. Complete papers are assigned
deterministically, longest first, to balance estimated total input tokens. All segments
of a paper stay with the same worker. This is corpus parallelism: GPU memory is not
pooled, and each paper/segment must still fit on one GPU.

Workers write to isolated `parallel/<partition-id>/worker-N/` folders. The parent waits
for all processes and validates all results before publishing the canonical matrix in
the original paper order. A failure stops publication and retains completed paper caches;
rerunning the same command resumes them. Empty workers load no model. Use only one
active invocation per output directory.

`embeddings/last_run.json` contains worker statuses and elapsed times for a parallel run;
per-paper timings remain in the worker folders. Runtime provenance stays attached to each
paper. On an older CUDA GPU, use the same explicit attention options as the validated
single-GPU configuration:

```bash
python -m batill embed --config configs/pipeline.json --workers 4 \
  --attention batill_sdpa_repeat_kv --sdpa-backend efficient
```

Partitioning, concurrent writes, GPU visibility assignment, merge validation and failure
handling have synthetic tests. Four-GPU inference and its speedup have **not** been run.

## Representation and filtering

- Exact duplicate PDFs are identified by their content hashes and skipped.
- Reading order, headings, pages and source block IDs are retained in structured JSON.
- Detected page headers, footers and reference sections are excluded from model inputs.
- Detected equation blocks are removed; marked inline math becomes `[FORMULA]`.
  This does not guarantee removal of unmarked formula fragments.
- Tables are retained as text. Their reading order and completeness need inspection.
- The title and an economic-topic instruction prefix each model input.
- A paper fitting the token limit is encoded as one input. Longer papers are segmented
  by section/block, with character ranges recorded for oversized blocks. No retained
  text is silently truncated.
- Each segment is encoded independently. Its content-token count weights the mean
  segment vector, and the resulting paper vector is L2-normalized.

The model and immutable revision are recorded in the configuration and every result.
See the [Qwen3-Embedding-4B model card](https://huggingface.co/Qwen/Qwen3-Embedding-4B)
for the underlying model. This pipeline's aggregation and instruction are modelling
choices to evaluate for the thesis; benchmark performance does not establish thematic
validity for private-equity research.

## Output contract

For either configured output folder:

```text
extraction/
  report.json               Current PDF selection, status and per-paper timings
  raw/                      Cached converter output
  documents/                Canonical structured documents
  readable/                 Formula-filtered prose for inspection
  timings/                  Timestamped extraction invocations
prepared/
  model.json                Pinned model configuration
  manifest.json             Exact ordered selection of input files
  preparation_report.json   Selected/usable counts and empty-paper exclusions
  inputs/                   Model strings, tokens and source ranges
  readable/                 Exact model input strings as text files
embeddings/
  embeddings.npz            float32 matrix, PDF hashes and optional bibliography keys
  manifest.json             Row identities, model settings and execution provenance
  papers/                   Resumable per-paper and per-segment vectors
  last_run.json             Last inference invocation, cache status and elapsed times
```

Load the final representation with:

```python
import json
import numpy as np
from pathlib import Path

folder = Path('outputs/corpus/embeddings')
manifest = json.loads((folder / 'manifest.json').read_text())
with np.load(folder / 'embeddings.npz', allow_pickle=False) as saved:
    vectors = saved['embeddings']  # (number of papers, 2560)
    paper_ids = saved['pdf_sha256']
```

Keep the manifest with the matrix. `pipeline.validate_outputs(...)` checks input
identity, archive checksum, row alignment, dimensions, finite values and aggregation.
All output references are relative. Files produced on another machine can occupy
exactly the same output directory without modifying downstream code. Their provenance
continues to identify the actual execution machine; imported vectors are not relabelled
as locally computed.

## Caching, resources and recovery

Extraction caches include the PDF hash, extractor version and options. Prepared-input
hashes include the model revision, instruction, filtering settings, text and segmentation.
Changing these choices creates new input identities. Manifests select the current inputs;
stale cache files are never included merely because they remain on disk.

Inference saves each successful paper independently and stops on failure. Reruns validate
and reuse matching per-paper vectors. If every vector is already present, the final
matrix can be assembled without loading model weights. Cached reads are not inference
speed measurements. Corrupt or mismatched results are rejected, not silently overwritten.

The default runtime uses CUDA and FP16. Runtime choices are explicit and stored separately
from semantic input identity. The optional `batill_sdpa_repeat_kv` attention implementation
repeats grouped key/value heads before exact PyTorch SDPA, permitting use of its efficient
kernel on older CUDA GPUs. It is a general backend option, not a dependency on a cluster.
The unused generation KV cache is disabled for embedding inference. CPU/MPS devices can
be selected explicitly, but long-context memory feasibility must be tested; there is no
automatic fallback or context reduction.

## Tests and research status

```bash
python -m pytest -q
```

Tests use synthetic data and tiny CPU tensors: they do not download models or execute
notebooks. They cover duplicate detection, extraction-cache invalidation, lossless input
coverage, attention equivalence, portable results, corruption checks and interrupted-run
recovery. See `docs/validation.md` for the nine-paper integration test.

The sample validates the software path, not the scientific usefulness of the map.
Inspect nearest neighbours and assess thematic relevance on the larger corpus before
selecting clustering settings. `Clustering PDFs.ipynb` loads validated paper-level vectors
from `outputs/corpus_sections/` by default; set `OUTPUT_DIR` to `outputs/corpus/`
to use the full-paper run instead. Run its setup/loading cells, then the hierarchical section
to inspect up to 15 clusters using `MAX_HIERARCHY_CLUSTERS`. The collapsed average-linkage
dendrogram, annotated cluster similarity heatmap and representative-title table share
the same C1-style labels. Heatmap cells average paper pairs; the diagonal excludes
self-pairs and is undefined for singleton clusters. Tied cut heights may produce fewer
clusters. The full paper-level plot is optional. Colour limits adapt to displayed values
unless explicitly fixed; compare numerical values as well as colours. Set `OUTPUT_DIR`
to `outputs/corpus_sections/` to analyse the section-based run.
PCA, UMAP and K-means are optional later sections. Old abstract-based
outputs have been cleared; the notebook has not been executed on the new corpus.
Install analysis dependencies if needed with `python -m pip install -e '.[analysis]'`.

`Clustering PDFs.ipynb` additionally includes a copyable hierarchical review report:
all titles for clusters with 1–2 papers, and three representative titles plus the
paper count for larger clusters. Small clusters are flagged for review, not removed.
The later analysis runs PCA/UMAP, then Leiden, then the unchanged K-means section.
Leiden includes PCA/UMAP views coloured by a selected resolution on the same coordinates.
Install `python -m pip install -e '.[graph]'`
in the notebook environment. It builds a positive cosine-weighted union kNN graph,
reports connectivity, sweeps resolution with multiple seeds and displays candidate
partitions with 2–15 communities, size diagnostics and representative titles.
Resolution controls granularity, not an exact count; singletons can still occur.
The graph is based on embedding similarity, not citations. The original
`archive/Clustering.ipynb` is preserved as the outdated version.

## Compare citation and text clustering

The final **section 7 of `Clustering PDFs.ipynb`** adds direct-citation Leiden.
Run the existing loading, UMAP and embedding-Leiden cells, then the new citation
section. K-means cells are needed only when you select K-means for comparison.
The new cells were left unexecuted for the user; existing computational cells and
outputs were preserved. Supporting functions were tested independently, including
an in-memory check on the reviewed graph.

The citation model uses the binary undirected union of directed citation links,
with one node per unique work (292 currently). Three citation isolates remain
unassigned; connected components and coverage are reported. The resolution sweep
uses multiple seeds, saves all memberships and reports community sizes and seed
ARI. Change `CITATION_RESOLUTION` to inspect a candidate. The initial 1.0 is not an
automatic recommendation. Representatives have the most links within a community.

Choose `TEXT_COMPARISON_METHOD = 'embedding_leiden'` or `'kmeans'`, then adjust
`TEXT_LEIDEN_RESOLUTION` or `TEXT_KMEANS_K`. The two UMAP panels reuse **exactly the
existing coordinates**, using canonical PDF representatives. ARI/AMI and an overlap
heatmap compare unique, citation-assigned works. Colours are matched by overlap
for display only. The optional `RUN_CITATION_SENSITIVITY` cell compares the selected
fit to an API-only graph without manual citation additions.

Outputs are written when the user runs the cells under
`outputs/citation_clustering/<selection-id>/with_manual/` (or `api_only/`), with
resolution/comparison subfolders. They include assignments, diagnostics, interactive
UMAP and overlap HTML plots, underlying coordinate/count tables, configuration and
input checksums. Implementation: `src/batill/citation_clustering.py`; tests:
`tests/test_citation_clustering.py`. Citation clustering is independent of the
embeddings; UMAP is only a display. A combined graph is a possible later extension,
not a substitute for comparing the two separate results first.

## Citation graph with OpenAlex (recommended retrieval route)

To explore the actual directed network, open
[citation graph actual.ipynb](citation%20graph%20actual.ipynb). It runs offline on
the reviewed snapshot and displays all 292 works and 2,539 citation links, including
isolates. It includes a static overview, an interactive graph with citation arrows
and evidence on hover, and a configurable view of one paper's incoming/outgoing
citations. Figures, standalone HTML, GraphML and tables are saved separately under
`outputs/citation_graph_actual/9ad47e071ee12c2f/`. A → B means A cites B;
node size reflects incoming citations within the selected corpus.

Open **Citation_graph_OpenAlex.ipynb**. This complete notebook was executed on the
current 294-PDF selection. It retrieves DOI batches with their reference lists,
then builds the induced directed graph. The Semantic Scholar notebook below is
preserved in full as an alternative; its slow reference retrieval is no longer
needed to use the OpenAlex graph.

**Use section 7 and the `reviewed/` subfolder for the next clustering step.**
Manual review now resolves all five held PDF identities: the 294 selected PDFs
map to **292 unique works** (two duplicate pairs), with **2,539 internal citation
links**, including **65 verified additions from local PDF references/footnotes**.
The largest weakly connected component contains 286 works. No PDFs or embedding
rows were removed. `reviewed/pdf_to_work.csv` records the mapping; original API
outputs remain in the parent folder. See `reviewed/report.md` for decisions and
sources. P301's 2013 NBER PDF is mapped to its verified 2014 published version;
reference-version differences remain possible. The eight manually supplemented
bibliographies are **partial**, especially for external references needed for
bibliographic coupling. Their coverage limitations remain explicitly labelled.

Manual code: `src/batill/citation_curation.py`; decisions:
`configs/citation_graph_manual_review.json`. The notebook includes an optional
candidate/evidence reproduction cell; approval remains a recorded manual step.
The final graph and full notebook have been executed using cached/local data.

The preserved API baseline has **289 approved work identities, 2,396 internal citation
edges, and 279 PDFs in the largest weakly connected component**. Eight approved
records have empty provider reference lists. Five PDF identities remain held:
P020/P308 and P154/P155 are two duplicate pairs; P301 has duplicate OpenAlex
records for its DOI. All 294 PDFs remain in the node table. Missing links and
isolates must be interpreted alongside these coverage limitations.

Results and a compact report: `outputs/citation_graph_openalex/9ad47e071ee12c2f/`.
The general path is `outputs/citation_graph_openalex/<selection-id>/`. Key outputs
are `nodes.csv`, `edges.csv`, `all_references.csv`, `citation_graph.graphml`,
`adjacency.npz`, `adjacency_order.csv` and `report.md`. A → B means A cites B.
Outside reference IDs are retained for later bibliographic coupling. Reference
lists from different versions are not pooled; verified DOI versions may identify
a cited target, and these edges are labelled.

Implementation: `src/batill/openalex.py`; offline tests: `tests/test_openalex.py`.
The notebook shows selection, optional key entry, recorded review decisions,
retrieval, export, coverage and individual-link inspection. Successful responses
are cached under `outputs/citation_graph_openalex/api_cache/`; repeat runs reuse
them. API errors stop promptly without a long retry loop. The completed run used
no API key. If quota/authentication requires one later, enter an **OpenAlex** key
at the hidden prompt or set `OPENALEX_API_KEY`; do not use a Semantic Scholar key.

The batch client can also run from the project directory:

```bash
python -m batill.openalex
```

The script retrieves/exports the graph; the notebook additionally displays the
results and writes the compact report. Install `.[citations,notebook]` if needed.
Identity corrections are documented and PDF-hash pinned in
`configs/openalex_doi_overrides.json`; PDF first-page evidence and pre-review
matching tables are in the run's `review/` folder. Existing approved Semantic
Scholar DOI metadata are reused offline when available; local bibliography DOIs
remain the first choice. Manual working-paper approvals follow the user's policy,
and their references may differ from the local PDFs. Preserve the raw cache and
review/input files with the thesis evidence for snapshot reproduction.

## Citation graph with Semantic Scholar

Open **Citation_graph.ipynb**, restart/select the `ml` kernel, and run its sections
in order. The notebook retains the **complete workflow for submission**: selection,
key entry, candidate matching, identity review, reference retrieval, graph export and
inspection. Section 3 defaults to reusing complete saved matches; set
`RERUN_MATCHING = True` to recompute them or apply changed overrides. No earlier
methodological stages need to be removed to simplify execution.

For the current completed matching run, the shorter resume path is sections
**1 → 2 → 3b → 4 → 5**; section 6 provides optional inspection. The key is entered
privately in section 2. Cached API responses are reused; PDFs and embeddings are
not recomputed. Dependencies: `python -m pip install -e '.[citations,notebook]'`.

All implementation code is included in `src/batill/`: `citation_graph.py` exposes
the individual stages, `semantic_scholar.py` handles API access, and
`citation_review.py` applies the saved review. `citation_pipeline.py` additionally
provides the same stages as a convenience wrapper for scripts. The workflow uses
the current hash-based exclusions and checks row alignment and reviewed identities. The current selection has 294 PDFs and 281
approved identities. Thirteen unresolved records remain present but do not supply
citation edges. The review is in `configs/citation_match_review.json`; 79 approvals
are explicitly flagged as versions of the same work whose reference lists may differ
from the local PDFs. The 184 automatic matches were not manually re-reviewed.
Review evidence: `outputs/citation_graph/9ad47e071ee12c2f/review/report.md`.

Enter the key at the hidden prompt, or define `SEMANTIC_SCHOLAR_API_KEY` in the
kernel environment. The client keeps the key in memory and does not persist it.
Completed responses are cached. After interruption, rerun the notebook; it will
reuse the cache. HTTP 429 retries wait 20 seconds, or longer when requested by the
server. Requests start two seconds apart and slow down to ten seconds apart after
rate limits, without automatic acceleration. Run only one retrieval at a time.

**A → B means A cites B.** The runner paginates outgoing reference lists and retains
internal edges between approved selected identities. It also accepts an exact DOI
alias when it identifies only one approved selected paper, allowing a reference to
its published edition to match an approved working-paper identity. Conflicting ID/DOI
matches are reported and skipped. Missing identities, malformed provider responses
and unresolved reference IDs remain explicit coverage limitations.

Outputs in `outputs/citation_graph/<selection-id>/` include `nodes.csv`, `edges.csv`,
`citation_graph.graphml`, sparse `adjacency.npz` and its row-order table, retrieval
audits and `run_manifest.json`. `all_resolved_references.csv` retains outside
references for later bibliographic-coupling analysis. The graph includes unapproved
and isolated nodes; these must not automatically be interpreted as topic clusters.

The active notebook includes all matching and review code; it has no dependency on
archived notebooks. Recovery snapshots remain outside the project in
`../batill_recovery/`. Changing corpus membership requires a matching run for that
new selection. Synthetic tests verify retrieval, graph direction, DOI aliases,
conflicts and export. This Semantic Scholar route is preserved for provenance. The completed final graph
uses the OpenAlex route and manual review; see `Citation_graph_OpenAlex.ipynb` and
the final thesis evidence guide above.

For submission, keep the full notebook, `src/batill/`, configuration files and the
relevant input/result evidence together. The API key must remain external. Cached
provider responses preserve the retrieval snapshot; a fresh API run can return
changed data. The repository's `outputs/` and bibliography/PDF ignore rules concern
version control, not whether those artifacts are needed to reproduce the analysis.

## Topic interpretation of the five PDF-text clusters

### Current corpus selection

`Clustering PDFs.ipynb` and `Topic analysis PDFs.ipynb` apply
`configs/analysis_exclusions.json` when loading validated PDF embeddings.
The current user-reviewed list excludes 7 papers following
scope/outlier review, 15 purely venture-capital papers and 2 standalone online
appendices, followed by two reviewed duplicate copies: **318 stored PDFs → 294
in-scope PDFs → 292 canonical PDFs for text clustering**. Filtering uses PDF hashes,
preserves manifest Paper IDs and keeps embedding rows aligned. Each loading cell
displays the full decision audit. Edit a record's `exclude` boolean to revise a
decision. The separate `configs/pdf_duplicates.json` records the two duplicate-copy
removals; no additional papers are automatically removed.

The PDFs and cached vectors remain intact; embedding inference is not needed.
Restart the notebook kernel and rerun setup/loading, clustering and projections.
Old clustering outputs were cleared, with notebook copies preserved under
`../batill_recovery/before_scope_filter_20260929_170237/`.

### Reuse the source notebook's assignments

`Topic analysis PDFs.ipynb` interprets **five K-means clusters and five
embedding-based Leiden communities separately**, using only the exported results
from `Clustering PDFs.ipynb`. It never fits, sweeps or merges clusters and never
loads direct-citation memberships. It retains 292 canonical PDFs, exactly as the text-clustering notebook does.
After the 24 scope exclusions, `configs/pdf_duplicates.json` removes two reviewed
duplicate copies before any clustering or projection. Canonical cached vectors
and original Paper IDs are preserved; embedding inference is not repeated. K-means IDs remain 0–4; embedding-Leiden IDs remain L1–L5.

Run with the **Python (batill)** kernel:

1. In `Clustering PDFs.ipynb`, run setup/loading, the embedding-Leiden graph and
   resolution sweep, and the K-means sweep. The five-group choice uses
   `kmeans_results[5]` and an explicit five-community embedding-Leiden resolution.
   The PDF-specific Leiden grid uses `np.arange(0.1, 1.25, 0.05)`: 0.10–1.20
   in steps of 0.05 (1.25 excluded). `SELECTED_LEIDEN_RESOLUTION`
   supplies the shared default for inspection, projections, export and comparison.
   For the 292-PDF corpus the choice is **0.95 (seed 44)**, which has the
   highest seed agreement among the tested five-community settings. The previous
   294-PDF setting, 0.875, now gives four communities.
2. Run **Export the selected five-cluster PDF-text solutions**, before the
   direct-citation section. This exports existing fits to
   `outputs/pdf_text_clusters/<selection-id>/five_clusters/` and records source
   settings, seeds, PDF IDs, vector fingerprints, checksums and original labels.
3. Run `Topic analysis PDFs.ipynb` from top to bottom. Missing exports, changed
   inputs, wrong cluster counts or incompatible identities stop the analysis;
   there is no reclustering fallback. The direct-citation section is not needed.

The two topic descriptors are pooled c-TF-IDF and mean per-paper TF-IDF contrast,
with inside/outside document coverage. Central, boundary and random papers include
verifiable source excerpts. The notebook displays a separate named table for each
method, records every named membership, and compares the two saved partitions.
Names are analyst proposals tied to membership checksums, with rationale and
reviewer fields; the notebook never overwrites the proposal CSV.

Evidence is saved under
`reports/topic_analysis_pdfs/<selection-id>/<assignment-checksum>/`.
`named_cluster_profiles.csv` contains the names and group diagnostics;
`named_memberships.csv` gives every paper's assignment and proposed name;
`cluster_review_cards.md`, `term_scores.csv` and `review_papers.csv` support review.
The displayed output folder identifies the exact run. A changed membership cannot
inherit an old name. The handoff implementation is in
`src/batill/pdf_topic_analysis.py`; the term/excerpt helpers remain in
`src/batill/topic_analysis.py` for other existing analyses.

`Topic_analysis_PDFs_v2.ipynb` is a separate exploratory expression viewer for
the historical **292 canonical-work thesis partitions** in `reports/thesis/evidence/`.
It can inspect a saved text, citation or hierarchy partition, or fit one new
K-means/text-Leiden setting on the existing vectors. Its helper is
`src/batill/topic_explorer.py`. Run it with the **Python (batill)** kernel from
the project directory. This viewer does not load the current named PDF-topic
exports or assign thematic names; its optional export writes a new folder under
`outputs/topic_analysis_2/` only when `SAVE_RESULTS = True`.

Immediately after the cluster comparison, section 5 provides two content-reviewed
papers for each saved PDF Leiden 0.95 cluster. The requested review prompts are
employment growth, LBO financing/process, going private/public, fund performance,
and health care/nursing homes. Each example includes a PE-relevance rationale,
two exact PDF passages with page references, centrality rank, cosine silhouette,
and a scope note. These are purposive qualitative examples; boundary cases and
dependencies between papers are explicit. The review is stored in
`configs/pdf_cluster_examples.json`; changed memberships or passages invalidate it.
This section automatically saves its ten examples, twenty passages, all-paper
diagnostics and manifest under `reports/pdf_cluster_examples/<review-checksum>/`.
Other partitions require a separate content review.

`Topic_analysis_Abstracts_v2.ipynb` recreates this expression viewer for all
**3,819 title-and-abstract records**. The selectable catalog includes K-means
**k=2–15** and all **18 Leiden resolutions (0.10–0.95)** from the abstract sweep.
Memberships are reproduced once using the source sweep's graph settings and
seed-selection rules, then cached with input checksums and package versions under
`outputs/abstract_text_clusters/<input-id>/explorer_catalog/`.
The default is `SOURCE = 'saved'`, `SAVED_PARTITION = 'leiden_r0.45'`:
**five-cluster Leiden at resolution 0.45**. Choose `kmeans_k5` for five-cluster
K-means. Leiden names include resolution to distinguish different partitions with
the same count. K-means IDs start at 0 and Leiden IDs at 1; all papers are included.
The workflow includes both term rankings, expression search, prevalence plots,
representative/boundary papers, full abstracts and supporting source fields.
Abstract term rankings require positive scores and at least two supporting papers
inside the cluster, without the PDF viewer's 10% cutoff. The notebook exposes
`MIN_CLUSTER_PAPERS` and `MIN_CLUSTER_FRACTION`; setting the fraction to `0.10`
restores the stricter rule. Coverage counts remain visible and exports record the
chosen threshold. `TOP_N = 16` displays up to 16 eligible expressions per cluster.
Its helper is `src/batill/abstract_topic_explorer.py`; optional exports go into
new folders under `outputs/topic_analysis_abstracts_v2/`.

`Topic_analysis_Citations_v2.ipynb` provides the same expression-exploration workflow
for the reviewed direct-citation graph. Its default is the **saved `citation_r0.9`**
partition: five communities, selected seed 44, 289 assigned canonical works and
three unassigned isolates. The catalog includes all 23 saved source resolutions
(0.10–1.20 in steps of 0.05). It imports existing seed memberships without refitting,
checks them against graph provenance and sweep diagnostics, and preserves assignment
snapshots under `outputs/citation_topic_partitions/<selection-id>/explorer_catalog/`.
Representatives use within-community citation degree; boundary examples use external
link fractions. Term evidence uses retained PDF text, excluding isolates from the
vocabulary and prevalence denominators. The default supports up to 16 expressions
with positive scores and at least two supporting community papers. Optional exports
under `outputs/topic_analysis_citations_v2/` preserve all memberships, including
isolates, and the selected resolution. The helper is `src/batill/citation_topic_explorer.py`.

Section 5, immediately after the cluster comparison, adds two content-reviewed
papers per saved citation 0.9 community. The requested strands are health care and
nursing homes (C1), management-buyout process and operating performance (C2),
financial reporting (C3), fund performance (C4), and practice management (C5).
Each paper has a PE-relevance rationale, two verified PDF passages, internal-degree
rank with ties, within-community link count and external-link share. Counts refer
to the reviewed corpus graph. The broader scope of C1 and the related practitioner
articles in C5's three-work component are explicit limitations of the interpretation.
`configs/citation_cluster_examples.json` binds the review to the exact graph,
memberships and texts. Evidence and all 292 works' diagnostics (including isolates)
are saved automatically under `reports/citation_cluster_examples/<review-checksum>/`.

`Topic_analysis_Ages_v2.ipynb` compares cluster composition within the same year bins
as verbal complexity: **2020–2026, 2010–2019, 2000–2009 and ≤1999**. It uses five
fixed saved partitions: PDF K-means k=5 and Leiden **0.95**, abstract K-means k=5
and Leiden **0.45**, and citation Leiden **0.9**. Every partition must have five
assigned clusters; no clustering or per-bin refitting is performed. PDF/citation
years come from the reviewed OpenAlex metadata, and abstract years from Scopus.
Each bin shows counts and percentages of its assigned papers across all five
clusters, including zeros. Citation isolates are audited separately; empty-bin
percentages stay undefined. Tables, paper-level year assignments, provenance and
PNG/PDF/SVG figures are saved under `reports/topic_age_analysis_v2/`.
Settings are in `configs/topic-age-v2-inputs.json`; the helper is
`src/batill/topic_age_v2.py`. This notebook uses the selected five-cluster snapshots,
including the PDF **0.95** export, rather than the historical PDF v2 explorer's
different thesis partitions.

`Topic_analysis_Ages_v3 test.ipynb` adds cluster names made from the **top eight
two-word expressions** under each v2 topic workbook's ranking and support rules.
The expressions label the tables, plot rows, paper inspection and exports; original
cluster IDs remain visible. Labels are computed once for the complete partition
and bound to its exact memberships, so they remain fixed across year bins.
For PDFs, the v2 workbook's ranking rules are applied to the age notebook's selected
five-cluster partitions instead of copying names from different historical fits.
Results and the label-evidence audit go to `reports/topic_age_analysis_v3_test/`.

### Earlier 318-paper topic analysis

The former `Topic_analysis.ipynb` is preserved as
`archive/Topic_analysis_318_papers.ipynb`, with its original saved outputs.
`reports/topic_analysis/` is its historical 318-paper baseline, not an input to
the current PDF topic notebook. The separate `Thesis_analysis.ipynb` concerns
canonical works and must not supply replacement PDF-level memberships here.

- [Compact findings](reports/topic_analysis/report.md)
- [Review cards for every cluster](reports/topic_analysis/cluster_review_cards.md)
- [Label proposals and blank reviewer fields](reports/topic_analysis/label_proposals.csv)
- [Settings, versions and artifact checksums](reports/topic_analysis/run_manifest.json)

These historical tables/plots remain in `reports/topic_analysis/`. Their names and
counts must not be attached to the current selected PDF clusters.

Source PDFs, bibliography exports, model caches and generated outputs are not
intended for public version control. Keep the required input and result evidence
with the thesis submission under the applicable access conditions. Historical test
files and temporary compute-transfer tooling remain outside the project; the
full-corpus vectors used in the final report have been retrieved.

### Reproduce all three saved-partition topic analyses

`Topic analysis Citations.ipynb` interprets the existing **direct-citation** Leiden
solution from `Clustering PDFs.ipynb`: resolution 1.0, selected seed 43, with reviewed
manual links. Its six communities contain 289 canonical works; three isolates
remain unassigned. Two PDF aliases do not count as additional works. Original
community IDs are preserved. The source notebook now exports the selected citation
fit to `outputs/citation_topic_partitions/<selection-id>/resolution_1/`.
The supplied snapshot was adopted from the existing saved results after checking
the historical graph hashes, authenticated labels and selected seed memberships.

The three topic notebooks perform no new clustering. PDFs use both saved
five-cluster embedding partitions; Abstracts use both saved six-cluster embedding
partitions for all 3,819 abstracts. Citations uses the six saved graph communities.
All compute descriptive c-TF-IDF, TF-IDF contrast, inside/outside coverage and
source excerpts. Citation representatives and boundary examples use graph links;
PDF/abstract representatives and diagnostics use their existing embeddings.
Citation isolates are excluded from term comparisons and naming, but retained in
the membership export. A small disconnected community is not necessarily a broad
research theme. Retained source text can include tables, references and extraction
artifacts; inspect the displayed passages and full memberships when reviewing names.

Names are **recorded AI-assisted analyst proposals**, not generated anew on rerun.
Their explicit inputs are `configs/pdf_topic_labels.csv`,
`configs/abstract_topic_labels.csv` and `configs/citation_topic_labels.csv`.
Each name is bound to its original method and membership; abstract/citation names
also bind the exact retained texts. PDF inputs additionally pass the partition
snapshot's prepared-text checksum validation. Regex locates words and excerpts;
it does not choose the names. Reviewer fields are initially blank. Changing a
reviewer field does not automatically replace the proposed label.

To reproduce the shared reference, unpack `outputs/topic-analysis-reproduction.tar.gz`
and run these commands **inside its `topic-analysis-reproduction/` directory**:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r configs/topic-analysis-requirements.lock
.venv/bin/python scripts/reproduce_topic_analyses.py --check
```

The lock pins the installed dependency closure; the verification environment uses
Python 3.12.7 on macOS Apple Silicon. Other operating systems are not certified by
this check and may need platform-specific wheels. No GPU, model download, API key,
PDF extraction or fresh clustering is required. Dependency installation needs
network access; the analysis itself runs offline. The script executes each
notebook in a separate fresh Python process, captures its displayed outputs, and
raises if clustering, embedding inference, extraction or socket connections are
attempted. No Jupyter server is necessary. For an interactive kernel, install it
with `.venv/bin/python -m ipykernel install --user --name batill-topics` and select
that kernel before running a notebook from top to bottom.

`--check` verifies the reference evidence before and after execution: deterministic
CSV, JSON and review-card files must match their recorded SHA-256 hashes exactly,
including memberships, terms, excerpts, names and diagnostics. Execution timestamps,
absolute paths in run manifests and notebook display output are excluded from byte
comparison. Run manifests still record current code/input/package hashes and verify
the generated artifact checksums. A failure is reported rather than silently
replacing the reference. Run `--analysis citations` (or `pdfs`/`abstracts`) to execute
one notebook. Only after deliberately reviewing a changed analysis should you use
`--record-reference` to replace the reference for all three.

To rebuild the handoff archive after a reviewed update:

```bash
.venv/bin/python scripts/reproduce_topic_analyses.py --check
.venv/bin/python scripts/build_topic_reproduction_bundle.py
```

The archive includes code, notebooks, explicit naming inputs, the dependency lock,
prepared source text, saved embeddings, graph files, exact partition snapshots,
historical citation provenance and reference evidence. `BUNDLE_MANIFEST.json`
lists the included file checksums. The original source notebook corresponding to
the PDF embedding snapshot is preserved beside that snapshot as
`source_notebook.ipynb`; the current source notebook additionally exports citations.
The output/input directories are normally ignored by Git, so **sending notebook
code alone is insufficient**. The archive is intended for the supervisor handoff
and contains retained source text and abstracts; it is not a public-code release.
Reproducing the original extraction, embedding inference or clustering is a
separate workflow and is not a prerequisite for reproducing these interpretations.
