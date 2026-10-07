"""Generate the abstract scope workbook (generation clears notebook outputs)."""
from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


def main():
    cells = []

    def md(text):
        cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))

    def code(text):
        cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip()))

    md('''
    # Clean the abstract corpus for private-equity research

    Run all cells with **Python (batill)**. This is the second cleaning stage,
    after the basic text cleaning in `ngramm emb.ipynb`. It creates a smaller,
    auditable corpus and an **Abstract Cleaning.xlsx** review workbook.

    The strict default excludes venture-capital papers, **including mixed PE/VC
    papers and incidental VC mentions**, chemical-industry material and identified
    trade news. A PE/buyout term in the title or at least two occurrences in the
    abstract provides positive relevance evidence. Startup finance, private
    placements, nonresearch document types and weaker PE matches are held for
    review. Healthcare, real estate and other sector studies remain eligible when
    they examine PE ownership or buyouts.

    These transparent lexical rules are a screening aid, **not a manual assessment
    of every paper**. They can exclude relevant work and retain false positives.
    Inspect the audit, especially the review queue and a sample of retained papers.
    **Only `keep` enters the analysis; `review` is withheld.**

    Text, metadata, EIDs and vectors are preserved exactly for retained records.
    No embeddings are recomputed. Old sources and results remain reproducible.
    The final cell activates a shared input configuration for both abstract
    clustering and topic notebooks. Changing scope requires rerunning clustering.
    ''')
    md('''
    ## 1. Load the original corpus and editable screening rules

    Always start from the full original cleaned corpus, so rerunning never
    progressively deletes records. The rules live in
    `configs/abstract_scope_rules.json`; individual reviewed decisions live in
    `configs/abstract_scope_overrides.csv`. No internet connection is needed.
    ''')
    code('''
    from pathlib import Path
    import json
    import sys
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display

    ROOT = next((p for p in (Path.cwd(), *Path.cwd().parents)
                 if (p / 'src/batill').is_dir()), None)
    if ROOT is None:
        raise FileNotFoundError('Open this notebook from the batill project folder.')
    if str(ROOT / 'src') not in sys.path:
        sys.path.insert(0, str(ROOT / 'src'))
    from batill.abstract_scope import (
        ORIGINAL_SOURCE, ORIGINAL_RUN, OVERRIDE_COLUMNS, screen_abstracts,
        build_scope_bundle, activate_scope, resolve_abstract_inputs,
    )
    from batill.abstract_clustering import load_abstract_embeddings
    from batill.storage import file_hash

    SOURCE = ROOT / ORIGINAL_SOURCE
    PARENT_RUN = ROOT / ORIGINAL_RUN
    RULES_FILE = ROOT / 'configs/abstract_scope_rules.json'
    OVERRIDES_FILE = ROOT / 'configs/abstract_scope_overrides.csv'
    ACTIVATE_FOR_ANALYSIS = False  # Original corpus restored; opt in explicitly to activate filtering.
    rules = json.loads(RULES_FILE.read_text())
    # Optional experiment: edit a rule here before screening, e.g.
    # rules['exclude_any_vc_mention'] = False  # Mixed papers then go to REVIEW.
    overrides = pd.read_csv(OVERRIDES_FILE, dtype=str, keep_default_na=False)
    source = pd.read_csv(SOURCE, dtype=str, keep_default_na=False)
    source_hash_before = file_hash(SOURCE)
    print(f'Original clean corpus: {len(source):,} papers; manual decisions: {len(overrides)}')
    display(pd.Series(rules, name='Value').to_frame())
    ''')
    md('''
    ## 2. Screen and inspect the decisions

    Precedence: chemistry → known trade news → VC → startup finance → private
    placement/crowdfunding ambiguity → missing PE evidence → document type →
    weak PE focus → keep. Manual decisions take precedence and retain their
    original automatic decision and matched terms in the audit.
    ''')
    code('''
    audit = screen_abstracts(source, rules, overrides)
    counts = audit.groupby(['decision', 'reason']).size().rename('papers').reset_index()
    display(counts)
    ax = audit.decision.value_counts().reindex(['keep', 'exclude', 'review'], fill_value=0).plot.barh(
        color=['#276C5B', '#B34B44', '#C39135'], figsize=(8, 3), rot=0)
    ax.set(xlabel='Papers', ylabel='', title='Private-equity scope decisions')
    plt.tight_layout()
    plt.show()
    assert len(audit) == len(source)
    assert audit.paper_id.tolist() == source.paper_id.tolist()
    assert set(audit.decision) <= {'keep', 'exclude', 'review'}
    assert audit.decision.value_counts().sum() == len(source)
    print('Review records are excluded from the analysis until explicitly accepted.')
    ''')
    md('''
    ## 3. Review evidence and record corrections

    Change `DECISION_TO_INSPECT` to `keep`, `exclude` or `review`; use `SEARCH`
    to search titles, abstracts and journals. Set `PAPER_ID` to read one complete
    abstract. The export includes every record, not just the displayed preview.

    To correct a decision, copy the record's `paper_id` and `record_sha256` into
    `configs/abstract_scope_overrides.csv`, set `decision` to `keep`, `exclude`
    or `review`, and give a reason. Keep only rows you have actually decided on.
    Then rerun this notebook. Unknown IDs, duplicate decisions, blank reasons and
    changed source records are rejected. The bundle's `override_template.csv`
    supplies IDs/hashes but its blank decisions must be filled or removed first.

    Excel is a review export: editing its cells does **not** change the dataset.
    ''')
    code('''
    DECISION_TO_INSPECT = 'review'
    SEARCH = ''
    PAPER_ID = ''
    selected = audit.loc[audit.decision.eq(DECISION_TO_INSPECT)]
    if SEARCH:
        mask = selected[['title', 'abstract', 'journal']].apply(
            lambda col: col.str.contains(SEARCH, case=False, regex=False)).any(axis=1)
        selected = selected.loc[mask]
    columns = ['paper_id', 'title', 'journal', 'decision', 'reason',
               'private_equity_matches', 'venture_capital_matches', 'record_sha256']
    with pd.option_context('display.max_colwidth', 110):
        display(selected[columns].head(30))
    print(f'{len(selected):,} matching records. Full tables are exported below.')
    if PAPER_ID:
        record = audit.set_index('paper_id').loc[PAPER_ID]
        print(record['title'], '\\n\\n', record['abstract'])
        display(record[OVERRIDE_COLUMNS[1:]].to_frame('Override evidence'))
    kept_preview = audit.loc[audit.decision.eq('keep')]
    print('Reproducible sample of retained titles:')
    display(kept_preview.sample(min(15, len(kept_preview)), random_state=42)[
        ['paper_id', 'title', 'journal', 'reason']])
    ''')
    md('''
    ## 4. Export the filtered corpus, audit, Excel workbook and aligned embeddings

    Outputs use a content-derived folder under `data/abstracts/private_equity/`.
    Changing rules, inputs or decisions creates a new folder; an identical run
    validates and reuses its existing bundle. Every CSV field is preserved.
    Embedding rows are selected by EID in the original matrix order, with a map
    back to their parent rows. New source hashes, row numbers, token audits,
    corpus counts and vector dimensions are validated by the clustering loader.
    ''')
    code('''
    BUNDLE, manifest, saved_audit = build_scope_bundle(
        ROOT, SOURCE, PARENT_RUN, rules, overrides)
    filtered_source = BUNDLE / 'abstracts_clean.csv'
    filtered_run = BUNDLE / 'embedding_run'
    papers, vectors, provenance = load_abstract_embeddings(filtered_run, filtered_source)
    original_papers, original_vectors, _ = load_abstract_embeddings(PARENT_RUN, SOURCE)
    positions = original_papers.reset_index().set_index('paper_id').loc[papers.paper_id, 'index'].to_numpy()
    assert np.array_equal(vectors, original_vectors[positions])
    expected = source.set_index('paper_id').loc[papers.paper_id]
    assert papers[source.columns].astype(str).set_index('paper_id').equals(expected)
    assert set(papers.paper_id) == set(saved_audit.loc[saved_audit.decision.eq('keep'), 'paper_id'])
    assert set(papers.paper_id).isdisjoint(saved_audit.loc[saved_audit.decision.ne('keep'), 'paper_id'])
    assert file_hash(SOURCE) == source_hash_before
    print(json.dumps({'bundle': str(BUNDLE), 'counts': manifest['counts'],
                      'embedding_shape': list(vectors.shape),
                      'embeddings_recomputed': False}, indent=2))
    print('Excel review workbook:', BUNDLE / 'Abstract Cleaning.xlsx')
    print('Review queue:', BUNDLE / 'review.csv')
    ''')
    md('''
    ## 5. Activate the cleaned sample for the analysis notebooks

    `ACTIVATE_FOR_ANALYSIS = True` updates `configs/abstract_analysis_input.json`
    and copies the Excel review workbook to the project root. The original clean
    corpus and its embedding run are unchanged. To switch to a different scope,
    change the rules/overrides and rerun this workbook.

    Run in this order with fresh kernels:

    1. **Abstract Cleaning.ipynb** — this workbook.
    2. **Clustering Abstracts.ipynb** — rerun all cells on the new input. Previous
       cluster memberships, projections and diagnostics cannot be reused.
    3. **Topic_analysis_Abstracts_v2.ipynb** — select a partition from the new
       catalog. The legacy **Topic analysis Abstracts.ipynb** also reads the new
       exported partitions. Old proposed names are rejected when memberships differ.

    Leiden resolution 0.45/0.50 need not produce the same number of clusters after
    filtering. Inspect the new diagnostics; no old count or topic label is imposed.
    PDF/citation analyses and historical age/thesis reports remain separate saved
    analyses. To update age/thesis reports, explicitly select newly exported topic
    evidence in those workflows; rerunning cleaning alone does not revise reports.
    ''')
    code('''
    if ACTIVATE_FOR_ANALYSIS:
        active = activate_scope(ROOT, BUNDLE)
        active_run, active_source = resolve_abstract_inputs(ROOT)
        assert active_run == filtered_run and active_source == filtered_source
        print(f'Activated {active["corpus_size"]:,} abstracts for clustering and topic analysis.')
        print('Review workbook:', ROOT / 'Abstract Cleaning.xlsx')
        print('Next: run Clustering Abstracts.ipynb from top to bottom in a fresh kernel.')
    else:
        print('Candidate exported. Existing analysis input selection is unchanged.')
    ''')
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python (batill)', 'language': 'python', 'name': 'batill'},
        'language_info': {'name': 'python', 'version': '3.12.7'},
    })
    nbf.write(notebook, ROOT / 'Abstract Cleaning.ipynb')


if __name__ == '__main__':
    main()
