"""Create the transparent, step-by-step notebook for the large abstract corpus."""
from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


def main():
    cells = []

    def md(source):
        cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(source).strip()))

    def code(source):
        cells.append(nbf.v4.new_code_cell(textwrap.dedent(source).strip()))

    md('''
    # Abstract screening — an explicit decision trail

    Run top to bottom with **Python (batill)**. This notebook contains every rule
    and transformation for this screening stage, from the original **3,819-paper
    corpus** to paper-level decisions, a retained dataset, matching embedding rows
    and 20 examples for discussion. It applies the rules below in order.

    1. **Drop** a paper if neither title nor abstract contains the search term.
    2. **Safe by title** if the title contains the term. Otherwise, **retain for
       further evaluation** if only the abstract contains it.
    3. For **abstract-only papers**, require at least one allowed keyword in the
       journal title. All safe title matches remain retained.
    4. Drop remaining abstract-only papers with a detected list mention of PE.
    5. Drop remaining abstract-only papers whose PE mentions occur only in the
       last third of the abstract, measured by word position.

    “Safe” is the agreed working rule, not an independent assessment of relevance.
    Journal matches are a screening rule, not proof of substantive PE relevance.
    The editable keyword list and every journal decision are shown below.
    The 20 examples do not create any additional decisions.

    The original files remain unchanged. This notebook exports a separate
    screened dataset and embeddings; the existing clustering/topic notebooks
    remain on the original corpus while we evaluate the next step together.
    ''')
    md('''
    ## 0. Starting point and provenance

    The upstream text-preparation and embedding code is in **ngramm emb.ipynb**
    (basic cleaning also exists in `scripts/clean_abstracts.py`). It produced
    `data/abstracts/abstracts_clean.csv` and the original complete BGE-M3 run.
    Its saved cleaning summary and the input checksums below record that earlier
    stage. This notebook starts from those exact inputs. It does not rerun text
    preparation or embedding inference.
    ''')
    code('''
    from pathlib import Path
    from datetime import datetime, timezone
    import hashlib
    import json
    import re
    import sys
    import tempfile
    import unicodedata
    from html import escape

    import numpy as np
    import pandas as pd
    from IPython.display import display, HTML

    ROOT = next((p for p in (Path.cwd(), *Path.cwd().parents)
                 if (p / 'src/batill').is_dir()), None)
    if ROOT is None:
        raise FileNotFoundError('Open this notebook in the batill project.')
    if str(ROOT / 'src') not in sys.path:
        sys.path.insert(0, str(ROOT / 'src'))
    from batill.abstract_clustering import load_abstract_embeddings

    SOURCE = ROOT / 'data/abstracts/abstracts_clean.csv'
    PARENT_RUN = ROOT / 'outputs/abstract_embeddings/bge_m3/20260930T174845_508054Z_full_3c498a9d'
    OUTPUT_ROOT = ROOT / 'reports/abstract_screening'
    NOTEBOOK = ROOT / 'Abstract Screening.ipynb'

    def sha256(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    source = pd.read_csv(SOURCE, dtype=str, keep_default_na=False)
    parent_papers, parent_vectors, parent_provenance = load_abstract_embeddings(PARENT_RUN, SOURCE)
    input_hashes = {str(SOURCE.relative_to(ROOT)): sha256(SOURCE)}
    input_hashes.update({str(p.relative_to(ROOT)): sha256(p) for p in PARENT_RUN.iterdir() if p.is_file()})
    cleaning_summary_file = ROOT / 'data/abstracts/cleaning_summary.json'
    if cleaning_summary_file.exists():
        upstream_cleaning = json.loads(cleaning_summary_file.read_text())
        input_hashes[str(cleaning_summary_file.relative_to(ROOT))] = sha256(cleaning_summary_file)
        display(pd.DataFrame([{'stage': 'Raw Scopus records', 'papers': upstream_cleaning['source_rows']},
                              {'stage': 'After basic text cleaning', 'papers': len(source)}]))
    assert source.paper_id.is_unique and source.paper_id.str.strip().ne('').all()
    assert set(source.paper_id) == set(parent_papers.paper_id)
    print(f'Start: {len(source):,} papers, embedding matrix {parent_vectors.shape}.')
    print('Original source SHA-256:', sha256(SOURCE))
    ''')
    md('''
    ## 1. Define and count the search terms

    The default follows the full term **“private equity”**, case-insensitively,
    including hyphenated forms such as “private-equity”. Whole-word boundaries
    avoid substring matches. Unicode and hyphen normalization are used only for
    matching; saved titles, abstracts and embedding inputs are not rewritten.

    `INCLUDE_ABBREVIATION = False` means standalone “PE” alone is not sufficient.
    Set it to `True` to count the abbreviation too. Both alternatives are shown
    below so the effect is visible before filtering; the selected setting is
    recorded in every run. “PE” is a lexical match, not a contextual judgement.
    ''')
    code('''
    INCLUDE_ABBREVIATION = False
    SAMPLE_SIZE = 20
    SAMPLE_SEED = 42

    PHRASE = re.compile(r'\\bprivate[\\s-]+equity\\b', re.IGNORECASE)
    ABBREVIATION = re.compile(r'\\bPE\\b', re.IGNORECASE)
    HYPHENS = str.maketrans({char: '-' for char in '\\u2010\\u2011\\u2012\\u2013\\u2014\\u2212'})

    def matching_text(text):
        return unicodedata.normalize('NFKC', text).translate(HYPHENS).replace('\\u00ad', '')

    audit = source.copy()
    for field in ('title', 'abstract'):
        normalized = source[field].map(matching_text)
        audit[field + '_phrase'] = normalized.map(lambda text: bool(PHRASE.search(text)))
        audit[field + '_abbreviation'] = normalized.map(lambda text: bool(ABBREVIATION.search(text)))
        audit[field + '_phrase_count'] = normalized.map(lambda text: len(PHRASE.findall(text)))
        audit[field + '_abbreviation_count'] = normalized.map(lambda text: len(ABBREVIATION.findall(text)))
        audit[field + '_matched'] = (audit[field + '_phrase'] |
            (INCLUDE_ABBREVIATION & audit[field + '_abbreviation']))

    def presence_counts(title, abstract):
        return {'Neither': int((~title & ~abstract).sum()),
                'Abstract only': int((~title & abstract).sum()),
                'Title only': int((title & ~abstract).sum()),
                'Both': int((title & abstract).sum())}

    comparison = pd.DataFrame({
        'Full phrase only': presence_counts(audit.title_phrase, audit.abstract_phrase),
        'Full phrase or PE': presence_counts(audit.title_phrase | audit.title_abbreviation,
                                            audit.abstract_phrase | audit.abstract_abbreviation),
    })
    assert comparison.sum().eq(len(source)).all()
    display(comparison)
    print('SELECTED RULE:', 'private equity OR standalone PE' if INCLUDE_ABBREVIATION else 'private equity (full phrase only)')
    ''')
    md('''
    ## 2. Apply the two agreed rules and account for every paper

    Only the **neither** category is dropped. Every title match is safe under the
    current rule, including papers without a match in their abstract. Every
    abstract-only match is retained with the status `review_abstract_only`.
    The audit records the reason and the matching fields for every original ID.
    ''')
    code('''
    no_match = ~audit.title_matched & ~audit.abstract_matched
    title_match = audit.title_matched
    abstract_only = ~audit.title_matched & audit.abstract_matched
    assert (no_match.astype(int) + title_match.astype(int) + abstract_only.astype(int)).eq(1).all()

    audit['status'] = np.select([no_match, title_match, abstract_only],
        ['dropped_no_match', 'safe_title', 'review_abstract_only'], default='ERROR')
    audit['reason'] = np.select([no_match, title_match, abstract_only],
        ['No selected term in title or abstract',
         'Title contains the selected term: safe under the agreed rule',
         'Abstract contains the selected term but title does not: retained pending evaluation'], default='ERROR')
    audit['retained'] = ~no_match

    dropped = audit.loc[no_match].copy()
    safe = audit.loc[title_match].copy()
    needs_review = audit.loc[abstract_only].copy()
    retained = source.loc[~no_match].copy()  # Original columns only: same schema as the original.
    stage_counts = pd.DataFrame([
        {'stage': 'Starting corpus', 'papers': len(source)},
        {'stage': 'Dropped: neither title nor abstract', 'papers': len(dropped)},
        {'stage': 'Retained: safe title matches', 'papers': len(safe)},
        {'stage': 'Retained: abstract only, pending review', 'papers': len(needs_review)},
        {'stage': 'Total retained after step 1', 'papers': len(retained)},
    ])
    display(stage_counts)
    assert len(dropped) + len(safe) + len(needs_review) == len(source)
    assert set(retained.paper_id) == set(safe.paper_id) | set(needs_review.paper_id)
    assert set(dropped.paper_id).isdisjoint(retained.paper_id)
    assert audit.status.ne('ERROR').all()
    print('No abstract-only papers have been removed.')
    ''')
    md('''
    ## 3. Journal keyword screen — abstract-only papers

    Keep an abstract-only paper if its journal title contains **at least one**
    keyword below (OR, not AND). Title matches remain safe regardless of journal.
    Papers already dropped for lacking the phrase stay dropped.

    The stems `econom`, `financ`, `manag`, `market`, and `invest` match within
    words, so they include Economics/Macroeconomics, Financial/Finance,
    Management, Marketing, and Investment/Investing. `accounting` is a whole
    word; acquisition and merger accept singular/plural forms. Matching is
    case-insensitive. “Aquisition” is treated as the intended **acquisition**;
    the misspelling is also accepted if it occurs in the source journal title.
    The full phrase **private equity** (including hyphenated forms) is also
    allowed, as agreed after reviewing the initial list. Missing journal titles
    fail this screen for abstract-only papers.

    Edit `JOURNAL_KEYWORDS` to change the policy. Each run records the exact
    patterns, matched keywords, stage-one status, and final decision per paper.
    The tables show **all journals** without truncation so exclusions can be
    inspected. Source titles are preserved verbatim in the audit.
    ''')
    code('''
    JOURNAL_KEYWORDS = {
        'econom...': r'econom\\w*',
        'financ...': r'financ\\w*',
        'accounting': r'\\baccounting\\b',
        'manag...': r'manag\\w*',
        'market...': r'market\\w*',
        'acquisition': r'\\b(?:acquisitions?|aquisitions?)\\b',
        'merger': r'\\bmergers?\\b',
        'invest...': r'invest\\w*',
        'private equity': r'\\bprivate[\\s-]+equity\\b',
        'business': r'\\bbusiness\\b',
        'corporate': r'\\bcorporate\\b',
    }
    JOURNAL_SCOPE = 'abstract_only'  # Agreed scope: never override safe title matches.
    assert JOURNAL_SCOPE == 'abstract_only'
    journal_patterns = {label: re.compile(pattern, re.IGNORECASE)
                        for label, pattern in JOURNAL_KEYWORDS.items()}
    display(pd.DataFrame(JOURNAL_KEYWORDS.items(), columns=['Allowed keyword', 'Pattern']))

    # Preserve the complete first-stage decisions before applying this new step.
    stage_one_audit = audit.copy()
    stage_one_review = needs_review.copy()
    audit['stage_one_status'] = audit.status
    audit['stage_one_reason'] = audit.reason
    audit['stage_one_retained'] = audit.retained
    audit['journal_matched_keywords'] = source.journal.map(matching_text).map(
        lambda text: '; '.join(label for label, pattern in journal_patterns.items() if pattern.search(text)))
    audit['journal_keyword_match'] = audit.journal_matched_keywords.ne('')
    audit['journal_screen_applied'] = abstract_only
    journal_excluded = abstract_only & ~audit.journal_keyword_match
    audit['journal_decision'] = np.select(
        [no_match, title_match, abstract_only & audit.journal_keyword_match, journal_excluded],
        ['not_applied_previously_dropped', 'not_applied_safe_title', 'keep_keyword_match', 'drop_no_keyword'],
        default='ERROR')
    audit.loc[journal_excluded, 'status'] = 'dropped_journal'
    audit.loc[journal_excluded, 'reason'] = 'Abstract-only paper: no allowed keyword in journal title'
    audit.loc[abstract_only & audit.journal_keyword_match, 'reason'] = (
        'Abstract-only paper: allowed journal keyword; retained pending relevance evaluation')
    audit['retained'] = audit.stage_one_retained & ~journal_excluded
    journal_dropped = audit.loc[journal_excluded].copy()
    safe = audit.loc[title_match].copy()
    needs_review = audit.loc[abstract_only & audit.retained].copy()
    retained = source.loc[audit.retained].copy()
    stage_counts = pd.concat([stage_counts, pd.DataFrame([
        {'stage': 'Journal screen: abstract-only papers evaluated', 'papers': int(abstract_only.sum())},
        {'stage': 'Journal screen: additional papers dropped', 'papers': len(journal_dropped)},
        {'stage': 'After journal screen: safe title matches', 'papers': len(safe)},
        {'stage': 'After journal screen: abstract-only, pending review', 'papers': len(needs_review)},
        {'stage': 'Total retained after journal screen', 'papers': len(retained)},
    ])], ignore_index=True)
    journal_summary = (audit.loc[abstract_only]
        .groupby(['journal', 'journal_matched_keywords', 'journal_decision'], dropna=False)
        .size().reset_index(name='papers')
        .sort_values(['journal_decision', 'papers', 'journal'], ascending=[True, False, True]))
    journal_keyword_counts = pd.DataFrame([
        {'keyword': label, 'abstract_only_papers': int(source.loc[abstract_only, 'journal']
            .map(matching_text).map(lambda text: bool(pattern.search(text))).sum())}
        for label, pattern in journal_patterns.items()])

    assert audit.loc[title_match, 'retained'].all()
    assert not audit.loc[no_match, 'retained'].any()
    assert audit.loc[abstract_only, 'retained'].equals(audit.loc[abstract_only, 'journal_keyword_match'])
    assert len(dropped) + len(journal_dropped) + len(safe) + len(needs_review) == len(source)
    assert set(retained.paper_id) == set(safe.paper_id) | set(needs_review.paper_id)
    assert audit.journal_decision.ne('ERROR').all()
    display(stage_counts)
    print('Keyword counts overlap when a journal matches more than one keyword:')
    display(journal_keyword_counts)
    print('All journals among the abstract-only papers (full table):')
    display(HTML(journal_summary.to_html(index=False, escape=True)))
    print(f'{len(journal_dropped):,} additional exclusions; all {len(safe):,} title matches preserved.')
    ''')
    md('''
    ## 4. Detect list mentions and measure mention positions

    These are explicit text heuristics, not a semantic classifier. They apply
    **only to abstract-only papers that passed the journal screen**. Safe title
    matches remain safe. Each PE occurrence is saved with its context and evidence.

    **List rule:** flag a PE mention inside a sequence of at least three short
    noun-like items separated by commas, semicolons, `and`, or `or`. Two items
    suffice inside parentheses or with an explicit example/list cue (`including`,
    `such as`, `e.g.`, etc.). Ordinary two-subject PE/VC comparisons stay eligible.
    Parentheses alone, the abbreviation `(PE)`, and a comma after a sentence
    about PE are not enough. A transparent clause-word guard rejects obvious
    sentences as list items. Any detected list mention triggers exclusion, even
    if another PE mention is not a list. The evidence includes both counts.
    This pattern can miss unusual lists or misread short phrases; inspect
    `dropped_list_mentions.csv` and `abstract_mention_evidence.csv`.
    For precision, the PE item must be the phrase itself with optional short
    modifiers from `PE_LIST_ITEM` (e.g. listed PE, PE funds, PE investments).
    PE buried inside a longer clause or topic description is not auto-excluded.

    **Position rule:** count words in the complete supplied abstract after the
    same Unicode normalization used for phrase matching. A word may include an
    internal hyphen/apostrophe. A mention's position is the number of words
    before its start divided by the total word count. Drop a paper only when
    **all** PE mentions start at or after **2/3**. Earlier mentions prevent this
    position exclusion. Copyright/publisher text, if present, is counted too.
    ''')
    code(r'''
    # ABSTRACT_MENTION_RULES: inspectable lexical heuristics.
    from bisect import bisect_right

    ABSTRACT_RULE_SCOPE = 'abstract_only'
    LAST_THIRD_START = 2 / 3
    MAX_LIST_ITEM_WORDS = 10
    PE_LIST_ITEM = re.compile(
        r'^(?:(?:the|a|an|listed|unlisted|global|domestic|international|foreign|local|'
        r'green|secondary|traditional|corporate|institutional|other|direct|indirect)\s+){0,3}'
        r'private[\s-]+equity'
        r'(?:[\s-]+(?:funds?|firms?|investors?|investments?|investing|financing|funding|'
        r'capital|markets?|placements?|ownership|buyouts?|acquisitions?|companies|'
        r'groups?|players?|partners?|partnerships?|assets?|sectors?|returns?|strategies|'
        r'backed|owned|sponsored|managed|style)){0,4}$', re.IGNORECASE)
    WORD = re.compile(r"\b\w+(?:[-']\w+)*\b")
    LIST_SEPARATOR = re.compile(r',|;|\b(?:and|or)\b|&', re.IGNORECASE)
    LIST_CUE = re.compile(
        r'\b(?:including|include[sd]?|such as|for example|namely|consist(?:s|ing)? of)\b|\be\.g\.',
        re.IGNORECASE)
    # These indicate clauses or transitions rather than short enumerated terms.
    CLAUSE_WORDS = set(('is are was were be been being has have had do does did can could will would '
        'shall should may might must that which who whose when where whether if '
        'although because however therefore thus hence while whereas unlike '
        'examine examines examined study studies studied investigate investigates investigated '
        'find finds found show shows showed reveal reveals revealed suggest suggests suggested '
        'demonstrate demonstrates demonstrated we our this these they their it its to through than '
        'increases decreases reduces improves affects influences provides generates').split())

    def parenthesis_spans(text):
        stack, spans = [], []
        for i, char in enumerate(text):
            if char == '(':
                stack.append(i)
            elif char == ')' and stack:
                spans.append((stack.pop() + 1, i))
        return spans

    def noun_like_item(piece):
        simple = piece
        for _ in range(10):
            updated = re.sub(r'\([^()]*\)', ' ', simple)
            if updated == simple:
                break
            simple = updated
        cues = list(LIST_CUE.finditer(simple))
        if cues:
            simple = simple[cues[-1].end():]
        simple = re.sub(r'^\s*(?:and|or)\b', '', simple, flags=re.IGNORECASE).strip(' :–—-')
        # A terminal full stop belongs to the sentence, not the last list item.
        simple = simple.rstrip('.')
        words = WORD.findall(simple.lower())
        return (0 < len(words) <= MAX_LIST_ITEM_WORDS
                and not set(words).intersection(CLAUSE_WORDS)
                and not re.search(r'[.!?=:]', simple)), bool(cues), simple

    def list_in_window(text, start, end, mention, parenthetical=False):
        window = text[start:end]
        depth, depths = 0, []
        for char in window:
            depths.append(depth)
            depth += (char == '(') - (char == ')' and depth > 0)
        separators = [m for m in LIST_SEPARATOR.finditer(window) if depths[m.start()] == 0]
        # Treat the Oxford-comma combination ', and' as one separator.
        merged = []
        for separator in separators:
            a, b = separator.span()
            if merged and not window[merged[-1][1]:a].strip():
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        starts = [0] + [b for a, b in merged]
        ends = [a for a, b in merged] + [len(window)]
        chunks = list(zip(starts, ends))
        index = next((i for i, (a, b) in enumerate(chunks)
                      if a <= mention.start() - start and mention.end() - start <= b), None)
        if index is None:
            return None
        valid = [noun_like_item(window[a:b]) for a, b in chunks]
        if not valid[index][0] or not PE_LIST_ITEM.fullmatch(valid[index][2]):
            return None
        left = right = index
        while left > 0 and valid[left - 1][0] and not valid[left][1]:
            left -= 1
        while right + 1 < len(chunks) and valid[right + 1][0] and not valid[right + 1][1]:
            right += 1
        size = right - left + 1
        explicit_cue = valid[left][1]
        if size < 3 and not (size >= 2 and (parenthetical or explicit_cue)):
            return None
        return {'kind': 'parenthetical_list' if parenthetical else
                        ('explicit_example_list' if explicit_cue else 'enumeration'),
                'items': size,
                'evidence_start': start + chunks[left][0],
                'evidence_end': start + chunks[right][1],
                'evidence': window[chunks[left][0]:chunks[right][1]].strip()}

    def abstract_mention_features(raw_text):
        text = matching_text(raw_text)
        words = list(WORD.finditer(text))
        word_starts = [word.start() for word in words]
        mentions = list(PHRASE.finditer(text))
        parentheses = parenthesis_spans(text)
        boundaries = [0] + [m.end() for m in re.finditer(r'(?<=[.!?])\s+(?=[A-Z])|\n+', text)] + [len(text)]
        details = []
        for number, mention in enumerate(mentions, 1):
            windows = sorted([(a, b, True) for a, b in parentheses
                              if a <= mention.start() and mention.end() <= b], key=lambda w: w[1] - w[0])
            sentence_index = max(0, bisect_right(boundaries, mention.start()) - 1)
            windows.append((boundaries[sentence_index], boundaries[sentence_index + 1], False))
            evidence = next((found for a, b, paren in windows
                             if (found := list_in_window(text, a, b, mention, paren))), None)
            word_index = max(0, bisect_right(word_starts, mention.start()) - 1)
            details.append({'mention_number': number, 'start_char': mention.start(),
                'end_char': mention.end(), 'word_index_zero_based': word_index,
                'word_fraction': word_index / len(words),
                'is_list_mention': evidence is not None,
                'list_kind': evidence['kind'] if evidence else '',
                'list_items': evidence['items'] if evidence else 0,
                'list_evidence': evidence['evidence'] if evidence else '',
                'evidence_start_char': evidence['evidence_start'] if evidence else None,
                'evidence_end_char': evidence['evidence_end'] if evidence else None,
                'context': text[max(0, mention.start() - 120):mention.end() + 120]})
        listed = sum(d['is_list_mention'] for d in details)
        return {'abstract_screen_word_count': len(words),
                'pe_mention_count': len(details), 'pe_list_mention_count': listed,
                'pe_nonlist_mention_count': len(details) - listed,
                'pe_any_list_mention': bool(listed),
                'pe_first_word_fraction': details[0]['word_fraction'] if details else None,
                'pe_only_last_third': bool(details) and all(d['word_fraction'] >= LAST_THIRD_START for d in details),
                'details': details}
    ''')
    md('''
    ## 5. Apply the new rules in sequence and account for overlaps

    List exclusions are applied first, then position exclusions. The overlap
    table shows papers meeting both criteria before either is applied, so no
    paper is double-counted. Earlier journal and phrase decisions are preserved.
    Each exclusion table contains the complete title, abstract, and reasons;
    the mention evidence table gives matched context and normalized-text offsets.
    ''')
    code('''
    assert ABSTRACT_RULE_SCOPE == 'abstract_only'
    journal_stage_audit = audit.copy()
    audit['journal_stage_retained'] = audit.retained
    features = [abstract_mention_features(text) for text in source.abstract]
    mention_rows = [dict(paper_id=paper_id, **detail)
                    for paper_id, feature in zip(source.paper_id, features)
                    for detail in feature['details']]
    mention_evidence = pd.DataFrame(mention_rows)
    for name in features[0]:
        if name != 'details':
            audit[name] = [feature[name] for feature in features]
    assert audit.pe_mention_count.equals(audit.abstract_phrase_count)
    eligible = abstract_only & audit.journal_stage_retained
    audit['abstract_rules_eligible'] = eligible
    rule_overlap = (audit.loc[eligible].groupby(['pe_any_list_mention', 'pe_only_last_third'])
                    .size().reset_index(name='papers'))
    list_excluded = eligible & audit.pe_any_list_mention
    audit.loc[list_excluded, 'status'] = 'dropped_list_mention'
    audit.loc[list_excluded, 'reason'] = 'Abstract-only paper: PE occurs in a detected enumeration'
    audit.loc[list_excluded, 'retained'] = False
    audit['list_rule_applied'] = eligible
    audit['list_rule_dropped'] = list_excluded
    after_list_count = int(audit.retained.sum())
    position_eligible = abstract_only & audit.retained
    position_excluded = position_eligible & audit.pe_only_last_third
    audit.loc[position_excluded, 'status'] = 'dropped_last_third'
    audit.loc[position_excluded, 'reason'] = 'Abstract-only paper: all PE mentions start in the last third by word count'
    audit.loc[position_excluded, 'retained'] = False
    audit['position_rule_applied'] = position_eligible
    audit['position_rule_dropped'] = position_excluded
    audit.loc[abstract_only & audit.retained, 'reason'] = 'Passed journal, list-mention and last-third rules; retained pending review'
    list_dropped = audit.loc[list_excluded].copy()
    position_dropped = audit.loc[position_excluded].copy()
    safe = audit.loc[title_match].copy()
    needs_review = audit.loc[abstract_only & audit.retained].copy()
    retained = source.loc[audit.retained].copy()
    stage_counts = pd.concat([stage_counts, pd.DataFrame([
        {'stage': 'List screen: abstract-only papers evaluated', 'papers': int(eligible.sum())},
        {'stage': 'List screen: additional papers dropped', 'papers': len(list_dropped)},
        {'stage': 'Total retained after list screen', 'papers': after_list_count},
        {'stage': 'Position screen: abstract-only papers evaluated', 'papers': int(position_eligible.sum())},
        {'stage': 'Position screen: additional papers dropped', 'papers': len(position_dropped)},
        {'stage': 'Final: safe title matches', 'papers': len(safe)},
        {'stage': 'Final: abstract-only, pending review', 'papers': len(needs_review)},
        {'stage': 'Total retained after all filters', 'papers': len(retained)},
    ])], ignore_index=True)
    assert audit.loc[title_match, 'retained'].all()
    assert not audit.loc[~audit.journal_stage_retained, 'retained'].any()
    assert not (needs_review.pe_any_list_mention | needs_review.pe_only_last_third).any()
    assert len(dropped) + len(journal_dropped) + len(list_dropped) + len(position_dropped) + len(retained) == len(source)
    assert set(retained.paper_id) == set(safe.paper_id) | set(needs_review.paper_id)
    display(stage_counts)
    print('Overlap before applying the two new rules (abstract-only, journal passed):')
    display(rule_overlap)
    print('Illustrative exclusion evidence (complete exclusion tables are exported):')
    display(HTML(mention_evidence.loc[mention_evidence.paper_id.isin(list_dropped.paper_id)
        & mention_evidence.is_list_mention, ['paper_id', 'list_kind', 'list_evidence']]
        .head(10).to_html(index=False, escape=True)))
    display(position_dropped[['paper_id', 'title', 'pe_mention_count', 'pe_first_word_fraction']].head(10))
    ''')
    md('''
    ## 6. Prepare 20 examples from the end of the funnel

    A reproducible random sample (`SAMPLE_SEED = 42`) samples the abstract-only papers remaining
    after every filter without selecting examples to support a particular exclusion rule.
    Every example displays its **complete title and abstract**, year, journal,
    document type and ID. The phrase is highlighted for inspection. Abbreviation
    matches are highlighted only if that option is enabled.

    While reading, consider: What is the actual research question? Is PE part of
    the sample, results or argument, or only background? Does the paper study
    both PE and VC? Do source type or other features recur? These are discussion
    prompts, **not additional filters**. Record any notes in the exported
    `review_notes.csv`; this stage does not apply those notes as decisions.
    ''')
    code('''
    examples = needs_review.sample(n=min(SAMPLE_SIZE, len(needs_review)), random_state=SAMPLE_SEED).copy()
    examples.insert(0, 'example_number', range(1, len(examples) + 1))
    assert len(examples) == min(SAMPLE_SIZE, len(needs_review))
    assert examples.paper_id.is_unique
    assert examples.retained.all()
    assert set(examples.paper_id).issubset(retained.paper_id)


    # Match on raw text for display, preserving the exact source string and escaping HTML.
    highlight_pattern = r'\\bprivate[\\s\\-\\u2010-\\u2014\\u2212]+equity\\b'
    if INCLUDE_ABBREVIATION:
        highlight_pattern += r'|\\bPE\\b'
    HIGHLIGHT = re.compile(highlight_pattern, re.IGNORECASE)

    def highlighted(text):
        parts, end = [], 0
        for match in HIGHLIGHT.finditer(text):
            parts.extend([escape(text[end:match.start()]), '<mark>', escape(match.group()), '</mark>'])
            end = match.end()
        parts.append(escape(text[end:]))
        return ''.join(parts)

    cards = []
    for row in examples.itertuples(index=False):
        cards.append(
            f'<article class="paper"><h3>{row.example_number}. {highlighted(row.title)}</h3>'
            f'<p class="meta"><b>ID:</b> {escape(row.paper_id)} | <b>Year:</b> {escape(row.year)}<br>'
            f'<b>Journal:</b> {escape(row.journal)} | <b>Type:</b> {escape(row.document_type)}<br>'
            f'<b>DOI:</b> {escape(row.doi) or "Not supplied"}</p>'
            f'<p><b>Full abstract</b></p><div class="abstract">{highlighted(row.abstract)}</div>'
            '<p class="state">Retained after all current filters — abstract-only match; further relevance review pending.</p></article>')
    examples_html = ('<!doctype html><html><head><meta charset="utf-8"><title>20 abstract-only examples</title>'
        '<style>.screening{font-family:system-ui,sans-serif;max-width:1050px;margin:auto;line-height:1.55}'
        '.screening .paper{padding:18px 22px;margin:18px 0;border:1px solid #bbb;border-radius:6px}'
        '.screening .meta,.screening .state{font-size:0.9em}.screening .abstract{white-space:pre-wrap}'
        '.screening mark{background:#ffe58a;color:#151515;padding:0 2px}</style></head><body>'
        '<main class="screening"><h2>After all filters: 20 retained abstract-only papers for discussion</h2>'
        + ''.join(cards) + '</main></body></html>')
    print(f'Prepared {len(examples)} examples from {len(needs_review):,} remaining abstract-only papers; displayed at the end.')
    ''')
    md('''
    ## 7. Keep the existing embeddings aligned with the retained papers

    Select rows by stable Scopus ID after validating the full original run.
    Safe-title papers and abstract-only papers passing all current rules are
    included. The row map shows
    precisely which original matrix row becomes each new row. No text is edited
    and no embedding is recomputed. Clustering and topic analysis will be a later
    step after we decide how to evaluate the abstract-only group.
    ''')
    code('''
    retained_ids = set(retained.paper_id)
    parent_rows = np.flatnonzero(parent_papers.paper_id.isin(retained_ids).to_numpy())
    selected = pd.read_csv(PARENT_RUN / 'selected_papers.csv', dtype=str, keep_default_na=False).iloc[parent_rows].copy()
    selected['embedding_row'] = np.arange(len(selected))
    retained_vectors = parent_vectors[parent_rows]
    row_map = pd.DataFrame({'paper_id': selected.paper_id.to_numpy(),
                            'parent_embedding_row': parent_rows,
                            'embedding_row': np.arange(len(selected))})
    assert set(selected.paper_id) == retained_ids
    assert selected[source.columns].set_index('paper_id').equals(
        source.set_index('paper_id').loc[selected.paper_id])
    assert np.array_equal(retained_vectors, parent_vectors[row_map.parent_embedding_row.to_numpy()])
    print(f'Retained matrix: {retained_vectors.shape}; includes {len(needs_review):,} abstract-only papers passing all filters.')
    display(row_map.head())
    ''')
    md('''
    ## 8. Save the complete decision trail and verify the exported run

    Each distinct combination of inputs, rules and notebook code gets its own
    folder under `reports/abstract_screening/`. It contains the decision audit,
    exact dropped/safe/review lists, retained source CSV, aligned embeddings,
    row map, 20 examples as CSV and HTML, a notebook source snapshot, upstream
    provenance and a manifest with counts and hashes. Identical reruns validate
    and reuse the existing snapshot. Changes create a new folder.

    `review_notes.csv` is kept separately from the generated snapshot and is never
    overwritten. Notes do not affect retention in this notebook. Existing analysis
    input configurations are not changed by this stage.
    ''')
    code('''
    notebook_source = json.loads(NOTEBOOK.read_text())
    for cell in notebook_source['cells']:
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
    code_sources = [''.join(c['source']) for c in notebook_source['cells'] if c['cell_type'] == 'code']
    rules = {'include_abbreviation': INCLUDE_ABBREVIATION, 'phrase_pattern': PHRASE.pattern,
             'abbreviation_pattern': ABBREVIATION.pattern, 'case_sensitive': False,
             'safe_rule': 'title match', 'drop_rule': 'no match in either title or abstract',
             'abstract_only_policy': 'journal keyword, then exclude lists, then exclude last-third-only mentions',
             'journal_scope': JOURNAL_SCOPE, 'journal_keywords': JOURNAL_KEYWORDS,
             'journal_missing_policy': 'exclude abstract-only papers',
             'abstract_rule_scope': ABSTRACT_RULE_SCOPE, 'last_third_start': LAST_THIRD_START,
             'list_policy': 'any qualifying list mention; 3 items or 2 with parentheses/explicit cue',
             'list_max_item_words': MAX_LIST_ITEM_WORDS,
             'pe_list_item_pattern': PE_LIST_ITEM.pattern,
             'list_clause_words': sorted(CLAUSE_WORDS), 'list_cue_pattern': LIST_CUE.pattern,
             'list_separator_pattern': LIST_SEPARATOR.pattern, 'word_pattern': WORD.pattern,
             'sample_pool': 'final retained abstract-only papers',
             'sample_size': SAMPLE_SIZE, 'sample_seed': SAMPLE_SEED}
    identity = {'input_hashes': input_hashes, 'rules': rules,
                'notebook_code_sha256': hashlib.sha256(json.dumps(code_sources).encode()).hexdigest()}
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    RUN_DIR = OUTPUT_ROOT / run_id
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if not RUN_DIR.exists():
        with tempfile.TemporaryDirectory(prefix='.screening-', dir=OUTPUT_ROOT) as temporary:
            stage = Path(temporary) / 'bundle'
            embed_dir = stage / 'embedding_run'
            embed_dir.mkdir(parents=True)
            tables = {'paper_decisions.csv': audit, 'dropped_no_match.csv': dropped,
                      'safe_title.csv': safe, 'abstract_only_review.csv': needs_review,
                      'abstracts_clean.csv': retained, 'examples_20.csv': examples,
                      'embedding_row_map.csv': row_map, 'stage_counts.csv': stage_counts,
                      'stage_one_decisions.csv': stage_one_audit,
                      'stage_one_abstract_only_review.csv': stage_one_review,
                      'dropped_journal.csv': journal_dropped,
                      'journal_summary.csv': journal_summary,
                      'journal_keyword_counts.csv': journal_keyword_counts,
                      'journal_stage_decisions.csv': journal_stage_audit,
                      'dropped_list_mentions.csv': list_dropped,
                      'dropped_last_third.csv': position_dropped,
                      'abstract_rule_overlap.csv': rule_overlap,
                      'abstract_mention_evidence.csv': mention_evidence}
            for name, table in tables.items():
                table.to_csv(stage / name, index=False)
            (stage / 'examples_20.html').write_text(examples_html, encoding='utf-8')
            (stage / 'notebook_source.ipynb').write_text(json.dumps(notebook_source, ensure_ascii=False, indent=1) + '\\n')
            if cleaning_summary_file.exists():
                (stage / 'upstream_cleaning_summary.json').write_bytes(cleaning_summary_file.read_bytes())
            selected.to_csv(embed_dir / 'selected_papers.csv', index=False)
            selected[['embedding_row', 'paper_id']].to_csv(embed_dir / 'paper_ids.csv', index=False)
            token_audit = pd.read_csv(PARENT_RUN / 'token_lengths.csv', dtype=str, keep_default_na=False)
            token_audit.set_index('paper_id', drop=False).loc[selected.paper_id].to_csv(embed_dir / 'token_lengths.csv', index=False)
            np.save(embed_dir / 'embeddings.npy', retained_vectors)
            model_fields = ('model', 'model_revision', 'input_column', 'input_format', 'pooling',
                            'normalization', 'dtype', 'max_tokens', 'truncation', 'chunking', 'packages')
            summary = {k: parent_provenance[k] for k in model_fields if k in parent_provenance}
            summary.update(status='complete', run_mode='full',
                source=str(RUN_DIR / 'abstracts_clean.csv'), source_sha256=sha256(stage / 'abstracts_clean.csv'),
                corpus_size=len(retained), selected_size=len(retained), shape=list(retained_vectors.shape),
                selected_over_limit=0, corpus_over_limit=0,
                maximum_sample_tokens=int(selected.token_count.astype(int).max()),
                maximum_corpus_tokens=int(selected.token_count.astype(int).max()),
                derivation='exact original embedding rows selected by paper_id; no inference',
                screening_stage='term_presence_then_title_safe_then_journal_then_lists_then_position',
                parent_run=str(PARENT_RUN), parent_input_hashes=parent_provenance['input_hashes'])
            (embed_dir / 'run_summary.json').write_text(json.dumps(summary, indent=2) + '\\n')
            manifest = {'run_id': run_id, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'identity': identity, 'counts': audit.status.value_counts().to_dict(),
                'source_papers': len(source), 'retained_papers': len(retained),
                'example_ids': examples.paper_id.tolist(), 'embeddings_recomputed': False,
                'clustering_rerun': False, 'analysis_inputs_changed': False,
                'artifacts_sha256': {str(p.relative_to(stage)): sha256(p) for p in stage.rglob('*') if p.is_file()}}
            (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\\n')
            # Check the derivative before moving the finished bundle into its final location.
            load_abstract_embeddings(embed_dir, stage / 'abstracts_clean.csv')
            stage.rename(RUN_DIR)

    manifest = json.loads((RUN_DIR / 'manifest.json').read_text())
    assert manifest['identity'] == identity
    for name, digest in manifest['artifacts_sha256'].items():
        assert sha256(RUN_DIR / name) == digest, f'Existing artifact changed: {name}'
    checked_papers, checked_vectors, _ = load_abstract_embeddings(RUN_DIR / 'embedding_run', RUN_DIR / 'abstracts_clean.csv')
    assert checked_papers.paper_id.tolist() == selected.paper_id.tolist()
    assert np.array_equal(checked_vectors, retained_vectors)
    for relative, digest in input_hashes.items():
        assert sha256(ROOT / relative) == digest, f'Original input changed: {relative}'

    notes_file = RUN_DIR / 'review_notes.csv'
    if not notes_file.exists():
        notes = needs_review[['paper_id', 'title']].copy()
        notes['review_decision'] = ''
        notes['review_reason'] = ''
        notes.to_csv(notes_file, index=False)
    print('Saved and verified:', RUN_DIR)
    print(f'{len(dropped):,} dropped by phrase; {len(journal_dropped):,} dropped by journal; '
          f'{len(list_dropped):,} dropped by list; {len(position_dropped):,} dropped by position; '
          f'{len(safe):,} safe by title; {len(needs_review):,} retained for review.')
    print('Notes file (not applied as decisions):', notes_file)
    print('Original inputs unchanged; all retained vectors exactly match their original rows.')
    display(HTML(f'<p><b>Full examples:</b> <code>{escape(str(RUN_DIR / "examples_20.html"))}</code></p>'))
    ''')
    md('''
    ## Next checkpoint

    Review the exclusion evidence and the updated 20-paper sample below.
    Passing these screens does not establish substantive relevance. Any
    agreed next rule should be added as a new, separately counted step,
    preserving all earlier audits. Insert future filters before the sample is
    prepared so it always reflects the latest retained review pool.

    After the review rules are settled, use the final retained CSV and its matching
    embedding run for new clustering and topic analysis. Do not reuse memberships
    from a differently sized corpus.
    ''')
    md('''
    ## 9. Inspect 20 papers remaining after the latest filter

    These complete abstracts are sampled only from the final retained abstract-only
    group. Safe title matches are outside this review pool. Rerunning after a new
    filter automatically updates the sample. Every example passed all current
    filters; use these to decide the next step in the funnel.
    ''')
    code('''
    assert examples.retained.all()
    assert set(examples.paper_id).issubset(needs_review.paper_id)
    display(HTML(examples_html))
    ''')
    md('''
    ## 10. Experiment: protected title matches that would fail other rules

    **Inspection only — no exclusions are applied.** This cell reads the approved
    screening audit pinned by `configs/abstract_final_input.json`, so the diagnostic
    refers to the same papers used by the Final notebooks, even if earlier screening
    cells have since been edited. It can also be run on its own.

    All papers with “private equity” in their title remain protected. The table
    below shows every protected paper that would fail the journal or list-mention
    rule, plus the last-third-only rule as a separate diagnostic. Counts overlap:
    a paper can trigger several rules. The list rule means **any** detected list
    mention, even when PE is also discussed elsewhere. Missing abstract mentions
    do not count as last-third-only mentions.

    Open a row's **Abstract and evidence** panel to inspect the complete abstract,
    detected list passages, and the first PE mention's word position. No source
    CSVs, retention decisions, embeddings, or analysis inputs are changed.
    ''')
    code(r'''
    # PROTECTED_TITLE_DIAGNOSTIC: read-only inspection of the approved funnel.
    from pathlib import Path
    from html import escape
    import json
    import re
    import sys
    import pandas as pd
    from IPython.display import display, HTML

    diagnostic_root = next(p for p in (Path.cwd(), *Path.cwd().parents)
                           if (p / 'src/batill').is_dir())
    if str(diagnostic_root / 'src') not in sys.path:
        sys.path.insert(0, str(diagnostic_root / 'src'))
    from batill.abstract_final import resolve_final_inputs
    from batill.storage import file_hash
    resolve_final_inputs(diagnostic_root)  # Validate pinned inputs without changing them.
    diagnostic_config = json.loads((diagnostic_root / 'configs/abstract_final_input.json').read_text())
    diagnostic_audit_path = diagnostic_root / diagnostic_config['screening_decisions']
    diagnostic_audit = pd.read_csv(diagnostic_audit_path, keep_default_na=False)
    diagnostic_manifest = json.loads((diagnostic_audit_path.parent / 'manifest.json').read_text())
    diagnostic_evidence_path = diagnostic_audit_path.parent / 'abstract_mention_evidence.csv'
    assert file_hash(diagnostic_evidence_path) == diagnostic_manifest['artifacts_sha256']['abstract_mention_evidence.csv']
    diagnostic_evidence = pd.read_csv(diagnostic_evidence_path, keep_default_na=False)

    protected_title_diagnostic = diagnostic_audit.loc[diagnostic_audit.title_matched].copy()
    assert protected_title_diagnostic.retained.all()
    protected_title_diagnostic['would_fail_journal'] = ~protected_title_diagnostic.journal_keyword_match
    protected_title_diagnostic['would_fail_list'] = protected_title_diagnostic.pe_any_list_mention
    protected_title_diagnostic['would_fail_last_third'] = protected_title_diagnostic.pe_only_last_third
    diagnostic_flags = ['would_fail_journal', 'would_fail_list', 'would_fail_last_third']
    protected_title_diagnostic['criteria_triggered'] = protected_title_diagnostic[diagnostic_flags].sum(axis=1)
    diagnostic_summary = pd.DataFrame([
        {'Diagnostic': 'All protected title-match papers — still retained', 'Papers': len(protected_title_diagnostic)},
        {'Diagnostic': 'Would fail journal keywords', 'Papers': int(protected_title_diagnostic.would_fail_journal.sum())},
        {'Diagnostic': 'Would fail PE list-mention rule', 'Papers': int(protected_title_diagnostic.would_fail_list.sum())},
        {'Diagnostic': 'Would fail journal OR list (unique papers)', 'Papers': int(protected_title_diagnostic[diagnostic_flags[:2]].any(axis=1).sum())},
        {'Diagnostic': 'Would fail last-third-only rule', 'Papers': int(protected_title_diagnostic.would_fail_last_third.sum())},
        {'Diagnostic': 'Would fail any of these rules (unique papers)', 'Papers': int(protected_title_diagnostic.criteria_triggered.gt(0).sum())},
    ])
    display(diagnostic_summary.style.hide(axis='index'))
    protected_title_flagged = protected_title_diagnostic.loc[
        protected_title_diagnostic.criteria_triggered.gt(0)].sort_values(
            ['criteria_triggered', 'title', 'paper_id'], ascending=[False, True, True])

    def diagnostic_highlight(text):
        parts, end = [], 0
        for match in re.finditer(r'\bprivate[\s\-\u2010-\u2014\u2212]+equity\b', str(text), re.IGNORECASE):
            parts.extend([escape(str(text)[end:match.start()]), '<mark>', escape(match.group()), '</mark>'])
            end = match.end()
        return ''.join(parts) + escape(str(text)[end:])

    diagnostic_list_passages = (diagnostic_evidence.loc[diagnostic_evidence.is_list_mention]
        .groupby('paper_id').list_evidence.apply(lambda rows: list(dict.fromkeys(rows))).to_dict())
    diagnostic_rows = []
    for paper in protected_title_flagged.itertuples(index=False):
        passages = diagnostic_list_passages.get(paper.paper_id, [])
        evidence_html = ('<ul>' + ''.join('<li>' + diagnostic_highlight(text) + '</li>' for text in passages) + '</ul>'
                         if passages else '<p>No detected list mention.</p>')
        position_text = (f'{float(paper.pe_first_word_fraction):.1%} of abstract words before the first mention'
                         if paper.pe_mention_count else 'No PE phrase in abstract; position rule does not apply')
        diagnostic_rows.append(
            '<tr><td>' + escape(paper.paper_id) + '</td><td>' + diagnostic_highlight(paper.title)
            + '</td><td>' + escape(str(paper.year)) + '</td><td>' + escape(paper.journal)
            + '</td><td>' + ('Yes' if paper.would_fail_journal else 'No')
            + '</td><td>' + ('Yes' if paper.would_fail_list else 'No')
            + '</td><td>' + ('Yes' if paper.would_fail_last_third else 'No')
            + '</td><td><details><summary>Abstract and evidence</summary>'
            + '<p><b>Journal keywords matched:</b> ' + escape(paper.journal_matched_keywords or 'None') + '</p>'
            + '<p><b>PE mentions:</b> ' + str(paper.pe_mention_count) + '; in lists: ' + str(paper.pe_list_mention_count)
            + '; outside lists: ' + str(paper.pe_nonlist_mention_count) + '</p>'
            + '<p><b>Position:</b> ' + escape(position_text) + '</p>'
            + '<p><b>Detected list passages:</b></p>' + evidence_html
            + '<p><b>Full abstract:</b></p><div style="white-space:pre-wrap">'
            + diagnostic_highlight(paper.abstract) + '</div>'
            + '<p><b>Actual decision: retained — protected by title.</b></p></details></td></tr>')
    protected_title_diagnostic_html = (
        '<style>.title-diagnostic{border-collapse:collapse;width:100%;font-size:13px}'
        '.title-diagnostic th,.title-diagnostic td{border:1px solid #ccc;padding:7px;text-align:left;vertical-align:top}'
        '.title-diagnostic details{min-width:260px}.title-diagnostic summary{cursor:pointer}'
        '.title-diagnostic mark{background:#ffe58a;color:#151515}</style>'
        '<table class="title-diagnostic"><thead><tr>'
        + ''.join('<th>' + heading + '</th>' for heading in [
            'Paper ID', 'Full title', 'Year', 'Journal', 'Would fail journal',
            'Would fail list', 'Would fail last third', 'Inspection'])
        + '</tr></thead><tbody>' + ''.join(diagnostic_rows) + '</tbody></table>')
    print(f'Approved screening run: {diagnostic_config["screening_run"]}. Showing all {len(protected_title_flagged):,} flagged title matches; no display limit.')
    print('Diagnostic only: all title matches remain retained; no files or embeddings are changed.')
    display(HTML(protected_title_diagnostic_html))
    ''')
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python (batill)', 'language': 'python', 'name': 'batill'},
        'language_info': {'name': 'python', 'version': '3.12.7'},
    })
    nbf.write(notebook, ROOT / 'Abstract Screening.ipynb')


if __name__ == '__main__':
    main()
