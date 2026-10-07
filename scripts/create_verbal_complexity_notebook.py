"""Build the standalone readability notebook; execution is a separate step."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
md = nbf.v4.new_markdown_cell


def code(source, tag=None):
    cell = nbf.v4.new_code_cell(source.strip())
    if tag:
        cell.metadata['tags'] = [tag]
    return cell


nb = nbf.v4.new_notebook()
nb.cells = [
md(r"""# Verbal complexity over time

Calculate **Gunning Fog and Flesch Reading Ease for every paper** in the same **292 canonical works** used in the thesis. Both scores use exactly the same cleaned Marker extraction, before embedding preparation. Each paper has equal weight in the time summaries. Higher Fog means more complex prose; higher Flesch Reading Ease means easier prose.

The first time bin includes **all years through 1999 (inclusive)**, followed by calendar decades. Small bins remain included. Every paper uses the publication year recorded in the saved OpenAlex metadata. This consistently dates the indexed work, which can differ from the available PDF version.

All analysis code is in this notebook. Run it from the project root with **Python (batill)**. Outputs go to `reports/verbal_complexity/`; all inputs are local cached data; no API calls, PDF reading, or embedding inference are needed.

Dependencies: `python -m pip install -e '.[readability,analysis,notebook]'`.
The English pronunciation dictionary must be installed once with:
`python -m nltk.downloader -d .venv/nltk_data cmudict`.
Subsequent runs work offline.
"""),
code(r"""
from pathlib import Path
import sys, json, re, hashlib, unicodedata
from importlib.metadata import version
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from bs4 import BeautifulSoup
import nltk
import textstat

ROOT = Path.cwd().resolve()
if not (ROOT / 'src/batill').is_dir():
    raise RuntimeError('Run this notebook from the batill repository root.')
sys.path.insert(0, str(ROOT / 'src'))
from batill.storage import file_hash, read_json, write_json

OUT = ROOT / 'reports/verbal_complexity'
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'texts').mkdir(exist_ok=True)
nltk.data.path.insert(0, str(ROOT / '.venv/nltk_data'))
# Fail clearly instead of allowing textstat to download resources silently.
nltk.data.find('corpora/cmudict')
assert version('textstat') == '0.7.13', 'Install the pinned readability extra.'
textstat.set_lang('en_US')
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.max_rows', 30)

CORPUS_PATH = ROOT / 'reports/thesis_v2/evidence/corpus.csv'
METADATA_PATH = ROOT / 'outputs/citation_graph_openalex/9ad47e071ee12c2f/reviewed/nodes.csv'
EXTRACTION = ROOT / 'outputs/corpus/extraction'
EARLY_BIN_END = 1999
EXPECTED_PAPERS = 292
"""),
md("""## 1. Load the exact corpus and cached extraction

Selection uses PDF hashes, not filenames or row positions. Missing, duplicate, or mismatched documents stop execution. No paper is dropped for having an unusual score or belonging to a small time bin.
"""),
code(r"""
papers = pd.read_csv(CORPUS_PATH)
metadata = pd.read_csv(METADATA_PATH)
assert len(papers) == EXPECTED_PAPERS
assert papers.pdf_sha256.is_unique and papers.paper_id.is_unique
papers = papers.merge(
    metadata[['Key', 'openalex_id', 'openalex_year']],
    left_on='pdf_sha256', right_on='Key', how='left',
    validate='one_to_one', suffixes=('', '_metadata'), indicator=True,
)
assert papers['_merge'].eq('both').all()
papers = papers.drop(columns='_merge')

report = read_json(EXTRACTION / 'report.json')
selected = set(papers.pdf_sha256)
records = [r for r in report if r['pdf_sha256'] in selected
           and r['status'] in {'extracted', 'cached'}]
assert len(records) == EXPECTED_PAPERS
assert len({r['pdf_sha256'] for r in records}) == EXPECTED_PAPERS
documents, document_paths = {}, {}
for row in records:
    path = EXTRACTION / row['document_path']
    doc = read_json(path)
    assert doc['pdf_sha256'] == row['pdf_sha256']
    documents[row['pdf_sha256']] = doc
    document_paths[row['pdf_sha256']] = path
assert set(documents) == selected
print(f'Loaded {len(documents)} cached documents for {len(papers)} canonical works.')
"""),
md(r"""## 2. Use OpenAlex publication years

Papers are grouped by the publication year recorded in the saved OpenAlex metadata; readability is calculated from the available PDF text. The OpenAlex work year can differ from the date of that PDF version. The same year source is used for all 292 works.

The metadata were joined by PDF hash in section 1. This section validates that every OpenAlex year is a positive whole number and assigns the time bin. Missing or invalid years stop execution rather than switching to another date source. No new OpenAlex request is made. `year_audit.csv` records the paper, OpenAlex work ID, year, source, and bin.
"""),
code(r"""
def time_bin(year, early_end=1999):
    if pd.isna(year):
        return 'Unknown year'
    year = int(year)
    if 2020 <= year <= 2026:
        return '2020–2026'
    return f'≤{early_end}' if year <= early_end else f'{year // 10 * 10}–{year // 10 * 10 + 9}'


def assign_openalex_years(frame, early_end=1999):
    years = pd.to_numeric(frame['openalex_year'], errors='coerce')
    valid = years.notna() & np.isfinite(years) & years.gt(0) & years.mod(1).eq(0)
    if not valid.all():
        invalid = frame.loc[~valid, 'paper_id'].tolist()
        raise ValueError(f'Missing or invalid OpenAlex publication years: {invalid}')
    result = frame.copy()
    result['year'] = years.astype('Int64')
    result['year_source'] = 'openalex_year'
    result['period'] = result.year.map(lambda y: time_bin(y, early_end))
    return result
""", 'date_functions'),
code(r"""
papers = assign_openalex_years(papers, EARLY_BIN_END)
papers[['paper_id', 'pdf_sha256', 'title', 'openalex_id', 'openalex_year',
        'year', 'year_source', 'period']].to_csv(OUT / 'year_audit.csv', index=False)
print(f'OpenAlex publication years available for all {len(papers)} papers: '
      f'{papers.year.min()}–{papers.year.max()}')
display(papers.groupby('period', sort=True).size().rename('papers').to_frame())
"""),
md(r"""## 3. Define text-cleaning and scoring functions

Keep `Text` blocks in reading order, starting at an identified abstract or introduction when available. Exclude blocks marked unusable by the extraction, structural blocks (tables, figures, equations, captions, headings, footnotes, page headers/footers), and sections labelled references, bibliography, appendices, acknowledgements, or supplementary material. Remove detected inline math and superscript citation markers. Keep in-text author/year citations and ordinary punctuation; do not stem, lowercase, remove stopwords, or join embedding segments. Remove a small explicit set of publisher/history boilerplate patterns and standalone dates.

These are layout-based filters, not a perfect linguistic reconstruction. Original block IDs and decisions are saved in `block_audit.csv`; the exact scored prose is saved per paper in `texts/`. Short texts, replacement characters, missing opening headings, and extreme mean sentence lengths are flagged for inspection, not automatically dropped. Existing extraction has OCR disabled, which can affect old scans. Abstracts and main text are included; appendices and footnotes are outside the defined scoring scope.
"""),
code(r"""
YEAR = r'(?:19|20)\d{2}'
MONTH = r'(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Sept\.?|Oct\.?|Nov\.?|Dec\.?)'

EXCLUDED_SECTION = re.compile(
    r'^(?:references|bibliography|literature cited|appendix|appendices|online appendix|'
    r'internet appendix|supplement|acknowledg|declaration of|conflicts? of interest|funding statement)', re.I)
BOILERPLATE = re.compile(
    r'^(?:©|copyright\b|received\b|accepted\b|revised\b|available online\b|'
    r'published online\b|advance access\b|https?://|doi\s*:|jel\b|keywords?\s*:|'
    r'key words\s*:|e-?mail\b|corresponding author|this is an open access article|'
    r'this article is (?:distributed|licensed)|we (?:thank|are grateful|acknowledge)|'
    r'we gratefully acknowledge|the authors (?:thank|acknowledge))', re.I)


def heading(text):
    return re.sub(r'^(?:(?:\d+(?:\.\d+)*|[IVX]+)[.)]?\s+)', '', text.strip(), flags=re.I)


def prepare_prose(document):
    blocks = document['blocks']
    openings = [i for i, b in enumerate(blocks) if b['type'] == 'SectionHeader'
                and re.match(r'^(?:abstract|introduction)\b', heading(b['text']), re.I)]
    start = min(openings) if openings else 0
    paragraphs, audit = [], []
    excluded_section = False
    for i, block in enumerate(blocks):
        kind = block['type']
        if kind == 'SectionHeader':
            excluded_section = bool(EXCLUDED_SECTION.match(heading(block['text'])))
        hierarchy_excluded = any(EXCLUDED_SECTION.match(heading(s)) for s in block.get('section', []))
        reason = None
        if not block.get('include'):
            reason = block.get('exclusion_reason') or 'extraction_excluded'
        elif i < start:
            reason = 'front_matter'
        elif kind != 'Text':
            reason = f'block_type_{kind}'
        elif excluded_section or hierarchy_excluded:
            reason = 'excluded_section'
        text = ''
        if reason is None:
            soup = BeautifulSoup(block.get('html') or '', 'html.parser')
            for tag in soup.find_all(['math', 'sup']):
                tag.replace_with(' ')
            text = soup.get_text(' ', strip=True) if block.get('html') else block['text']
            text = unicodedata.normalize('NFKC', text).replace('\u00ad', '')
            text = re.sub(r'(?<=[A-Za-z])-\s*\n\s*(?=[a-z])', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            if not text:
                reason = 'empty'
            elif BOILERPLATE.match(text):
                reason = 'boilerplate'
            elif re.fullmatch(rf'(?:\d{{1,2}}\s+)?{MONTH}(?:\s+\d{{1,2}},?)?\s+{YEAR}[.*†‡]?', text, re.I):
                reason = 'standalone_date'
            elif not re.search(r'[A-Za-z]', text):
                reason = 'no_letters'
        keep = reason is None
        if keep:
            paragraphs.append(text)
        audit.append(dict(block_id=block['id'], page=block.get('page'), block_type=kind,
            retained=keep, reason=reason or 'prose',
            original_characters=len(block['text']), scored_characters=len(text) if keep else 0))
    return '\n\n'.join(paragraphs), audit, bool(openings)


def score_prose(text):
    words = textstat.lexicon_count(text, removepunct=True)
    if not words:
        return dict(word_count=0, sentence_count=0, complex_word_count=0,
            syllable_count=0, words_per_sentence=np.nan, complex_word_pct=np.nan,
            syllables_per_word=np.nan, fog=np.nan, flesch_reading_ease=np.nan)
    sentences = textstat.sentence_count(text)
    # Count occurrences, not unique vocabulary; threshold is THREE syllables.
    complex_words = textstat.difficult_words(text, syllable_threshold=3, unique=False)
    fog = textstat.gunning_fog(text)
    reconstructed = 0.4 * (words / sentences + 100 * complex_words / words)
    assert np.isclose(fog, reconstructed), 'Component counts differ from the pinned Fog implementation.'
    syllables = textstat.syllable_count(text)
    flesch = textstat.flesch_reading_ease(text)
    reconstructed_flesch = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
    assert np.isclose(flesch, reconstructed_flesch), 'Component counts differ from the pinned Flesch implementation.'
    return dict(word_count=words, sentence_count=sentences,
        complex_word_count=complex_words, words_per_sentence=words / sentences,
        complex_word_pct=100 * complex_words / words, fog=fog,
        syllable_count=syllables, syllables_per_word=syllables / words,
        flesch_reading_ease=flesch)
""", 'score_functions'),
md(r"""## 4. Clean the text and calculate both scores for all 292 papers

\[
\mathrm{Fog}=0.4\left(\frac{\text{words}}{\text{sentences}}+
100\frac{\text{complex-word occurrences}}{\text{words}}\right).
\]

\[
\mathrm{Flesch\ Reading\ Ease}=206.835
-1.015\frac{\text{words}}{\text{sentences}}
-84.6\frac{\text{syllables}}{\text{words}}.
\]

Implementation: **textstat 0.7.13**, English (`en_US`), with the CMU pronunciation dictionary and Pyphen fallback. Textstat's Fog difficult-word implementation uses a three-syllable threshold plus its easy-word vocabulary; this is an automated variant, not a manual implementation of all of Gunning's proper-name/compound-word exceptions. Count repeated complex words repeatedly. Flesch instead uses the average syllables per word across the whole text. We save the component counts and verify that they reconstruct both returned scores.

**Interpretation:** higher Fog indicates greater verbal complexity, whereas higher Flesch Reading Ease indicates easier reading. Conventional Flesch ranges are 90–100 very easy, 80–90 easy, 70–80 fairly easy, 60–70 standard, 50–60 fairly difficult, 30–50 difficult, and 0–30 very difficult (upper boundaries belong to the next range). Flesch is not an education-year scale. Values below 0 or above 100 are possible and are retained without clipping. These ranges are approximate guides; neither measure tests understanding or research quality. The two measures share sentence-length information, so agreement is not independent validation.

Implementation references: [Textstat](https://github.com/textstat/textstat), [pinned release](https://pypi.org/project/textstat/0.7.13/).
"""),
code(r"""
score_rows, block_rows, cleaned_texts = [], [], {}
for i, paper in enumerate(papers.itertuples(index=False), 1):
    document = documents[paper.pdf_sha256]
    text, audit, has_opening = prepare_prose(document)
    metrics = score_prose(text)
    flags = []
    if metrics['word_count'] < 100:
        flags.append('fewer_than_100_words')
    if metrics['sentence_count'] < 5:
        flags.append('fewer_than_5_sentences')
    if not has_opening:
        flags.append('opening_heading_not_detected')
    if '\ufffd' in text:
        flags.append('replacement_characters')
    if metrics['words_per_sentence'] > 60 or metrics['words_per_sentence'] < 5:
        flags.append('check_sentence_boundaries')
    text_path = OUT / 'texts' / f'{paper.paper_id}.txt'
    text_path.write_text(text, encoding='utf-8')
    cleaned_texts[paper.paper_id] = text
    score_rows.append(dict(pdf_sha256=paper.pdf_sha256, **metrics,
        text_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest(),
        text_path=str(text_path.relative_to(ROOT)),
        scored_blocks=sum(a['retained'] for a in audit),
        quality_flags='; '.join(flags)))
    block_rows.extend(dict(paper_id=paper.paper_id, pdf_sha256=paper.pdf_sha256, **a) for a in audit)
    if i % 50 == 0 or i == len(papers):
        print(f'Scored {i}/{len(papers)} papers')

results = papers.drop(columns=['text_sha256'], errors='ignore').merge(
    pd.DataFrame(score_rows), on='pdf_sha256', validate='one_to_one')
assert len(results) == EXPECTED_PAPERS and results.pdf_sha256.is_unique
results.to_csv(OUT / 'paper_scores.csv', index=False)
pd.DataFrame(block_rows).to_csv(OUT / 'block_audit.csv', index=False)
for metric in ['fog', 'flesch_reading_ease']:
    print(f'Finite {metric} scores: {np.isfinite(results[metric]).sum()}/{len(results)}')
display(results[['paper_id', 'title', 'year', 'period', 'word_count', 'sentence_count',
                 'complex_word_pct', 'syllable_count', 'syllables_per_word',
                 'fog', 'flesch_reading_ease', 'quality_flags']].sort_values('fog'))
"""),
md("""## 5. Compare time bins

Separate plots show Fog and Flesch Reading Ease on their own scales. In each plot, the line shows the **mean paper-level score**; boxes and dots show the distribution. Median and quartiles are also exported. Every scored paper counts once, regardless of its length. The first bin spans all years through 1999; the last bin is labelled 2020–2026 to reflect the corpus coverage. Horizontal positions are categories, not equal-duration intervals. All selected papers have a validated OpenAlex publication year. A decrease in Fog or an increase in Flesch Reading Ease indicates easier prose. No small bin is removed and no significance claim is inferred from the descriptive plots.
"""),
code(r"""
def summarize(frame, periods):
    return frame.groupby('period', observed=True).agg(
        n_papers=('pdf_sha256', 'size'), n_scored=('fog', 'count'),
        first_year=('year', 'min'), last_year=('year', 'max'),
        mean_fog=('fog', 'mean'), median_fog=('fog', 'median'),
        sd_fog=('fog', 'std'), q25=('fog', lambda s: s.quantile(.25)),
        q75=('fog', lambda s: s.quantile(.75)),
        mean_words_per_sentence=('words_per_sentence', 'mean'),
        mean_complex_word_pct=('complex_word_pct', 'mean'),
        n_flesch_scored=('flesch_reading_ease', 'count'),
        mean_flesch_reading_ease=('flesch_reading_ease', 'mean'),
        median_flesch_reading_ease=('flesch_reading_ease', 'median'),
        sd_flesch_reading_ease=('flesch_reading_ease', 'std'),
        q25_flesch_reading_ease=('flesch_reading_ease', lambda s: s.quantile(.25)),
        q75_flesch_reading_ease=('flesch_reading_ease', lambda s: s.quantile(.75)),
        mean_syllables_per_word=('syllables_per_word', 'mean'),
    ).reindex(periods)

dated = results.copy()
max_year = int(dated.year.max())
periods = [f'≤{EARLY_BIN_END}'] + [time_bin(d, EARLY_BIN_END) for d in range(2000, max_year // 10 * 10 + 1, 10)]
summary = summarize(dated, periods)
summary.to_csv(OUT / 'period_summary.csv')
display(summary.round(3))
print(f'Undated papers: {results.year.isna().sum()}')

def plot_readability(metric, mean_column, label, title, basename):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                             gridspec_kw={'height_ratios': [4, 1]})
    groups = [dated.loc[dated.period.eq(p), metric].dropna().to_numpy() for p in periods]
    positions = np.arange(1, len(periods) + 1)
    axes[0].boxplot(groups, positions=positions, showfliers=False,
        patch_artist=True, boxprops={'facecolor': '#dce8f3'},
        medianprops={'color': '#263746', 'linewidth': 1.8})
    rng = np.random.default_rng(42)
    for x, values in zip(positions, groups):
        axes[0].scatter(x + rng.uniform(-.16, .16, len(values)), values,
                        color='#477998', alpha=.35, s=17, edgecolors='none')
    axes[0].plot(positions, summary[mean_column], 'o-', color='#b34c32', label='Mean per paper')
    axes[0].set(ylabel=label, title=title)
    axes[0].legend(frameon=False)
    axes[0].grid(axis='y', alpha=.2)
    axes[1].bar(positions, [len(g) for g in groups], color='#477998', width=.55)
    for x, group in zip(positions, groups):
        axes[1].text(x, len(group), str(len(group)), ha='center', va='bottom')
    axes[1].set(ylabel='Papers', xlabel='OpenAlex publication year')
    axes[1].set_ylim(0, max(map(len, groups)) * 1.3)
    axes[1].set_xticks(positions, periods)
    fig.tight_layout()
    fig.savefig(OUT / f'{basename}.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUT / f'{basename}.pdf', bbox_inches='tight')
    plt.show()
    plt.close(fig)

plot_readability('fog', 'mean_fog', 'Gunning Fog index (higher = harder)',
                 'Verbal complexity of the 292-paper corpus', 'fog_by_period')
plot_readability('flesch_reading_ease', 'mean_flesch_reading_ease',
                 'Flesch Reading Ease (higher = easier)',
                 'Reading ease of the 292-paper corpus', 'flesch_reading_ease_by_period')
"""),
md("""## 6. Inspect scores and text

The analysis above includes every scored paper. The tables below show papers with the highest and lowest scores for each measure alongside their sentence-length and word-complexity components. Differences across decades can reflect topic composition as well as writing style; these are descriptive results for this collected corpus.

Use `INSPECT_PAPER` to read the exact scored text alongside its OpenAlex publication year and rejected blocks. This makes cleaning choices and extreme scores auditable without reopening embedding inputs.
"""),
code(r"""
inspection_columns = ['paper_id', 'title', 'fog', 'flesch_reading_ease',
                      'words_per_sentence', 'complex_word_pct', 'syllables_per_word', 'quality_flags']
for metric, direction in [('fog', 'higher = harder'), ('flesch_reading_ease', 'higher = easier')]:
    print(f'{metric}: five highest scores ({direction})')
    display(results.nlargest(5, metric)[inspection_columns])
    print(f'{metric}: five lowest scores ({direction})')
    display(results.nsmallest(5, metric)[inspection_columns])

INSPECT_PAPER = results.loc[results.fog.idxmax(), 'paper_id']
display(results.loc[results.paper_id.eq(INSPECT_PAPER),
    ['paper_id', 'title', 'year', 'year_source', 'word_count', 'fog', 'flesch_reading_ease', 'quality_flags']])
print(cleaned_texts[INSPECT_PAPER][:5000])
audit_frame = pd.DataFrame(block_rows)
display(audit_frame.loc[audit_frame.paper_id.eq(INSPECT_PAPER)].groupby('reason').size().rename('blocks').to_frame())
"""),
md("""## 7. Reproducibility and completeness

`paper_scores.csv` is the paper-level result, `period_summary.csv` contains summaries for both measures. `fog_by_period` and `flesch_reading_ease_by_period` are the PNG/PDF figure exports. OpenAlex year provenance, exact scored texts, and block decisions accompany them. The manifest records input hashes, software versions, and the dictionary hash. An empty extraction is missing data, never a score of zero. The final check requires both scores to be finite and a year to be present for every selected paper.
"""),
code(r"""
dictionary = nltk.data.find('corpora/cmudict/cmudict')
with dictionary.open() as stream:
    dictionary_hash = hashlib.sha256(stream.read()).hexdigest()
manifest = {
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'n_selected': len(results), 'n_scored': int(np.isfinite(results.fog).sum()),
    'n_flesch_scored': int(np.isfinite(results.flesch_reading_ease).sum()),
    'early_bin_end_inclusive': EARLY_BIN_END,
    'score_implementation': 'textstat.gunning_fog, en_US; difficult-word occurrences, threshold=3',
    'flesch_implementation': 'textstat.flesch_reading_ease, en_US; 206.835 - 1.015*words/sentences - 84.6*syllables/words; unclipped',
    'text_scope': 'Cached Marker Text blocks; abstract/main prose; structural/back-matter/boilerplate filters',
    'year_policy': 'Saved OpenAlex publication year (openalex_year) for every paper; no fallback',
    'versions': {p: version(p) for p in ['textstat', 'nltk', 'pyphen', 'pandas', 'numpy', 'matplotlib', 'beautifulsoup4']},
    'cmudict_sha256': dictionary_hash,
    'inputs': {str(p.relative_to(ROOT)): file_hash(p) for p in [CORPUS_PATH, METADATA_PATH, EXTRACTION / 'report.json']},
    'documents': {sha: file_hash(path) for sha, path in document_paths.items()},
    'paper_order': results.pdf_sha256.tolist(),
    'output_checksums': {p.name: file_hash(p) for p in OUT.glob('*.csv')},
}
write_json(OUT / 'run_manifest.json', manifest)
assert len(results) == EXPECTED_PAPERS
assert np.isfinite(results[['fog', 'flesch_reading_ease']].to_numpy()).all(), 'Inspect missing scores in paper_scores.csv.'
assert results.year.notna().all()
assert results.year.eq(results.openalex_year).all()
assert results.year_source.eq('openalex_year').all()
assert summary.n_scored.sum() == EXPECTED_PAPERS
assert summary.n_flesch_scored.sum() == EXPECTED_PAPERS
print(f'Complete: Fog and Flesch scores for all {len(results)} papers. Results saved to {OUT.relative_to(ROOT)}')
"""),
]
nb.metadata = {
    'kernelspec': {'display_name': 'Python (batill)', 'language': 'python', 'name': 'batill'},
    'language_info': {'name': 'python', 'version': '3.12'},
}
nbf.write(nb, ROOT / 'Verbal_Complexity.ipynb')

if __name__ == '__main__':
    print(ROOT / 'Verbal_Complexity.ipynb')
