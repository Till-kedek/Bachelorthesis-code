"""Reproducible, descriptive topic evidence for existing full-paper embeddings.

No model inference, text summarisation API or automatic paper exclusion occurs here.
"""
from itertools import combinations
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, ENGLISH_STOP_WORDS
from sklearn.metrics import adjusted_rand_score, silhouette_samples
from sklearn.metrics.pairwise import cosine_distances
from threadpoolctl import threadpool_limits

from .embedding import load_plans
from .graph_clustering import similarity_graph, leiden_partition
from .pipeline import validate_outputs
from .storage import fingerprint


# Formatting/URL artefacts only; economic and methodological words are retained.
EXTRA_STOP_WORDS = {'formula', 'http', 'https', 'www', 'com', 'org', 'pdf', 'doi', 'et', 'al'}
STOP_WORDS = set(ENGLISH_STOP_WORDS) | EXTRA_STOP_WORDS


def term_analyzer(text):
    """Unigrams and contiguous bigrams; never join across punctuation/stopwords.

    Stopwords are filtered *after* forming bigrams, so 'funds in markets' does not
    become the invented phrase 'funds markets'. No stemming or synonym merging.
    """
    text = unicodedata.normalize('NFKC', text).lower().replace('-', ' ')
    for sentence in re.split(r'[.!?;:\n\r,()\[\]{}]+', text):
        words = re.findall(r"[^\W\d_]+|\d+", sentence, flags=re.UNICODE)
        good = lambda w: len(w) >= 2 and w.isalpha() and w not in STOP_WORDS
        yield from (w for w in words if good(w))
        yield from (f'{a} {b}' for a, b in zip(words, words[1:]) if good(a) and good(b))


def load_topic_corpus(output_dir):
    """Read validated vectors and exact retained source pieces in matching row order."""
    output_dir = Path(output_dir)
    validation = validate_outputs(output_dir)
    plans = load_plans(output_dir / 'prepared')
    with np.load(output_dir / 'embeddings' / 'embeddings.npz', allow_pickle=False) as archive:
        embeddings = archive['embeddings'].copy()
        assert archive['pdf_sha256'].tolist() == [p['pdf_sha256'] for p in plans]
    rows, passages = [], []
    for i, plan in enumerate(plans):
        pieces = []
        prefix = f"Instruct: {plan['config']['instruction']}\nQuery: Title: {plan['title']}\n\n"
        for segment_index, segment in enumerate(plan['segments'], 1):
            if not segment['text'].startswith(prefix):
                raise ValueError('Unexpected prepared-input prefix')
            body = segment['text'][len(prefix):]
            cursor = 0
            for source in segment['sources']:
                length = source['end'] - source['start']
                text = body[cursor:cursor + length]
                if len(text) != length:
                    raise ValueError('Source offsets do not match prepared text')
                pieces.append(text)
                passages.append({'row': i, 'segment': segment_index, 'page': source['page'],
                    'block_id': source['block_id'], 'source_start': source['start'],
                    'source_end': source['end'], 'text': text})
                cursor += length + 2
            if cursor - 2 != len(body):
                raise ValueError('Prepared source ranges do not cover the input')
        # Lines separate source blocks, preventing artificial cross-block phrases.
        text = '\n'.join(pieces)
        rows.append({'row': i, 'paper_id': f'P{i + 1:03d}', 'pdf_sha256': plan['pdf_sha256'],
            'input_id': plan['id'], 'title': ' '.join(plan['title'].split()),
            'source_filename': plan['source_filename'], 'segments': len(plan['segments']),
            'characters': len(text), 'text_sha256': fingerprint(text), 'text': text})
    return pd.DataFrame(rows), embeddings, pd.DataFrame(passages), validation


def canonical_labels(labels):
    """One-based IDs by first member in the fixed corpus order."""
    mapping = {x: i + 1 for i, x in enumerate(dict.fromkeys(labels))}
    return np.array([mapping[x] for x in labels])


def fit_partitions(embeddings, ks=(2, 3, 4, 5, 6), *,
                   kmeans_seeds=(11, 22, 33, 44, 55), leiden_seeds=(42, 43, 44),
                   resolutions=None, neighbours=15, prefer_reference=True):
    """Retain K-means fits and Leiden fits with explicit selection provenance.

    Existing Leiden reference resolutions are preferred (2:.1, 3:.5, 5:.75, 6:1).
    Set prefer_reference=False for a new corpus without those historical choices.
    For any missing count, use the tested exact-count fit with highest seed ARI,
    breaking ties by lower resolution. The full sweep is returned for inspection.
    No partition is forced into a requested count by merging communities.
    """
    if resolutions is None:
        resolutions = np.round(np.arange(.1, 1.101, .025), 3).tolist()
    distance = cosine_distances(embeddings)
    distance = np.maximum((distance + distance.T) / 2, 0)
    np.fill_diagonal(distance, 0)
    solutions, metrics, seed_labels = {}, [], []

    def retain(name, method, labels, seed_ari, **extra):
        labels = canonical_labels(labels)
        scores = silhouette_samples(distance, labels, metric='precomputed')
        sizes = np.bincount(labels)[1:]
        solutions[name] = {'labels': labels, 'silhouettes': scores, 'method': method, **extra}
        metrics.append({'solution': name, 'method': method, 'k': len(sizes),
            'cosine_silhouette': scores.mean(), 'negative_fraction': np.mean(scores < 0),
            'seed_ARI': seed_ari, 'smallest': sizes.min(), 'largest': sizes.max(),
            'sizes': ', '.join(map(str, sizes)), **extra})

    with threadpool_limits(limits=1):
        for k in ks:
            runs = []
            for seed in kmeans_seeds:
                model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(embeddings)
                labels = model.labels_
                runs.append((model.inertia_, seed, labels))
                seed_labels.append({'run': f'kmeans_k{k}_seed{seed}', 'labels': canonical_labels(labels).tolist()})
            best = min(runs, key=lambda r: r[0])
            stability = np.mean([adjusted_rand_score(a[2], b[2]) for a, b in combinations(runs, 2)])
            retain(f'kmeans_k{k}', 'K-means', best[2], stability,
                   seed=best[1], resolution=None, inertia=float(best[0]),
                   selection='minimum inertia across five seeds')
            print(f'K-means k={k} complete', flush=True)

    graph = similarity_graph(embeddings, neighbours=neighbours)
    sweep, best_by_resolution = [], {}
    for resolution in resolutions:
        runs = [leiden_partition(graph, resolution, seed) for seed in leiden_seeds]
        best = max(runs, key=lambda r: r['quality'])
        stability = np.mean([adjusted_rand_score(a['labels'], b['labels']) for a, b in combinations(runs, 2)])
        k = len(np.unique(best['labels']))
        sweep.append({'resolution': resolution, 'k': k, 'seed_ARI': stability,
                      'seed': best['seed'], 'seed_cluster_counts': ', '.join(str(len(np.unique(r['labels']))) for r in runs)})
        best_by_resolution[resolution] = best
        seed_labels.extend({'run': f'leiden_r{resolution}_seed{r["seed"]}', 'labels': r['labels'].tolist()} for r in runs)
    sweep = pd.DataFrame(sweep)
    preferred = {2: .1, 3: .5, 5: .75, 6: 1.} if prefer_reference else {}
    for k in ks:
        choices = sweep.loc[sweep.k == k].sort_values(['seed_ARI', 'resolution'], ascending=[False, True])
        if choices.empty:
            print(f'No tested Leiden setting produced {k} communities', flush=True)
            continue
        reference = choices.loc[choices.resolution == preferred.get(k)]
        chosen = (reference if len(reference) else choices).iloc[0]
        why = 'existing reference resolution' if len(reference) else 'highest seed ARI among exact-count candidates'
        best = best_by_resolution[float(chosen.resolution)]
        retain(f'leiden_k{k}', 'Leiden', best['labels'], chosen.seed_ARI,
               seed=int(best['seed']), resolution=float(chosen.resolution), selection=why)
        print(f'Leiden k={k}: resolution={chosen.resolution}', flush=True)
    return solutions, pd.DataFrame(metrics), sweep, graph, seed_labels


def build_vocabulary(papers, min_df=3, max_df=.9):
    """One fixed vocabulary for every method/count; count once per paper for coverage."""
    vectorizer = CountVectorizer(analyzer=term_analyzer, min_df=min_df, max_df=max_df)
    counts = vectorizer.fit_transform(papers['text'])
    tfidf = TfidfTransformer(sublinear_tf=True).fit_transform(counts)
    return {'counts': counts, 'tfidf': tfidf, 'terms': vectorizer.get_feature_names_out(),
            'min_df': min_df, 'max_df': max_df}


def topic_terms(vocabulary, labels, top_n=20, *, min_cluster_fraction=0.1,
                min_cluster_papers=2):
    """c-TF-IDF and mean per-paper TF-IDF contrast, with paper-level prevalence.

    cTF(c,t) = count(c,t)/sum_t count(c,t)
    cIDF(t) = log(1 + mean_cluster_length / corpus_count(t))
    (BERTopic's default formula, using unrounded mean length.)
    """
    if not np.isfinite(min_cluster_fraction) or not 0 <= min_cluster_fraction <= 1:
        raise ValueError('min_cluster_fraction must be between 0 and 1')
    if not isinstance(min_cluster_papers, (int, np.integer)) or min_cluster_papers < 1:
        raise ValueError('min_cluster_papers must be a positive integer')
    counts, tfidf, terms = (vocabulary[k] for k in ('counts', 'tfidf', 'terms'))
    groups = sorted(np.unique(labels))
    pooled = np.vstack([np.asarray(counts[labels == c].sum(axis=0)).ravel() for c in groups])
    lengths = pooled.sum(axis=1)
    ctfidf = np.divide(pooled, lengths[:, None], out=np.zeros_like(pooled, dtype=float),
                      where=lengths[:, None] != 0) * np.log1p(lengths.mean() / pooled.sum(axis=0))
    binary = counts > 0
    records = []
    for i, cluster in enumerate(groups):
        inside = labels == cluster
        in_count = np.asarray(binary[inside].sum(axis=0)).ravel()
        out_count = np.asarray(binary[~inside].sum(axis=0)).ravel()
        contrast = np.asarray(tfidf[inside].mean(axis=0) - tfidf[~inside].mean(axis=0)).ravel()
        # Display support threshold prevents one long paper from defining a label.
        support = max(min_cluster_papers, int(np.ceil(min_cluster_fraction * inside.sum())))
        for method, scores in [('c_tf_idf', ctfidf[i]), ('tfidf_contrast', contrast)]:
            eligible = (in_count >= support) & (scores > 0)
            top = np.flatnonzero(eligible)[np.argsort(-scores[eligible], kind='stable')[:top_n]]
            for rank, term in enumerate(top, 1):
                pin, pout = in_count[term] / inside.sum(), out_count[term] / (~inside).sum()
                records.append({'cluster': int(cluster), 'ranking': method, 'rank': rank,
                    'term': terms[term], 'score': scores[term], 'inside_count': int(in_count[term]),
                    'inside_n': int(inside.sum()), 'outside_count': int(out_count[term]),
                    'outside_n': int((~inside).sum()), 'inside_prevalence': pin,
                    'outside_prevalence': pout, 'prevalence_gap': pin - pout})
    return pd.DataFrame(records)


def describe_partition(papers, embeddings, passages, result, terms, random_seed=42):
    """Save every assignment plus central/boundary/random examples and source excerpts."""
    labels, sil = result['labels'], result['silhouettes']
    membership = papers.drop(columns='text').copy()
    membership['cluster'] = labels
    membership['cosine_silhouette'] = sil
    normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarity = normalized @ normalized.T
    summaries, examples = [], []
    for cluster in sorted(np.unique(labels)):
        members = np.flatnonzero(labels == cluster)
        centrality = similarity[np.ix_(members, members)].sum(axis=1) - 1
        centrality /= max(1, len(members) - 1)
        ranked = members[np.argsort(-centrality, kind='stable')]
        central = ranked[:3]
        boundary = members[np.argsort(sil[members], kind='stable')[:2]]
        rest = np.setdiff1d(members, np.union1d(central, boundary))
        random = np.random.default_rng(random_seed + int(cluster)).choice(rest, size=min(2, len(rest)), replace=False)
        rows = terms[terms.cluster == cluster]
        ranked_terms = rows[rows.ranking == 'tfidf_contrast'].sort_values('rank')['term'].tolist()
        summaries.append({'cluster': int(cluster), 'papers': len(members),
            'mean_silhouette': sil[members].mean(), 'negative_fraction': np.mean(sil[members] < 0),
            'membership_id': fingerprint(sorted(papers.iloc[members]['pdf_sha256'].tolist())),
            'ctfidf_terms': '; '.join(rows[rows.ranking == 'c_tf_idf'].sort_values('rank').term.head(10)),
            'contrast_terms': '; '.join(ranked_terms[:10]),
            'representatives': '\n'.join(f"{papers.iloc[j].paper_id}: {papers.iloc[j].title}" for j in central)})
        roles = {}
        for role, indices in [('representative', central), ('boundary', boundary), ('random', random)]:
            for index in indices:
                roles.setdefault(int(index), []).append(role)
        for index, role in roles.items():
            options = passages[(passages.row == index) & (passages.text.str.len() >= 100)]
            found = None
            for phrase in ranked_terms[:10]:
                for _, passage in options.iterrows():
                    pattern = r'(?<!\w)' + r'\s+'.join(map(re.escape, phrase.split())) + r'(?!\w)'
                    match = re.search(pattern, passage.text, re.I)
                    if match:
                        begin = max(0, match.start() - 130)
                        end = min(len(passage.text), begin + 600)
                        found = {**passage.to_dict(), 'term': phrase, 'excerpt_start': begin,
                                 'excerpt_end': end, 'excerpt': passage.text[begin:end]}
                        break
                if found is not None:
                    break
            if found is None:
                options = passages[passages.row == index]
                p = options.iloc[0]
                found = {**p.to_dict(), 'term': '', 'excerpt_start': 0,
                         'excerpt_end': min(600, len(p.text)), 'excerpt': p.text[:600]}
            examples.append({**membership.iloc[index].to_dict(), 'selection': '; '.join(role),
                **{k: found[k] for k in ('segment', 'page', 'block_id', 'source_start', 'term',
                                        'excerpt_start', 'excerpt_end', 'excerpt')}})
    return pd.DataFrame(summaries), membership, pd.DataFrame(examples)


def partition_agreement(left, right):
    """Label-invariant ARI and maximum one-to-one matching; return contingency too."""
    table = pd.crosstab(pd.Series(left, name='left_cluster'), pd.Series(right, name='right_cluster'))
    a, b = linear_sum_assignment(-table.to_numpy())
    return {'ARI': adjusted_rand_score(left, right),
            'matched_papers': int(table.to_numpy()[a, b].sum()),
            'papers': len(left), 'matched_fraction': float(table.to_numpy()[a, b].sum() / len(left))}, table


def compare_solutions(solutions, folder):
    """Cross-method and adjacent-count transitions; no assumption that fits are nested."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for a, b in combinations(solutions, 2):
        ka, kb = len(np.unique(solutions[a]['labels'])), len(np.unique(solutions[b]['labels']))
        same_method = solutions[a]['method'] == solutions[b]['method']
        if (not same_method and ka == kb) or (same_method and abs(ka - kb) == 1):
            scores, table = partition_agreement(solutions[a]['labels'], solutions[b]['labels'])
            table.to_csv(folder / f'{a}__{b}.csv')
            rows.append({'left': a, 'right': b, 'comparison': 'adjacent counts' if same_method else 'same count', **scores})
    return pd.DataFrame(rows)


def write_review_cards(profiles, term_scores, review_papers, proposals, destination,
                       allow_unmatched=False):
    """Use exact-match labels; optionally leave changed memberships pending review."""
    if proposals.proposed_label.isna().any() or proposals.proposed_label.astype(str).str.strip().eq('').any():
        raise ValueError('Each label proposal must contain a nonempty name')
    proposal_columns = ['solution', 'cluster', 'membership_id', 'proposed_label', 'status']
    proposal_columns += [name for name in ('rationale', 'review_caveat') if name in proposals]
    joined = profiles.merge(proposals[proposal_columns],
        on=['solution', 'cluster', 'membership_id'], how='left', validate='one_to_one')
    joined['proposal_matched'] = joined.proposed_label.notna()
    if (not joined.proposal_matched.all() or len(joined) != len(proposals)) and not allow_unmatched:
        raise ValueError('Label proposals do not match this run; review changed memberships before reusing names')
    if allow_unmatched:
        missing = ~joined.proposal_matched
        joined.loc[missing, 'proposed_label'] = 'Name pending review'
        joined.loc[missing, 'status'] = 'Membership changed; inspect current papers before assigning a name'
    lines = ['# Cluster review cards', '',
        'Provisional analyst labels; verify candidates and screen the full membership before accepting them.',
        'Each cluster lists three central titles. Full boundary/random examples and source excerpts are in review_papers.csv.', '']
    for _, row in joined.iterrows():
        lines.extend([f"## {row.solution}, cluster {row.cluster}: {row.proposed_label}",
            f"**{row.papers} papers**; mean silhouette {row.mean_silhouette:.3f}; "
            f"negative silhouettes {row.negative_fraction:.1%}. {row.status}.", '',
            f"c-TF-IDF: {row.ctfidf_terms}", '', f"TF-IDF contrast: {row.contrast_terms}", ''])
        for column, heading in [('rationale', 'Naming evidence'), ('review_caveat', 'Review note')]:
            value = row.get(column)
            if pd.notna(value) and str(value).strip():
                lines.extend([f'{heading}: {value}', ''])
        lines.extend(['| Top contrast term | Papers inside | Papers outside |', '|---|---:|---:|'])
        terms = term_scores[(term_scores.solution == row.solution) & (term_scores.cluster == row.cluster)
                            & (term_scores.ranking == 'tfidf_contrast')].sort_values('rank').head(5)
        lines.extend(f'| {t.term} | {t.inside_count}/{t.inside_n} | {t.outside_count}/{t.outside_n} |'
                     for _, t in terms.iterrows())
        lines.extend(['', 'Representative papers:', ''])
        lines.extend(f'- {title}' for title in row.representatives.splitlines())
        sample = review_papers[(review_papers.solution == row.solution) & (review_papers.cluster == row.cluster)
                              & review_papers.selection.str.contains('boundary')]
        lines.extend(['', 'Boundary papers to check:', ''])
        lines.extend(f'- {p.paper_id}: {p.title} (silhouette {p.cosine_silhouette:.3f}; {p.source_filename})'
                     for _, p in sample.iterrows())
        lines.extend(['', ''])
    Path(destination).write_text('\n'.join(lines), encoding='utf-8')
    return joined
