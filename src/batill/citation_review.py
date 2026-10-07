"""Apply explicit, auditable identity-review decisions without making API requests."""
from difflib import SequenceMatcher
from pathlib import Path
import re

import pandas as pd

from .citation_graph import normalize_title, normalize_doi
from .storage import fingerprint, read_json


def apply_match_review(matches, review_file):
    """Return reviewed matches and an audit; never change the corpus or raw matches.

    Decisions are keyed by PDF hash and pin both the original proposal and selected
    candidate metadata. A changed title, DOI or proposed identity requires a new
    review. Held papers remain present but cannot supply graph edges. Review entries
    for papers subsequently excluded from the corpus are reported and not applied.
    """
    config = read_json(review_file)
    if config.get('schema_version') != 1:
        raise ValueError('Unsupported citation-review schema')
    result = matches.fillna('').copy()
    if result.Key.duplicated().any():
        raise ValueError('Duplicate PDF keys in match table')
    result = result.reset_index(drop=True)
    for field in ('review_reason', 'review_version_status', 'review_original_s2_id'):
        result[field] = ''
    row_by_key = {key: i for i, key in enumerate(result.Key)}
    seen, audit = set(), []
    for entry in config['papers']:
        key = entry['pdf_sha256']
        if key in seen or not re.fullmatch(r'[0-9a-f]{64}', key):
            raise ValueError('Duplicate or invalid PDF key in review file')
        seen.add(key)
        if entry['decision'] not in ('accept', 'hold') or not entry.get('reason'):
            raise ValueError('Review decisions must be accept/hold with a reason')
        if key not in row_by_key:
            audit.append({'Key': key, 'paper_id': entry['paper_id'], 'decision': 'not_in_current_selection',
                          'category': entry['category'], 'reason': entry['reason']})
            continue
        i = row_by_key[key]
        row = result.iloc[i]
        if (row.query_title != entry['expected_query_title']
                or normalize_doi(row.doi) != normalize_doi(entry['expected_doi'])
                or row.s2_id != entry['expected_s2_id']):
            raise ValueError(f"Stale review for {row.paper_id}: title/DOI/proposal changed. "
                             'Review the new match or remove its old decision before continuing.')
        result.loc[i, 'review_reason'] = entry['reason']
        result.loc[i, 'review_version_status'] = entry['version_status']
        result.loc[i, 'review_original_s2_id'] = row.s2_id
        if entry['decision'] == 'accept':
            candidate = entry.get('chosen_candidate') or {}
            if not re.fullmatch(r'[0-9a-f]{40}', candidate.get('paperId', '')):
                raise ValueError(f'Invalid approved Semantic Scholar identity for {row.paper_id}')
            if entry['version_status'] not in ('publication_identity_confirmed', 'same_work_version_not_verified'):
                raise ValueError(f'Missing version assessment for {row.paper_id}')
            updates = {
                's2_id': candidate['paperId'], 's2_title': candidate.get('title', ''),
                's2_year': candidate.get('year'),
                's2_doi': normalize_doi((candidate.get('externalIds') or {}).get('DOI', '')),
                's2_url': candidate.get('url') or f"https://www.semanticscholar.org/paper/{candidate['paperId']}",
                'reference_count': candidate.get('referenceCount'),
                'citation_count': candidate.get('citationCount'),
                'title_similarity': SequenceMatcher(None, normalize_title(row.query_title),
                                                    normalize_title(candidate.get('title', ''))).ratio(),
                'approved': True, 'match_status': 'reviewed_' + entry['category'],
            }
            for column, value in updates.items():
                result.loc[i, column] = value
        else:
            result.loc[i, 'approved'] = False
            result.loc[i, 'match_status'] = 'review_pending_' + entry['category']
        audit.append({'Key': key, 'paper_id': row.paper_id, 'decision': entry['decision'],
            'category': entry['category'], 'query_title': row.query_title,
            'original_s2_id': row.s2_id, 'reviewed_s2_id': result.loc[i, 's2_id'],
            'version_status': entry['version_status'], 'reason': entry['reason'],
            'scope_note': entry.get('scope_note', ''), 'suggested_lookup': entry.get('suggested_lookup', '')})
    active = result.loc[result.approved.eq(True)]
    if active.s2_id.eq('').any() or active.s2_id.duplicated().any():
        raise ValueError('Review creates empty/duplicate approved identities; resolve them before retrieval')
    result.attrs['review_config_id'] = fingerprint(config)
    result.attrs['review_file'] = str(Path(review_file))
    return result, pd.DataFrame(audit)
