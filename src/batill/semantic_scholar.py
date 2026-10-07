"""Small cached Semantic Scholar Graph API client; credentials are never persisted."""
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
import logging
import time
from urllib.parse import quote

from .storage import fingerprint, read_json, write_json

BASE_URL = 'https://api.semanticscholar.org/graph/v1'
PAPER_FIELDS = 'title,year,authors,externalIds,url,referenceCount,citationCount'
logger = logging.getLogger(__name__)


class ScholarError(RuntimeError):
    def __init__(self, status, message):
        self.status = status
        super().__init__(message)


class ScholarClient:
    """Serial requests at <=1 RPS, retries and a reusable response cache.

    Cache identities contain only endpoint/parameters. 429/5xx and transient
    connection failures retry; authentication errors stop immediately. The caller
    controls refresh explicitly. No credentials or request headers enter files.
    """
    def __init__(self, api_key, cache_dir, *, interval=2.0, retries=6,
                 refresh=False, session=None, sleep=time.sleep, clock=time.monotonic):
        if not api_key or interval < 1 or retries < 0:
            raise ValueError('Supply an API key, interval >=1 second and retries >=0')
        import requests
        self.session = session or requests.Session()
        self.session.headers.update({'x-api-key': api_key, 'User-Agent': 'batill-thesis-citation-graph/1'})
        self.cache_dir = Path(cache_dir)
        self.interval, self.retries, self.refresh = interval, retries, refresh
        self.sleep, self.clock, self.last_request = sleep, clock, None
        self.used_cache_files = set()

    def get(self, route, params=None):
        import requests
        params = params or {}
        identity = {'base_url': BASE_URL, 'route': route, 'params': params}
        path = self.cache_dir / f'{fingerprint(identity)}.json'
        if path.exists() and not self.refresh:
            cached = read_json(path)
            if cached['request'] != identity:
                raise ValueError('API cache identity mismatch')
            self.used_cache_files.add(path)
            return cached['payload']
        for attempt in range(self.retries + 1):
            if self.last_request is not None:
                self.sleep(max(0, self.interval - (self.clock() - self.last_request)))
            self.last_request = self.clock()
            try:
                response = self.session.get(f'{BASE_URL}/{route}', params=params, timeout=(10, 60))
            except requests.RequestException:
                if attempt == self.retries:
                    raise ScholarError(0, 'Semantic Scholar connection failed after retries; rerun to resume.') from None
                self.sleep(min(60, 2 ** (attempt + 1)))
                continue
            status = response.status_code
            if status in (200, 404):
                payload = response.json() if status == 200 else None
                write_json(path, {'request': identity, 'status': status,
                    'fetched_at_utc': datetime.now(timezone.utc).isoformat(), 'payload': payload})
                self.used_cache_files.add(path)
                return payload
            if status in (401, 403):
                raise ScholarError(status, f'Semantic Scholar HTTP {status}: check API key/access; key was not logged.')
            if status == 429:
                # Keep the slower pace for the lifetime of this client.
                self.interval = max(self.interval, min(10., self.interval * 2))
            if status == 429 or status >= 500:
                if attempt < self.retries:
                    # Fixed rate-limit cooldown; respect a longer provider Retry-After.
                    backoff = 20 if status == 429 else min(60, 2 ** (attempt + 1))
                    retry_after = response.headers.get('Retry-After', '')
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        try:
                            delay = (parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds()
                        except (ValueError, TypeError, OverflowError):
                            delay = backoff
                    delay = max(backoff, delay)
                    logger.warning('Semantic Scholar HTTP %s: waiting %.0fs before retry %s/%s '
                                   '(request interval %.1fs). Completed responses are cached.',
                                   status, delay, attempt + 1, self.retries, self.interval)
                    self.sleep(delay)
                    continue
            if status == 429:
                raise ScholarError(status, 'Semantic Scholar is still rate-limiting requests after retries. '
                    'Pause and rerun with refresh=False to reuse completed responses; no paper was skipped.')
            raise ScholarError(status, f'Semantic Scholar HTTP {status} for {route}; rerun after resolving the error.')

    def paper(self, identifier):
        return self.get(f'paper/{quote(str(identifier), safe="")}', {'fields': PAPER_FIELDS})

    def search(self, title, limit=5):
        response = self.get('paper/search', {'query': title.replace('-', ' '),
                                             'limit': limit, 'fields': PAPER_FIELDS})
        return (response or {}).get('data', [])

    def references(self, paper_id):
        """Yield all reference pages using the API's next offset, not nested fields."""
        offset, visited = 0, set()
        while True:
            if offset in visited:
                raise ScholarError(0, 'Reference pagination repeated an offset; incomplete retrieval.')
            visited.add(offset)
            response = self.get(f'paper/{quote(paper_id, safe="")}/references',
                                {'fields': 'title,year,externalIds', 'limit': 1000, 'offset': offset})
            if response is None:
                raise ScholarError(404, 'Reference endpoint did not find the resolved paper.')
            if not isinstance(response.get('data'), list):
                raise ScholarError(0, 'Unexpected reference response; cannot claim complete retrieval.')
            yield response['data']
            if response.get('next') is None:
                break
            next_offset = int(response['next'])
            if next_offset <= offset:
                raise ScholarError(0, 'Reference pagination did not advance; incomplete retrieval.')
            offset = next_offset
