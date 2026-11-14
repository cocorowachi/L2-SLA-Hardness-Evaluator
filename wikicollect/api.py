"""MediaWiki API client: descriptive User-Agent, sequential rate-limited requests,
maxlag/429 backoff, and an on-disk response cache so every stage is resumable."""
import hashlib
import json
import time
from collections.abc import Iterator
from pathlib import Path

import requests


class APIError(RuntimeError):
    pass


class WikiAPI:
    def __init__(self, contact: str, cache_dir: Path, min_interval=0.5, maxlag=5, max_retries=6):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = (
            f"wikicollect/0.1 (SLA difficulty research; contact: {contact or 'unset'}) "
            f"python-requests/{requests.__version__}"
        )
        self.cache_dir = Path(cache_dir)
        self.min_interval = min_interval
        self.maxlag = maxlag
        self.max_retries = max_retries
        self._last = 0.0
        self.n_requests = 0
        self.max_requests: int | None = None  # cap on uncached requests (used when no contact is set)

    def get(self, site: str, **params) -> dict:
        """GET api.php on `site` ("en" -> en.wikipedia.org, or a full host). Cached forever."""
        host = site if "." in site else f"{site}.wikipedia.org"
        params = {"format": "json", "formatversion": "2", "maxlag": self.maxlag, **params}
        key = hashlib.sha1(json.dumps([host, sorted(params.items())], ensure_ascii=False).encode()).hexdigest()
        path = self.cache_dir / key[:2] / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text("utf-8"))
        data = self._fetch(host, params)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), "utf-8")
        return data

    def query_all(self, site: str, **params) -> Iterator[dict]:
        """Run action=query, following `continue`, yielding each response's `query` block."""
        cont: dict = {}
        while True:
            data = self.get(site, action="query", **params, **cont)
            if "query" in data:
                yield data["query"]
            if "continue" not in data:
                return
            cont = data["continue"]

    def _fetch(self, host: str, params: dict) -> dict:
        if self.max_requests is not None and self.n_requests >= self.max_requests:
            raise APIError(f"request cap of {self.max_requests} reached; set api.contact in config.toml")
        for attempt in range(self.max_retries):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            self.n_requests += 1
            try:
                r = self.session.get(f"https://{host}/w/api.php", params=params, timeout=30)
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                # Wikimedia: honour Retry-After; without it wait at least 5 s, backing off.
                time.sleep(float(r.headers.get("Retry-After", 5 * 2 ** attempt)))
                continue
            r.raise_for_status()
            data = r.json()
            err = data.get("error")
            if err and err.get("code") == "maxlag":
                time.sleep(float(r.headers.get("Retry-After", 5)))
                continue
            if err:
                raise APIError(f"{host}: {err.get('code')}: {err.get('info')}")
            return data
        raise APIError(f"{host}: gave up after {self.max_retries} attempts ({params})")


def query_pages(api: WikiAPI, site: str, titles: list[str], **params) -> dict[str, dict | None]:
    """Query up to 50 titles; map each *requested* title to its page record (None if missing),
    following normalization and redirects and merging list props across continuations."""
    pages: dict[str, dict] = {}
    rename: dict[str, str] = {}
    for q in api.query_all(site, titles="|".join(titles), **params):
        for step in ("normalized", "redirects"):
            for n in q.get(step, []):
                rename[n["from"]] = n["to"]
        for p in q.get("pages", []):
            merged = pages.setdefault(p["title"], {})
            for k, v in p.items():
                if isinstance(v, list) and isinstance(merged.get(k), list):
                    merged[k] = merged[k] + v
                else:
                    merged[k] = v
    out = {}
    for t in titles:
        final = t
        for _ in range(3):  # normalized -> redirect -> (rare) normalized again
            final = rename.get(final, final)
        p = pages.get(final)
        out[t] = None if p is None or p.get("missing") or p.get("invalid") else p
    return out


def chunks(items: list, n: int) -> Iterator[list]:
    for i in range(0, len(items), n):
        yield items[i:i + n]
