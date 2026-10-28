#!/usr/bin/env python3
"""
Download full texts for the papers listed in data/papers.tsv.

For each paperId the script
1. asks the Semantic Scholar Graph API for the open-access PDF link,
2. falls back to Europe PMC (PMC open-access PDF, found by PMID or DOI),
3. saves the PDF and converts it to plain text with pypdf.

Output:
- <pdf-dir>/<paperId>.pdf
- <output-dir>/<paperId>.txt
- <status-file>: one row per attempt (paperId, status, source, detail, timestamp)

The script is resumable. Papers with an existing text file are skipped, and papers
that failed earlier are skipped unless --retry-failed is given.

Environment:
- SEMANTIC_SCHOLAR_API_KEY: API key for the Semantic Scholar Graph API
- CONTACT_EMAIL: optional address added to the User-Agent header
"""

import argparse
import csv
import io
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import requests
from pypdf import PdfReader

S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
EPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EPMC_PDF_URL = "https://europepmc.org/backend/ptpmcrender.fcgi"
S2_BATCH_SIZE = 100
MAX_PDF_BYTES = 150 * 1024 * 1024
MIN_TEXT_CHARS = 2000
STATUS_HEADER = ["paperId", "status", "source", "detail", "timestamp"]


class RateLimiter:
    """Allow at most one call every min_interval seconds across threads."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._next - now
            if delay > 0:
                time.sleep(delay)
                now = time.monotonic()
            self._next = now + self.min_interval


def make_session(contact: str | None) -> requests.Session:
    session = requests.Session()
    agent = "biognosia-corpus/0.1"
    if contact:
        agent += f" (mailto:{contact})"
    session.headers["User-Agent"] = agent
    return session


def request_with_retry(session, method, url, limiter, retries=5, **kwargs):
    """Send a request, honouring Retry-After on 429/5xx responses."""
    delay = 2.0
    response = None
    for _ in range(retries):
        limiter.wait()
        try:
            response = session.request(method, url, timeout=60, **kwargs)
        except requests.RequestException:
            time.sleep(delay)
            delay *= 2
            continue
        if response.status_code not in (429, 500, 502, 503, 504):
            return response
        retry_after = response.headers.get("Retry-After")
        time.sleep(float(retry_after) if retry_after and retry_after.isdigit() else delay)
        delay *= 2
    return response


def lookup_open_access_pdfs(session, limiter, api_key: str | None, paper_ids: list[str]) -> dict[str, str]:
    """Return {paperId: pdf url} for the papers that Semantic Scholar lists as open access."""
    headers = {"x-api-key": api_key} if api_key else {}
    response = request_with_retry(
        session,
        "POST",
        S2_BATCH_URL,
        limiter,
        params={"fields": "openAccessPdf"},
        json={"ids": paper_ids},
        headers=headers,
    )
    if response is None or response.status_code != 200:
        return {}
    urls = {}
    for paper_id, item in zip(paper_ids, response.json(), strict=True):
        pdf = (item or {}).get("openAccessPdf") or {}
        if pdf.get("url"):
            urls[paper_id] = pdf["url"]
    return urls


def find_pmcid(session, limiter, pmid: str, doi: str) -> str | None:
    """Look up the PMC id of an open-access article in Europe PMC by PMID, then DOI."""
    queries = []
    if pmid:
        queries.append(f"EXT_ID:{pmid} AND SRC:MED")
    if doi:
        queries.append(f'DOI:"{doi}"')
    for query in queries:
        response = request_with_retry(
            session,
            "GET",
            EPMC_SEARCH_URL,
            limiter,
            params={"query": query, "format": "json", "resultType": "lite", "pageSize": 1},
        )
        if response is None or response.status_code != 200:
            continue
        results = response.json().get("resultList", {}).get("result", [])
        if results and results[0].get("pmcid") and results[0].get("isOpenAccess") == "Y":
            return results[0]["pmcid"]
    return None


def fetch_pdf(session, limiter, url: str, params: dict | None = None) -> bytes | None:
    """Download a PDF and return its bytes, or None if the response is not a PDF."""
    response = request_with_retry(session, "GET", url, limiter, params=params, stream=True, allow_redirects=True)
    if response is None or response.status_code != 200:
        return None
    content = bytearray()
    for block in response.iter_content(1 << 20):
        content.extend(block)
        if len(content) > MAX_PDF_BYTES:
            return None
    return bytes(content) if content.startswith(b"%PDF") else None


def pdf_to_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(pages).strip()


def load_papers(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_status(path: Path) -> dict[str, str]:
    """Return the last recorded status for each paperId."""
    latest = {}
    if path.exists():
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                latest[row["paperId"]] = row["status"]
    return latest


class StatusLog:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with open(path, "w", newline="", encoding="utf-8") as handle:
                csv.writer(handle, delimiter="\t").writerow(STATUS_HEADER)

    def write(self, paper_id: str, status: str, source: str = "", detail: str = "") -> None:
        stamp = datetime.now(UTC).isoformat(timespec="seconds")
        with self._lock, open(self.path, "a", newline="", encoding="utf-8") as handle:
            csv.writer(handle, delimiter="\t").writerow([paper_id, status, source, detail, stamp])


def try_source(session, limiter, paper_id, source, url, params, args, log) -> bool:
    """Download one candidate PDF, convert it to text and store both. Return True on success."""
    pdf_bytes = fetch_pdf(session, limiter, url, params)
    if pdf_bytes is None:
        return False
    try:
        text = pdf_to_text(pdf_bytes)
    except Exception as exc:
        log.write(paper_id, "pdf_error", source, str(exc)[:200])
        return False
    if len(text) < MIN_TEXT_CHARS:
        log.write(paper_id, "short_text", source, f"{len(text)} chars")
        return False
    args.pdf_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.pdf_dir / f"{paper_id}.pdf").write_bytes(pdf_bytes)
    (args.output_dir / f"{paper_id}.txt").write_text(text, encoding="utf-8")
    log.write(paper_id, "ok", source)
    return True


def process_paper(session, limiters, paper, s2_url, args, log) -> bool:
    paper_id = paper["paperId"]
    tried = False
    if s2_url:
        tried = True
        if try_source(session, limiters["generic"], paper_id, "semantic_scholar", s2_url, None, args, log):
            return True
    pmcid = find_pmcid(session, limiters["epmc"], paper.get("pmid", ""), paper.get("doi", ""))
    if pmcid:
        tried = True
        params = {"accid": pmcid, "blobtype": "pdf"}
        if try_source(session, limiters["epmc"], paper_id, "europe_pmc", EPMC_PDF_URL, params, args, log):
            return True
    log.write(paper_id, "download_failed" if tried else "no_open_access")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download full texts for the paper list")
    parser.add_argument("--papers-tsv", type=Path, required=True, help="Paper list (paperId, doi, pmid, ...)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for <paperId>.txt files")
    parser.add_argument("--pdf-dir", type=Path, required=True, help="Directory for <paperId>.pdf files")
    parser.add_argument("--status-file", type=Path, required=True, help="Download status log (TSV)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel download threads")
    parser.add_argument("--s2-interval", type=float, default=1.1, help="Seconds between Semantic Scholar requests")
    parser.add_argument("--epmc-interval", type=float, default=0.2, help="Seconds between Europe PMC requests")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N papers (0 = all)")
    parser.add_argument("--retry-failed", action="store_true", help="Retry papers that failed earlier")
    args = parser.parse_args()

    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if not api_key:
        print("SEMANTIC_SCHOLAR_API_KEY is not set. Semantic Scholar requests will be rate limited.", file=sys.stderr)

    papers = load_papers(args.papers_tsv)
    status = load_status(args.status_file)
    todo = []
    for paper in papers:
        paper_id = paper["paperId"]
        if (args.output_dir / f"{paper_id}.txt").exists():
            continue
        if paper_id in status and not args.retry_failed:
            continue
        todo.append(paper)
    if args.limit > 0:
        todo = todo[: args.limit]
    print(f"{len(papers)} papers listed, {len(todo)} to download")

    session = make_session(os.environ.get("CONTACT_EMAIL"))
    s2_limiter = RateLimiter(args.s2_interval)
    limiters = {"epmc": RateLimiter(args.epmc_interval), "generic": RateLimiter(0.1)}
    log = StatusLog(args.status_file)

    done = 0
    succeeded = 0
    for start in range(0, len(todo), S2_BATCH_SIZE):
        batch = todo[start : start + S2_BATCH_SIZE]
        urls = lookup_open_access_pdfs(session, s2_limiter, api_key, [p["paperId"] for p in batch])
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(process_paper, session, limiters, paper, urls.get(paper["paperId"]), args, log)
                for paper in batch
            ]
            for future in futures:
                succeeded += int(future.result())
        done += len(batch)
        print(f"{done}/{len(todo)} processed, {succeeded} downloaded")


if __name__ == "__main__":
    main()
