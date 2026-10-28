import csv

import pytest

from src.papers import download_papers as dp


class FakeResponse:
    def __init__(self, status=200, payload=None, content=b"", headers=None):
        self.status_code = status
        self._payload = payload
        self._content = content
        self.headers = headers or {}

    def json(self):
        return self._payload

    def iter_content(self, size):
        yield self._content


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.responses.pop(0)


def no_wait():
    return dp.RateLimiter(0.0)


def test_lookup_returns_only_papers_with_pdf_links():
    session = FakeSession(
        [FakeResponse(payload=[{"openAccessPdf": {"url": "http://x/a.pdf"}}, {"openAccessPdf": None}, None])]
    )
    urls = dp.lookup_open_access_pdfs(session, no_wait(), "key", ["p1", "p2", "p3"])
    assert urls == {"p1": "http://x/a.pdf"}
    assert session.calls[0][2]["headers"] == {"x-api-key": "key"}


def test_find_pmcid_requires_open_access():
    session = FakeSession(
        [
            FakeResponse(payload={"resultList": {"result": [{"pmcid": "PMC1", "isOpenAccess": "N"}]}}),
            FakeResponse(payload={"resultList": {"result": [{"pmcid": "PMC2", "isOpenAccess": "Y"}]}}),
        ]
    )
    assert dp.find_pmcid(session, no_wait(), "123", "10.1/abc") == "PMC2"


def test_fetch_pdf_rejects_non_pdf_content():
    assert dp.fetch_pdf(FakeSession([FakeResponse(content=b"<html>")]), no_wait(), "http://x") is None
    assert (
        dp.fetch_pdf(FakeSession([FakeResponse(content=b"%PDF-1.7 data")]), no_wait(), "http://x") == b"%PDF-1.7 data"
    )


def test_status_log_and_resume(tmp_path):
    log = dp.StatusLog(tmp_path / "status.tsv")
    log.write("p1", "ok", "semantic_scholar")
    log.write("p2", "no_open_access")
    assert dp.load_status(tmp_path / "status.tsv") == {"p1": "ok", "p2": "no_open_access"}


def test_load_papers_reads_tsv(tmp_path):
    path = tmp_path / "papers.tsv"
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["paperId", "doi", "pmid"])
        writer.writerow(["p1", "10.1/a", "5"])
    assert dp.load_papers(path) == [{"paperId": "p1", "doi": "10.1/a", "pmid": "5"}]


@pytest.mark.parametrize("retry_after,expected_calls", [("0", 2)])
def test_retry_on_429(retry_after, expected_calls):
    session = FakeSession([FakeResponse(status=429, headers={"Retry-After": retry_after}), FakeResponse(status=200)])
    response = dp.request_with_retry(session, "GET", "http://x", no_wait())
    assert response.status_code == 200
    assert len(session.calls) == expected_calls
