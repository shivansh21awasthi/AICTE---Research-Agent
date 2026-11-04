"""
literature_search.py — Literature retrieval module.

Sources used (all free / no API key required for basic access):
  1. arXiv      — preprints in CS, physics, maths, biology, …
  2. Semantic Scholar — broad academic search (REST API)

Optionally uses IBM Watson Discovery when credentials are provided.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import arxiv
import requests

from .config import config

logger = logging.getLogger("research_agent.search")

SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
SEMANTIC_SCHOLAR_FIELDS = "title,authors,year,abstract,externalIds,url,citationCount"


@dataclass
class Paper:
    """Represents a retrieved research paper."""

    title: str
    authors: list[str]
    year: Optional[int]
    abstract: str
    url: str
    source: str                          # "arxiv" | "semantic_scholar" | "discovery"
    citation_count: int = 0
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    pdf_url: Optional[str] = None
    tags: list[str] = field(default_factory=list)

    def short_citation(self) -> str:
        """APA-style short citation string."""
        first_author = self.authors[0] if self.authors else "Unknown"
        year = self.year or "n.d."
        return f"{first_author} et al. ({year}). {self.title}."

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "abstract": self.abstract,
            "url": self.url,
            "source": self.source,
            "citation_count": self.citation_count,
            "arxiv_id": self.arxiv_id,
            "doi": self.doi,
            "pdf_url": self.pdf_url,
            "tags": self.tags,
        }


class LiteratureSearcher:
    """
    Searches multiple academic sources and returns a deduplicated list of Paper objects.
    """

    def __init__(self) -> None:
        self._seen_titles: set[str] = set()

    # ── Public API ───────────────────────────────────────────────────────────

    def search(self, query: str, max_results: Optional[int] = None) -> list[Paper]:
        """
        Search all configured sources for the given query.

        Parameters
        ----------
        query : str
            Natural language or keyword research query.
        max_results : int, optional
            Cap on total papers returned (default: config.max_search_results).

        Returns
        -------
        list[Paper]
            Deduplicated list of Paper objects, sorted by citation count desc.
        """
        limit = max_results or config.max_search_results
        self._seen_titles = set()
        papers: list[Paper] = []

        # 1. arXiv
        try:
            papers.extend(self._search_arxiv(query, limit))
            logger.info("arXiv returned %d papers", len(papers))
        except Exception as exc:
            logger.warning("arXiv search failed: %s", exc)

        # 2. Semantic Scholar (fills remaining quota)
        remaining = limit - len(papers)
        if remaining > 0:
            try:
                ss_papers = self._search_semantic_scholar(query, remaining)
                papers.extend(ss_papers)
                logger.info("Semantic Scholar added %d papers", len(ss_papers))
            except Exception as exc:
                logger.warning("Semantic Scholar search failed: %s", exc)

        # 3. IBM Watson Discovery (if configured)
        if config.discovery_api_key and config.discovery_project_id:
            remaining = limit - len(papers)
            if remaining > 0:
                try:
                    disc_papers = self._search_discovery(query, remaining)
                    papers.extend(disc_papers)
                    logger.info("Watson Discovery added %d papers", len(disc_papers))
                except Exception as exc:
                    logger.warning("Watson Discovery search failed: %s", exc)

        # Sort by citation count (best evidence first)
        papers.sort(key=lambda p: p.citation_count, reverse=True)
        return papers[:limit]

    # ── arXiv ────────────────────────────────────────────────────────────────

    def _search_arxiv(self, query: str, limit: int) -> list[Paper]:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=limit,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        papers = []
        for result in client.results(search):
            title_key = result.title.lower().strip()
            if title_key in self._seen_titles:
                continue
            self._seen_titles.add(title_key)
            papers.append(
                Paper(
                    title=result.title,
                    authors=[str(a) for a in result.authors],
                    year=result.published.year if result.published else None,
                    abstract=result.summary or "",
                    url=result.entry_id,
                    source="arxiv",
                    arxiv_id=result.get_short_id(),
                    pdf_url=result.pdf_url,
                    tags=result.categories,
                )
            )
        return papers

    # ── Semantic Scholar ─────────────────────────────────────────────────────

    def _search_semantic_scholar(self, query: str, limit: int) -> list[Paper]:
        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": SEMANTIC_SCHOLAR_FIELDS,
        }
        time.sleep(1)  # Respect rate limit (100 req/5 min unauthenticated)
        resp = requests.get(SEMANTIC_SCHOLAR_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        papers = []
        for item in data.get("data", []):
            title = item.get("title", "")
            title_key = title.lower().strip()
            if title_key in self._seen_titles:
                continue
            self._seen_titles.add(title_key)

            ext_ids = item.get("externalIds") or {}
            papers.append(
                Paper(
                    title=title,
                    authors=[a.get("name", "") for a in item.get("authors", [])],
                    year=item.get("year"),
                    abstract=item.get("abstract") or "",
                    url=item.get("url") or "",
                    source="semantic_scholar",
                    doi=ext_ids.get("DOI"),
                    arxiv_id=ext_ids.get("ArXiv"),
                    citation_count=item.get("citationCount") or 0,
                )
            )
        return papers

    # ── IBM Watson Discovery ─────────────────────────────────────────────────

    def _search_discovery(self, query: str, limit: int) -> list[Paper]:
        """
        Query IBM Watson Discovery (Natural Language Query).
        Requires DISCOVERY_API_KEY, DISCOVERY_URL, DISCOVERY_PROJECT_ID in .env
        """
        from ibm_watson import DiscoveryV2
        from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

        authenticator = IAMAuthenticator(config.discovery_api_key)
        discovery = DiscoveryV2(version="2020-08-30", authenticator=authenticator)
        discovery.set_service_url(config.discovery_url)

        response = discovery.query(
            project_id=config.discovery_project_id,
            natural_language_query=query,
            count=limit,
        ).get_result()

        papers = []
        for result in response.get("results", []):
            title = result.get("title") or result.get("extracted_metadata", {}).get("title", "Untitled")
            title_key = title.lower().strip()
            if title_key in self._seen_titles:
                continue
            self._seen_titles.add(title_key)

            papers.append(
                Paper(
                    title=title,
                    authors=[],
                    year=None,
                    abstract=result.get("text", [""])[0] if isinstance(result.get("text"), list) else result.get("text", ""),
                    url=result.get("url", ""),
                    source="discovery",
                )
            )
        return papers
