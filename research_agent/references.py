"""
references.py — Citation and reference management.

Features:
  - Add, deduplicate, and organise Paper objects
  - Export to BibTeX, APA plain-text, and JSON
  - Simple in-memory citation store
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .literature_search import Paper

logger = logging.getLogger("research_agent.references")


class ReferenceManager:
    """
    Stores and exports references for a research session.

    Usage
    -----
    rm = ReferenceManager()
    rm.add(paper)
    rm.export_bibtex("refs.bib")
    rm.export_apa("refs.txt")
    rm.export_json("refs.json")
    """

    def __init__(self) -> None:
        self._refs: dict[str, "Paper"] = {}   # key = cite_key

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _make_cite_key(paper: "Paper") -> str:
        """Generate a BibTeX-safe cite key: FirstAuthorYEARkeyword."""
        first_author = (paper.authors[0] if paper.authors else "Unknown").split()[-1]
        year = str(paper.year or "nd")
        keyword = re.sub(r"[^a-z]", "", paper.title.split()[0].lower()) if paper.title else "paper"
        raw = f"{first_author}{year}{keyword}"
        return re.sub(r"[^A-Za-z0-9]", "", raw)

    def _unique_key(self, base_key: str) -> str:
        key = base_key
        i = 1
        while key in self._refs:
            key = f"{base_key}{chr(96 + i)}"   # appends a, b, c, …
            i += 1
        return key

    # ── Public API ───────────────────────────────────────────────────────────

    def add(self, paper: "Paper") -> str:
        """
        Add a paper to the reference store.

        Returns
        -------
        str
            The assigned cite key.
        """
        base_key = self._make_cite_key(paper)
        cite_key = self._unique_key(base_key)
        self._refs[cite_key] = paper
        logger.debug("Added reference [%s]: %s", cite_key, paper.title)
        return cite_key

    def add_many(self, papers: "list[Paper]") -> list[str]:
        return [self.add(p) for p in papers]

    def all_papers(self) -> list[tuple[str, "Paper"]]:
        """Return (cite_key, Paper) pairs."""
        return list(self._refs.items())

    def count(self) -> int:
        return len(self._refs)

    # ── Export: BibTeX ────────────────────────────────────────────────────────

    def export_bibtex(self, path: str | Path | None = None) -> str:
        """
        Render all references as BibTeX entries.

        Parameters
        ----------
        path : str | Path | None
            If given, write output to file; otherwise just return the string.
        """
        lines = []
        for key, paper in self._refs.items():
            authors = " and ".join(paper.authors) if paper.authors else "Unknown"
            entry = [
                f"@article{{{key},",
                f"  author    = {{{authors}}},",
                f"  title     = {{{paper.title}}},",
                f"  year      = {{{paper.year or 'n.d.'}}},",
                f"  url       = {{{paper.url}}},"
            ]
            if paper.doi:
                entry.append(f"  doi       = {{{paper.doi}}},")
            if paper.arxiv_id:
                entry.append(f"  eprint    = {{{paper.arxiv_id}}},")
            entry.append("}")
            lines.append("\n".join(entry))

        bibtex = "\n\n".join(lines)
        if path:
            Path(path).write_text(bibtex, encoding="utf-8")
            logger.info("BibTeX written to %s (%d entries)", path, len(self._refs))
        return bibtex

    # ── Export: APA plain text ────────────────────────────────────────────────

    def export_apa(self, path: str | Path | None = None) -> str:
        """Render all references as APA-style plain text."""
        entries = []
        for paper in self._refs.values():
            authors = ", ".join(paper.authors[:3])
            if len(paper.authors) > 3:
                authors += " et al."
            year = f"({paper.year})." if paper.year else "(n.d.)."
            doi_part = f" https://doi.org/{paper.doi}" if paper.doi else (f" {paper.url}" if paper.url else "")
            entries.append(f"{authors} {year} {paper.title}.{doi_part}")

        apa = "\n\n".join(entries)
        if path:
            Path(path).write_text(apa, encoding="utf-8")
            logger.info("APA references written to %s (%d entries)", path, len(self._refs))
        return apa

    # ── Export: JSON ──────────────────────────────────────────────────────────

    def export_json(self, path: str | Path | None = None) -> str:
        """Export all references as a JSON list."""
        data = [
            {"cite_key": k, **p.to_dict()}
            for k, p in self._refs.items()
        ]
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        if path:
            Path(path).write_text(json_str, encoding="utf-8")
            logger.info("JSON references written to %s (%d entries)", path, len(self._refs))
        return json_str
