"""
summarizer.py — Paper summarisation using IBM Granite.

Provides per-paper summarisation and multi-paper synthesis.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .granite_client import GraniteClient

if TYPE_CHECKING:
    from .literature_search import Paper

logger = logging.getLogger("research_agent.summarizer")


class Summarizer:
    """
    Uses IBM Granite to produce summaries at different granularities:
      - Single paper abstract → concise summary
      - Full paper text       → structured summary (background / methods / results)
      - Collection of papers  → comparative synthesis
    """

    def __init__(self, client: GraniteClient) -> None:
        self._client = client

    # ── Single-paper summary ─────────────────────────────────────────────────

    def summarize_abstract(self, paper: "Paper") -> str:
        """Return a 2–3 sentence summary of a paper abstract."""
        if not paper.abstract:
            return "No abstract available."
        return self._client.summarize(paper.abstract, max_sentences=3)

    def summarize_full_text(self, text: str, paper_title: str = "") -> str:
        """
        Produce a structured summary (Background / Methods / Results / Conclusion)
        for a full paper text chunk.
        """
        heading = f'Paper: "{paper_title}"\n\n' if paper_title else ""
        prompt = (
            f"You are an expert academic reviewer. "
            f"Read the following research paper excerpt and produce a structured summary "
            f"with four clearly labelled sections: "
            f"Background, Methods, Results, and Conclusion. "
            f"Be concise and precise.\n\n"
            f"{heading}"
            f"Text:\n{text[:3000]}\n\n"
            f"Structured Summary:"
        )
        return self._client.generate(prompt)

    # ── Multi-paper synthesis ────────────────────────────────────────────────

    def synthesize(self, papers: "list[Paper]", research_question: str) -> str:
        """
        Generate a literature synthesis paragraph that compares and contrasts
        a set of papers relative to a research question.

        Parameters
        ----------
        papers : list[Paper]
            Papers to synthesise.
        research_question : str
            The overarching research question guiding synthesis.

        Returns
        -------
        str
            A coherent multi-paper synthesis from IBM Granite.
        """
        if not papers:
            return "No papers available for synthesis."

        # Build a compact context block: title + abstract snippet
        context_lines = []
        for i, p in enumerate(papers[:8], 1):  # cap at 8 to stay within token budget
            snippet = (p.abstract or "")[:400]
            context_lines.append(f"[{i}] {p.title} ({p.year or 'n.d.'}) — {snippet}")
        context = "\n\n".join(context_lines)

        prompt = (
            f"You are an expert researcher writing a literature review. "
            f"Research question: {research_question}\n\n"
            f"Based on the following papers, write a coherent synthesis paragraph "
            f"that identifies key themes, agreements, and gaps. "
            f"Cite papers by their number in square brackets.\n\n"
            f"Papers:\n{context}\n\n"
            f"Synthesis:"
        )
        return self._client.generate(prompt)

    # ── Key-findings extraction ──────────────────────────────────────────────

    def extract_key_findings(self, paper: "Paper") -> str:
        """
        Ask Granite to extract bullet-point key findings from a paper abstract.
        """
        if not paper.abstract:
            return "• No abstract available."
        prompt = (
            f"Extract the 3–5 most important findings from this research paper abstract as bullet points.\n\n"
            f"Title: {paper.title}\n"
            f"Abstract: {paper.abstract}\n\n"
            f"Key findings:"
        )
        return self._client.generate(prompt)
