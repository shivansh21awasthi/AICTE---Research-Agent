"""
agent.py — Core ResearchAgent orchestrator.

Ties together all modules:
  LiteratureSearcher → Summarizer → ReferenceManager
  → ReportGenerator → HypothesisSuggester
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .config import config, logger as root_logger
from .granite_client import GraniteClient
from .literature_search import LiteratureSearcher, Paper
from .summarizer import Summarizer
from .references import ReferenceManager
from .report_generator import ReportGenerator

logger = logging.getLogger("research_agent.agent")


@dataclass
class ResearchSession:
    """Holds all artefacts produced during one research session."""

    query: str
    papers: list[Paper] = field(default_factory=list)
    summaries: dict[str, str] = field(default_factory=dict)   # title → summary
    synthesis: str = ""
    hypotheses: str = ""
    report_md: str = ""
    report_pdf_path: Optional[Path] = None
    cite_keys: list[str] = field(default_factory=list)


class ResearchAgent:
    """
    High-level Research Agent powered by IBM Granite (watsonx.ai).

    Capabilities
    ------------
    - Literature search (arXiv + Semantic Scholar + IBM Watson Discovery)
    - Paper summarisation per abstract and full text
    - Multi-paper synthesis
    - Hypothesis generation
    - Research paper section drafting
    - Full report generation (Markdown + PDF)
    - Reference management (BibTeX / APA / JSON export)

    Quick start
    -----------
    >>> from research_agent import ResearchAgent
    >>> agent = ResearchAgent()
    >>> session = agent.run("Impact of transformer models on biomedical NLP")
    >>> print(session.synthesis)
    """

    def __init__(self) -> None:
        root_logger.info("Initialising Research Agent with IBM Granite…")
        try:
            config.validate()
        except ValueError as exc:
            root_logger.warning(str(exc))
            root_logger.warning("Agent running in DEMO mode — IBM Granite calls will fail until credentials are set.")

        self._granite = GraniteClient()
        self._searcher = LiteratureSearcher()
        self._summarizer = Summarizer(self._granite)
        self._ref_manager = ReferenceManager()
        self._reporter = ReportGenerator(self._granite)

        # Ensure output directory exists
        config.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Main pipeline ────────────────────────────────────────────────────────

    def run(
        self,
        research_question: str,
        max_results: Optional[int] = None,
        generate_pdf: bool = True,
    ) -> ResearchSession:
        """
        Execute the full research pipeline for a given question.

        Parameters
        ----------
        research_question : str
            The research question or topic (plain English).
        max_results : int, optional
            Override MAX_SEARCH_RESULTS from .env.
        generate_pdf : bool
            Whether to produce a PDF report in addition to Markdown.

        Returns
        -------
        ResearchSession
            All artefacts from the session.
        """
        session = ResearchSession(query=research_question)

        # Step 1: Literature search
        logger.info("Step 1/5 — Searching literature for: %s", research_question)
        session.papers = self.search(research_question, max_results=max_results)

        # Step 2: Summarise each paper
        logger.info("Step 2/5 — Summarising %d papers…", len(session.papers))
        session.summaries = self.summarize_papers(session.papers)

        # Step 3: Add papers to reference manager
        logger.info("Step 3/5 — Registering references…")
        session.cite_keys = self._ref_manager.add_many(session.papers)

        # Step 4: Synthesis & hypothesis generation
        logger.info("Step 4/5 — Synthesising literature & generating hypotheses…")
        session.synthesis = self._summarizer.synthesize(session.papers, research_question)
        session.hypotheses = self._granite.suggest_hypotheses(session.synthesis)

        # Step 5: Generate report
        logger.info("Step 5/5 — Generating research report…")
        safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in research_question)[:60]
        md_path = config.output_dir / f"{safe_name}.md"
        session.report_md = self._reporter.generate_markdown(
            research_question=research_question,
            papers=session.papers,
            synthesis=session.synthesis,
            hypotheses=session.hypotheses,
            ref_manager=self._ref_manager,
            output_path=md_path,
        )

        if generate_pdf:
            pdf_path = config.output_dir / f"{safe_name}.pdf"
            self._reporter.generate_pdf(session.report_md, pdf_path)
            session.report_pdf_path = pdf_path

        # Also export BibTeX and APA references
        self._ref_manager.export_bibtex(config.output_dir / f"{safe_name}.bib")
        self._ref_manager.export_apa(config.output_dir / f"{safe_name}_apa.txt")
        self._ref_manager.export_json(config.output_dir / f"{safe_name}_refs.json")

        logger.info("Research Agent pipeline complete. Outputs in: %s", config.output_dir)
        return session

    # ── Individual capabilities (usable standalone) ──────────────────────────

    def search(self, query: str, max_results: Optional[int] = None) -> list[Paper]:
        """Search academic sources and return Paper objects."""
        return self._searcher.search(query, max_results=max_results)

    def summarize_papers(self, papers: list[Paper]) -> dict[str, str]:
        """Summarise a list of papers. Returns {title: summary} dict."""
        summaries: dict[str, str] = {}
        for paper in papers:
            try:
                summaries[paper.title] = self._summarizer.summarize_abstract(paper)
            except Exception as exc:
                logger.warning("Could not summarise '%s': %s", paper.title, exc)
                summaries[paper.title] = paper.abstract[:300] if paper.abstract else "N/A"
        return summaries

    def suggest_hypotheses(self, context: str, n: int = 3) -> str:
        """Generate n research hypotheses from a context string."""
        return self._granite.suggest_hypotheses(context, n=n)

    def draft_section(self, section_title: str, notes: str) -> str:
        """Draft a research paper section from notes using IBM Granite."""
        return self._granite.draft_section(section_title, notes)

    def export_references(self, fmt: str = "bibtex", path: Optional[str] = None) -> str:
        """
        Export current session references.

        Parameters
        ----------
        fmt : str
            One of 'bibtex', 'apa', or 'json'.
        path : str, optional
            File path to write to.
        """
        if fmt == "bibtex":
            return self._ref_manager.export_bibtex(path)
        elif fmt == "apa":
            return self._ref_manager.export_apa(path)
        elif fmt == "json":
            return self._ref_manager.export_json(path)
        raise ValueError(f"Unknown format '{fmt}'. Choose from: bibtex, apa, json")

    def get_reference_count(self) -> int:
        """Return total number of references registered in this session."""
        return self._ref_manager.count()
