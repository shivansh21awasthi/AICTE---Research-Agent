"""
main.py — CLI entry point for the Research Agent.

Usage
-----
    python main.py "transformer models in biomedical NLP"
    python main.py "quantum computing error correction" --results 15
    python main.py "climate change ML models" --no-pdf --format apa
    python main.py draft --section "Introduction" --notes "notes.txt"
    python main.py hypotheses --context "context.txt" --n 5
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()


def _print_banner() -> None:
    console.print(
        Panel.fit(
            "[bold blue]Research Agent[/bold blue]\n"
            "[dim]Powered by IBM Granite on IBM watsonx.ai[/dim]",
            border_style="blue",
        )
    )


# ── Main CLI group ────────────────────────────────────────────────────────────

@click.group(invoke_without_command=True)
@click.argument("query", required=False)
@click.option("--results", "-r", default=None, type=int, help="Max papers to retrieve (overrides .env).")
@click.option("--no-pdf", is_flag=True, default=False, help="Skip PDF report generation.")
@click.option("--format", "-f", "ref_format",
              type=click.Choice(["bibtex", "apa", "json"], case_sensitive=False),
              default="bibtex", show_default=True,
              help="Reference export format.")
@click.pass_context
def cli(ctx: click.Context, query: str | None, results: int | None, no_pdf: bool, ref_format: str) -> None:
    """
    Research Agent — AI-powered academic research assistant.

    Pass a QUERY to run the full pipeline, or use a sub-command for focused tasks.

    \b
    Examples:
      python main.py "deep learning for drug discovery"
      python main.py search "CRISPR gene editing"
      python main.py hypotheses --context "context.txt"
      python main.py draft --section "Methods"
    """
    if ctx.invoked_subcommand is not None:
        return  # let sub-command handle

    if not query:
        click.echo(ctx.get_help())
        return

    _print_banner()
    _run_full_pipeline(query, results, no_pdf, ref_format)


def _run_full_pipeline(query: str, results: int | None, no_pdf: bool, ref_format: str) -> None:
    from research_agent import ResearchAgent

    console.print(f"\n[bold]Research Question:[/bold] {query}\n")

    with console.status("[bold green]Initialising IBM Granite client…[/bold green]"):
        agent = ResearchAgent()

    with console.status("[bold green]Running full research pipeline…[/bold green]"):
        session = agent.run(
            research_question=query,
            max_results=results,
            generate_pdf=not no_pdf,
        )

    # ── Results table ─────────────────────────────────────────────────────────
    table = Table(
        title=f"Papers Found ({len(session.papers)})",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold cyan",
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Title", min_width=30)
    table.add_column("Authors", max_width=25)
    table.add_column("Year", width=6)
    table.add_column("Citations", width=9)
    table.add_column("Source", width=14)

    for i, paper in enumerate(session.papers, 1):
        authors = ", ".join(paper.authors[:2])
        if len(paper.authors) > 2:
            authors += " et al."
        table.add_row(
            str(i),
            textwrap.shorten(paper.title, width=60, placeholder="…"),
            authors or "N/A",
            str(paper.year or "N/A"),
            str(paper.citation_count),
            paper.source,
        )

    console.print(table)

    # ── Synthesis ─────────────────────────────────────────────────────────────
    console.print("\n[bold cyan]Literature Synthesis[/bold cyan]")
    console.print(Panel(session.synthesis, border_style="cyan"))

    # ── Hypotheses ────────────────────────────────────────────────────────────
    console.print("\n[bold magenta]Proposed Hypotheses[/bold magenta]")
    console.print(Panel(session.hypotheses, border_style="magenta"))

    # ── Output summary ────────────────────────────────────────────────────────
    from research_agent.config import config
    console.print("\n[bold green]Output files written to:[/bold green]", str(config.output_dir))
    console.print(f"  - Markdown report")
    if not no_pdf:
        console.print(f"  - PDF report")
    console.print(f"  - BibTeX / APA / JSON references")
    console.print(f"\n[dim]References: {agent.get_reference_count()} papers indexed[/dim]\n")


# ── Sub-commands ─────────────────────────────────────────────────────────────

@cli.command("search")
@click.argument("query")
@click.option("--results", "-r", default=None, type=int, help="Max papers.")
def cmd_search(query: str, results: int | None) -> None:
    """Search academic sources for QUERY and display results."""
    _print_banner()
    from research_agent import ResearchAgent

    with console.status("[bold green]Searching literature…[/bold green]"):
        agent = ResearchAgent()
        papers = agent.search(query, max_results=results)

    if not papers:
        console.print("[yellow]No papers found.[/yellow]")
        return

    table = Table(title=f"Search Results for: {query}", box=box.SIMPLE_HEAVY, header_style="bold cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("Title", min_width=40)
    table.add_column("Year", width=6)
    table.add_column("URL")

    for i, paper in enumerate(papers, 1):
        table.add_row(str(i), textwrap.shorten(paper.title, 60, placeholder="…"),
                      str(paper.year or "N/A"), paper.url)

    console.print(table)


@cli.command("summarize")
@click.argument("query")
@click.option("--results", "-r", default=5, show_default=True, type=int)
def cmd_summarize(query: str, results: int) -> None:
    """Search and summarise papers on a topic using IBM Granite."""
    _print_banner()
    from research_agent import ResearchAgent

    agent = ResearchAgent()
    with console.status("[bold green]Searching…[/bold green]"):
        papers = agent.search(query, max_results=results)

    with console.status("[bold green]Summarising with IBM Granite…[/bold green]"):
        summaries = agent.summarize_papers(papers)

    for title, summary in summaries.items():
        console.print(f"\n[bold]{title}[/bold]")
        console.print(Panel(summary, border_style="green"))


@cli.command("hypotheses")
@click.option("--context", "-c", default=None, help="Context text or path to a .txt file.")
@click.option("--query", "-q", default=None, help="Run a search first and use synthesis as context.")
@click.option("--n", default=3, show_default=True, help="Number of hypotheses to generate.")
def cmd_hypotheses(context: str | None, query: str | None, n: int) -> None:
    """Generate research hypotheses using IBM Granite."""
    _print_banner()
    from research_agent import ResearchAgent

    agent = ResearchAgent()

    if context:
        ctx_path = Path(context)
        if ctx_path.exists():
            context_text = ctx_path.read_text(encoding="utf-8")
        else:
            context_text = context
    elif query:
        with console.status("[bold green]Running search + synthesis…[/bold green]"):
            papers = agent.search(query)
            from research_agent.summarizer import Summarizer
            from research_agent.granite_client import GraniteClient
            syn = agent._granite if hasattr(agent, "_granite") else GraniteClient()
            from research_agent.summarizer import Summarizer as Sum
            summarizer = Sum(syn)
            context_text = summarizer.synthesize(papers, query)
    else:
        console.print("[red]Provide --context TEXT or --query TOPIC.[/red]")
        sys.exit(1)

    with console.status("[bold green]Generating hypotheses with IBM Granite…[/bold green]"):
        result = agent.suggest_hypotheses(context_text, n=n)

    console.print("\n[bold magenta]Proposed Hypotheses[/bold magenta]")
    console.print(Panel(result, border_style="magenta"))


@cli.command("draft")
@click.option("--section", "-s", required=True, help="Section title (e.g., 'Introduction').")
@click.option("--notes", "-n", default=None, help="Notes text or path to a .txt file.")
def cmd_draft(section: str, notes: str | None) -> None:
    """Draft a research paper section using IBM Granite."""
    _print_banner()
    from research_agent import ResearchAgent

    agent = ResearchAgent()

    if notes:
        notes_path = Path(notes)
        notes_text = notes_path.read_text(encoding="utf-8") if notes_path.exists() else notes
    else:
        notes_text = click.prompt(f"Enter notes for the '{section}' section")

    with console.status(f"[bold green]Drafting '{section}' section with IBM Granite…[/bold green]"):
        result = agent.draft_section(section, notes_text)

    console.print(f"\n[bold blue]Section: {section}[/bold blue]")
    console.print(Panel(result, border_style="blue"))


if __name__ == "__main__":
    cli()
