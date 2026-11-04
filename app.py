"""
app.py — Gradio web interface for the Research Agent.

Run:
    python app.py

Then open:  http://localhost:7860
Public URL: printed in terminal (via share=True)
"""

from __future__ import annotations

import os
import textwrap

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import gradio as gr
from research_agent import ResearchAgent
from research_agent.config import config


# ── Helpers ───────────────────────────────────────────────────────────────────

def _papers_to_markdown(papers) -> str:
    if not papers:
        return "_No papers found._"
    lines = ["| # | Title | Authors | Year | Source |",
             "|---|-------|---------|------|--------|"]
    for i, p in enumerate(papers, 1):
        authors = ", ".join(p.authors[:2]) + (" et al." if len(p.authors) > 2 else "")
        title_link = f"[{textwrap.shorten(p.title, 60, placeholder='…')}]({p.url})" if p.url else textwrap.shorten(p.title, 60, placeholder="…")
        lines.append(f"| {i} | {title_link} | {authors or 'N/A'} | {p.year or 'N/A'} | {p.source} |")
    return "\n".join(lines)


def _summaries_to_markdown(summaries: dict) -> str:
    if not summaries:
        return "_No summaries available._"
    parts = []
    for title, summary in summaries.items():
        parts.append(f"**{title}**\n\n{summary}\n")
    return "\n---\n".join(parts)


# ── Tab 1: Full Pipeline ──────────────────────────────────────────────────────

def run_full_pipeline(topic: str, max_results: int, skip_pdf: bool, progress=gr.Progress()):
    if not topic.strip():
        return "Please enter a research topic.", "", "", "", None

    try:
        progress(0.05, desc="Initialising IBM Granite…")
        agent = ResearchAgent()

        progress(0.15, desc="Searching literature…")
        session = agent.run(
            research_question=topic.strip(),
            max_results=int(max_results),
            generate_pdf=not skip_pdf,
        )

        progress(0.95, desc="Formatting results…")

        papers_md   = _papers_to_markdown(session.papers)
        summary_md  = _summaries_to_markdown(session.summaries)
        synthesis   = session.synthesis or "_No synthesis generated._"
        hypotheses  = session.hypotheses or "_No hypotheses generated._"

        # Return the markdown report file for download
        safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in topic.strip())[:60]
        md_path = config.output_dir / f"{safe_name}.md"
        report_file = str(md_path) if md_path.exists() else None

        progress(1.0, desc="Done!")
        return papers_md, summary_md, synthesis, hypotheses, report_file

    except Exception as exc:
        err = f"**Error:** {exc}"
        return err, "", "", "", None


# ── Tab 2: Search Only ────────────────────────────────────────────────────────

def run_search(topic: str, max_results: int, progress=gr.Progress()):
    if not topic.strip():
        return "Please enter a search topic."
    try:
        progress(0.1, desc="Connecting…")
        agent = ResearchAgent()
        progress(0.4, desc="Searching arXiv & Semantic Scholar…")
        papers = agent.search(topic.strip(), max_results=int(max_results))
        progress(1.0, desc="Done!")
        return _papers_to_markdown(papers)
    except Exception as exc:
        return f"**Error:** {exc}"


# ── Tab 3: Hypotheses ─────────────────────────────────────────────────────────

def run_hypotheses(context_or_topic: str, mode: str, n: int, progress=gr.Progress()):
    if not context_or_topic.strip():
        return "Please enter a topic or context text."
    try:
        progress(0.1, desc="Initialising…")
        agent = ResearchAgent()

        if mode == "Search topic first":
            progress(0.3, desc="Searching & synthesising…")
            papers = agent.search(context_or_topic.strip())
            from research_agent.summarizer import Summarizer
            summarizer = Summarizer(agent._granite)
            context_text = summarizer.synthesize(papers, context_or_topic.strip())
        else:
            context_text = context_or_topic.strip()

        progress(0.7, desc="Generating hypotheses with IBM Granite…")
        result = agent.suggest_hypotheses(context_text, n=int(n))
        progress(1.0, desc="Done!")
        return result
    except Exception as exc:
        return f"**Error:** {exc}"


# ── Tab 4: Draft Section ──────────────────────────────────────────────────────

def run_draft(section: str, notes: str, progress=gr.Progress()):
    if not section.strip():
        return "Please enter a section title."
    if not notes.strip():
        return "Please enter some notes for the section."
    try:
        progress(0.2, desc="Initialising…")
        agent = ResearchAgent()
        progress(0.5, desc=f"Drafting '{section}' with IBM Granite…")
        result = agent.draft_section(section.strip(), notes.strip())
        progress(1.0, desc="Done!")
        return result
    except Exception as exc:
        return f"**Error:** {exc}"


# ── Build UI ──────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(title="Research Agent — IBM Granite") as demo:

        gr.Markdown(
            """
            # Research Agent
            ### Powered by IBM Granite on IBM watsonx.ai
            *Search academic literature · Summarise papers · Generate hypotheses · Draft sections*
            ---
            """
        )

        # ── Tab 1: Full Pipeline ──────────────────────────────────────────────
        with gr.Tab("Full Pipeline"):
            gr.Markdown("Run the complete research workflow: search → summarise → synthesise → hypotheses → report.")
            with gr.Row():
                with gr.Column(scale=3):
                    fp_topic = gr.Textbox(label="Research Topic", placeholder="e.g. transformer models in biomedical NLP", lines=2)
                with gr.Column(scale=1):
                    fp_max   = gr.Slider(minimum=3, maximum=20, value=10, step=1, label="Max Papers")
                    fp_nopdf = gr.Checkbox(label="Skip PDF generation", value=True)
                    fp_btn   = gr.Button("Run Full Pipeline", variant="primary")

            with gr.Tabs():
                with gr.Tab("Papers Found"):
                    fp_papers = gr.Markdown()
                with gr.Tab("Paper Summaries"):
                    fp_summaries = gr.Markdown()
                with gr.Tab("Literature Synthesis"):
                    fp_synthesis = gr.Markdown()
                with gr.Tab("Hypotheses"):
                    fp_hypotheses = gr.Markdown()
                with gr.Tab("Download Report"):
                    fp_report = gr.File(label="Markdown Report")

            fp_btn.click(
                fn=run_full_pipeline,
                inputs=[fp_topic, fp_max, fp_nopdf],
                outputs=[fp_papers, fp_summaries, fp_synthesis, fp_hypotheses, fp_report],
            )

        # ── Tab 2: Search ─────────────────────────────────────────────────────
        with gr.Tab("Search Papers"):
            gr.Markdown("Search arXiv and Semantic Scholar without running the full LLM pipeline.")
            with gr.Row():
                with gr.Column(scale=3):
                    s_topic = gr.Textbox(label="Search Query", placeholder="e.g. CRISPR gene editing", lines=1)
                with gr.Column(scale=1):
                    s_max = gr.Slider(minimum=3, maximum=20, value=10, step=1, label="Max Results")
                    s_btn = gr.Button("Search", variant="primary")
            s_out = gr.Markdown()
            s_btn.click(fn=run_search, inputs=[s_topic, s_max], outputs=s_out)

        # ── Tab 3: Hypotheses ─────────────────────────────────────────────────
        with gr.Tab("Generate Hypotheses"):
            gr.Markdown("Generate research hypotheses from a topic (searches first) or from your own context text.")
            with gr.Row():
                with gr.Column(scale=3):
                    h_input = gr.Textbox(
                        label="Topic or Context Text",
                        placeholder="Enter a research topic OR paste your own context paragraph…",
                        lines=4,
                    )
                with gr.Column(scale=1):
                    h_mode = gr.Radio(
                        choices=["Search topic first", "Use text as context directly"],
                        value="Search topic first",
                        label="Input Mode",
                    )
                    h_n   = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Number of Hypotheses")
                    h_btn = gr.Button("Generate Hypotheses", variant="primary")
            h_out = gr.Textbox(label="Proposed Hypotheses", lines=12)
            h_btn.click(fn=run_hypotheses, inputs=[h_input, h_mode, h_n], outputs=h_out)

        # ── Tab 4: Draft Section ──────────────────────────────────────────────
        with gr.Tab("Draft Paper Section"):
            gr.Markdown("Draft a section of a research paper from your notes using IBM Granite.")
            with gr.Row():
                with gr.Column():
                    d_section = gr.Textbox(label="Section Title", placeholder="e.g. Introduction, Methods, Discussion", lines=1)
                    d_notes   = gr.Textbox(label="Your Notes", placeholder="Paste bullet points or rough notes here…", lines=6)
                    d_btn     = gr.Button("Draft Section", variant="primary")
            d_out = gr.Textbox(label="Drafted Section", lines=14)
            d_btn.click(fn=run_draft, inputs=[d_section, d_notes], outputs=d_out)

        gr.Markdown(
            """
            ---
            *AICTE Research Agent Challenge — IBM Cloud Lite + IBM Granite `ibm/granite-4-h-small`*
            """
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,           # generates a public gradio.live link
        inbrowser=True,       # auto-opens your browser
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    )
