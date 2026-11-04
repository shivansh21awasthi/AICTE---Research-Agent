"""
app_streamlit.py — Streamlit web interface for the Research Agent.

Local run:
    streamlit run app_streamlit.py

Deploy:
    Push to GitHub → connect at https://share.streamlit.io
    Set secrets in the Streamlit Cloud dashboard (IBM_CLOUD_API_KEY, WATSONX_PROJECT_ID)
"""

from __future__ import annotations

import os
import textwrap

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Research Agent — IBM Granite",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject Streamlit Cloud secrets into env vars ──────────────────────────────
# When deployed, credentials come from st.secrets instead of .env
# Falls back silently to .env if no secrets.toml is present
try:
    for key in ("IBM_CLOUD_API_KEY", "WATSONX_PROJECT_ID", "WATSONX_URL",
                "GRANITE_MODEL_ID", "GRANITE_EMBEDDING_MODEL_ID"):
        if key in st.secrets:
            os.environ[key] = st.secrets[key]
except Exception:
    pass  # No secrets.toml — will use .env via python-dotenv


# ── Agent import (lazy so missing credentials show a nice error) ──────────────
@st.cache_resource(show_spinner="Initialising IBM Granite client…")
def get_agent():
    from research_agent import ResearchAgent
    return ResearchAgent()


# ── Helpers ───────────────────────────────────────────────────────────────────

def papers_to_df(papers):
    import pandas as pd
    rows = []
    for p in papers:
        authors = ", ".join(p.authors[:2]) + (" et al." if len(p.authors) > 2 else "")
        rows.append({
            "Title": p.title,
            "Authors": authors or "N/A",
            "Year": p.year or "N/A",
            "Citations": p.citation_count,
            "Source": p.source,
            "URL": p.url,
        })
    return pd.DataFrame(rows)


def summaries_to_text(summaries: dict) -> str:
    parts = []
    for title, summary in summaries.items():
        parts.append(f"**{title}**\n\n{summary}")
    return "\n\n---\n\n".join(parts)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg", width=80)
    st.title("Research Agent")
    st.caption("Powered by IBM Granite on IBM watsonx.ai")
    st.divider()

    st.markdown("**Model**")
    st.code(os.environ.get("GRANITE_MODEL_ID", "ibm/granite-4-h-small"))

    st.markdown("**Capabilities**")
    st.markdown("""
- 🔍 Literature search (arXiv)
- 📄 Paper summarisation
- 🧬 Literature synthesis
- 💡 Hypothesis generation
- ✍️ Section drafting
- 📁 BibTeX / APA / JSON export
""")
    st.divider()
    st.caption("AICTE Research Agent Challenge")
    st.caption("IBM Cloud Lite + IBM Granite")


# ── Header ────────────────────────────────────────────────────────────────────

st.title("🔬 Research Agent")
st.markdown("*AI-powered academic research assistant — search, summarise, synthesise, and hypothesise*")
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Full Pipeline",
    "🔍 Search Papers",
    "💡 Generate Hypotheses",
    "✍️ Draft Section",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Full Pipeline
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.subheader("Full Research Pipeline")
    st.caption("Search → Summarise → Synthesise → Hypotheses → Report")

    col1, col2 = st.columns([3, 1])
    with col1:
        fp_topic = st.text_input(
            "Research Topic",
            placeholder="e.g. transformer models in biomedical NLP",
            key="fp_topic",
        )
    with col2:
        fp_max = st.slider("Max Papers", 3, 20, 10, key="fp_max")
        fp_nopdf = st.checkbox("Skip PDF", value=True, key="fp_nopdf")

    fp_run = st.button("Run Full Pipeline", type="primary", key="fp_run", use_container_width=True)

    if fp_run:
        if not fp_topic.strip():
            st.error("Please enter a research topic.")
        else:
            try:
                agent = get_agent()

                with st.status("Running research pipeline…", expanded=True) as status:
                    st.write("Searching literature…")
                    from research_agent.config import config
                    session = agent.run(
                        research_question=fp_topic.strip(),
                        max_results=fp_max,
                        generate_pdf=not fp_nopdf,
                    )
                    status.update(label="Pipeline complete!", state="complete")

                # Results
                r1, r2 = st.tabs(["Papers Found", "Paper Summaries"])

                with r1:
                    if session.papers:
                        df = papers_to_df(session.papers)
                        st.dataframe(
                            df,
                            use_container_width=True,
                            column_config={"URL": st.column_config.LinkColumn("URL")},
                            hide_index=True,
                        )
                    else:
                        st.info("No papers found.")

                with r2:
                    if session.summaries:
                        for title, summary in session.summaries.items():
                            with st.expander(textwrap.shorten(title, 80, placeholder="…")):
                                st.write(summary)
                    else:
                        st.info("No summaries available.")

                st.subheader("Literature Synthesis")
                st.info(session.synthesis or "No synthesis generated.")

                st.subheader("Proposed Hypotheses")
                st.success(session.hypotheses or "No hypotheses generated.")

                # Download report
                safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in fp_topic.strip())[:60]
                md_path = config.output_dir / f"{safe_name}.md"
                if md_path.exists():
                    st.subheader("Download Report")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.download_button(
                            label="Download Markdown Report",
                            data=md_path.read_text(encoding="utf-8"),
                            file_name=f"{safe_name}.md",
                            mime="text/markdown",
                        )
                    bib_path = config.output_dir / f"{safe_name}.bib"
                    with col_b:
                        if bib_path.exists():
                            st.download_button(
                                label="Download BibTeX",
                                data=bib_path.read_text(encoding="utf-8"),
                                file_name=f"{safe_name}.bib",
                                mime="text/plain",
                            )

            except Exception as exc:
                st.error(f"Error: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Search Papers
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("Search Academic Papers")
    st.caption("Searches arXiv and Semantic Scholar — no LLM calls, very fast.")

    col1, col2 = st.columns([3, 1])
    with col1:
        s_topic = st.text_input("Search Query", placeholder="e.g. CRISPR gene editing", key="s_topic")
    with col2:
        s_max = st.slider("Max Results", 3, 20, 10, key="s_max")

    s_run = st.button("Search", type="primary", key="s_run", use_container_width=True)

    if s_run:
        if not s_topic.strip():
            st.error("Please enter a search query.")
        else:
            try:
                with st.spinner("Searching literature…"):
                    agent = get_agent()
                    papers = agent.search(s_topic.strip(), max_results=s_max)

                if papers:
                    st.success(f"Found {len(papers)} papers")
                    df = papers_to_df(papers)
                    st.dataframe(
                        df,
                        use_container_width=True,
                        column_config={"URL": st.column_config.LinkColumn("URL")},
                        hide_index=True,
                    )
                else:
                    st.warning("No papers found.")
            except Exception as exc:
                st.error(f"Error: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Generate Hypotheses
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("Generate Research Hypotheses")
    st.caption("IBM Granite proposes novel, testable hypotheses from your topic or context.")

    h_mode = st.radio(
        "Input Mode",
        ["Search a topic first (recommended)", "Paste your own context text"],
        horizontal=True,
        key="h_mode",
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        h_input = st.text_area(
            "Topic" if "Search" in h_mode else "Context Text",
            placeholder="Enter a research topic…" if "Search" in h_mode else "Paste your context paragraph here…",
            height=120,
            key="h_input",
        )
    with col2:
        h_n = st.slider("Number of Hypotheses", 1, 10, 3, key="h_n")

    h_run = st.button("Generate Hypotheses", type="primary", key="h_run", use_container_width=True)

    if h_run:
        if not h_input.strip():
            st.error("Please enter a topic or context text.")
        else:
            try:
                agent = get_agent()
                if "Search" in h_mode:
                    with st.spinner("Searching & synthesising…"):
                        papers = agent.search(h_input.strip())
                        from research_agent.summarizer import Summarizer
                        summarizer = Summarizer(agent._granite)
                        context_text = summarizer.synthesize(papers, h_input.strip())
                    st.caption(f"Synthesised from {len(papers)} papers")
                else:
                    context_text = h_input.strip()

                with st.spinner("Generating hypotheses with IBM Granite…"):
                    result = agent.suggest_hypotheses(context_text, n=h_n)

                st.subheader("Proposed Hypotheses")
                st.success(result)

            except Exception as exc:
                st.error(f"Error: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Draft Section
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.subheader("Draft a Research Paper Section")
    st.caption("IBM Granite writes a formal academic section from your rough notes.")

    d_section = st.text_input(
        "Section Title",
        placeholder="e.g. Introduction, Methods, Discussion, Conclusion",
        key="d_section",
    )
    d_notes = st.text_area(
        "Your Notes",
        placeholder="Paste bullet points or rough notes here…",
        height=150,
        key="d_notes",
    )

    d_run = st.button("Draft Section", type="primary", key="d_run", use_container_width=True)

    if d_run:
        if not d_section.strip():
            st.error("Please enter a section title.")
        elif not d_notes.strip():
            st.error("Please enter some notes.")
        else:
            try:
                with st.spinner(f"Drafting '{d_section}' with IBM Granite…"):
                    agent = get_agent()
                    result = agent.draft_section(d_section.strip(), d_notes.strip())

                st.subheader(f"Drafted: {d_section}")
                st.info(result)
                st.download_button(
                    "Download as .txt",
                    data=result,
                    file_name=f"{d_section.replace(' ', '_')}_draft.txt",
                    mime="text/plain",
                )
            except Exception as exc:
                st.error(f"Error: {exc}")
