# Research Agent — IBM Granite + IBM watsonx.ai

> **AICTE Problem Statement #1** — An AI-powered Research Agent that autonomously searches academic literature, summarises papers, manages citations, generates reports, and proposes research hypotheses — all powered by **IBM Granite** models on **IBM Cloud (Lite tier)**.

---

## 📐 Architecture

```
main.py (CLI)
    └── ResearchAgent (agent.py)
            ├── LiteratureSearcher   (literature_search.py)  ← arXiv + Semantic Scholar + Watson Discovery
            ├── GraniteClient        (granite_client.py)      ← IBM Granite via watsonx.ai
            ├── Summarizer           (summarizer.py)          ← per-paper + multi-paper synthesis
            ├── ReferenceManager     (references.py)          ← BibTeX / APA / JSON export
            └── ReportGenerator      (report_generator.py)    ← Markdown + PDF reports
```

### IBM Cloud Services Used

| Service | Purpose | Tier |
|---|---|---|
| **IBM watsonx.ai** | LLM inference host | Lite (free) |
| **IBM Granite 13B Instruct v2** | Summarisation, drafting, hypotheses | Lite |
| **IBM Slate 125M Embeddings** | Semantic text similarity | Lite |
| **IBM Watson Discovery** *(optional)* | Enterprise literature search | Lite (optional) |

---

## 🚀 Quick Start

### 1. Clone & create virtual environment

```bash
git clone <repo-url>
cd "AICTE - Research Agent"
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure IBM Cloud credentials

```bash
copy .env.example .env        # Windows
# cp .env.example .env        # macOS/Linux
```

Open `.env` and fill in **at minimum**:

```env
IBM_CLOUD_API_KEY=<your IBM Cloud API key>
WATSONX_PROJECT_ID=<your watsonx.ai project ID>
```

> **How to get credentials:**
> 1. Sign up at https://cloud.ibm.com (Lite/free tier)
> 2. Create a **watsonx.ai** service instance
> 3. Go to **IAM → API keys** → create a new key
> 4. In watsonx.ai, create a project → copy its **Project ID** from the Manage tab

### 4. Run the agent

```bash
# Full pipeline — search, summarise, synthesise, report
python main.py "transformer models in biomedical NLP"

# Limit results
python main.py "quantum computing error correction" --results 8

# Skip PDF generation
python main.py "climate change ML predictions" --no-pdf

# Search only
python main.py search "CRISPR gene editing"

# Summarise papers
python main.py summarize "deep learning drug discovery" --results 5

# Generate hypotheses from a context file
python main.py hypotheses --context context.txt --n 5

# Draft a paper section
python main.py draft --section "Introduction" --notes "notes.txt"
```

---

## 📁 Project Structure

```
AICTE - Research Agent/
├── .env.example               ← template (copy to .env and fill in)
├── .env                       ← your secrets (never commit)
├── requirements.txt
├── setup.py
├── main.py                    ← CLI entry point
├── research_agent/
│   ├── __init__.py
│   ├── agent.py               ← ResearchAgent orchestrator
│   ├── config.py              ← .env loader + Config dataclass
│   ├── granite_client.py      ← IBM Granite / watsonx.ai wrapper
│   ├── literature_search.py   ← arXiv + Semantic Scholar + Discovery
│   ├── summarizer.py          ← paper + synthesis summarisation
│   ├── references.py          ← BibTeX / APA / JSON citation manager
│   └── report_generator.py   ← Markdown + PDF report builder
├── tests/
│   ├── test_config.py
│   └── test_references.py
└── output/                    ← generated reports and references
```

---

## ⚙️ Configuration Reference (`.env`)

| Variable | Default | Description |
|---|---|---|
| `IBM_CLOUD_API_KEY` | — | **Required.** IBM Cloud IAM API key |
| `WATSONX_PROJECT_ID` | — | **Required.** watsonx.ai project ID |
| `WATSONX_URL` | `https://us-south.ml.cloud.ibm.com` | watsonx.ai regional endpoint |
| `GRANITE_MODEL_ID` | `ibm/granite-13b-instruct-v2` | Text generation model |
| `GRANITE_EMBEDDING_MODEL_ID` | `ibm/slate-125m-english-rtrvr` | Embedding model |
| `DISCOVERY_API_KEY` | *(blank)* | Watson Discovery API key (optional) |
| `DISCOVERY_URL` | *(blank)* | Watson Discovery URL (optional) |
| `DISCOVERY_PROJECT_ID` | *(blank)* | Watson Discovery project ID (optional) |
| `MAX_SEARCH_RESULTS` | `10` | Papers to retrieve per query |
| `MAX_NEW_TOKENS` | `800` | Granite generation token limit |
| `TEMPERATURE` | `0.3` | Generation temperature |
| `TOP_P` | `0.9` | Nucleus sampling parameter |
| `OUTPUT_DIR` | `./output` | Directory for reports and references |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## 🧪 Tests

```bash
pip install pytest
pytest tests/ -v
```

Unit tests for `ReferenceManager` and `Config` run without IBM API credentials.

---

## 📤 Output Files

For every research session, the agent writes to `./output/`:

| File | Description |
|---|---|
| `<topic>.md` | Full Markdown research report |
| `<topic>.pdf` | PDF version of the report |
| `<topic>.bib` | BibTeX references |
| `<topic>_apa.txt` | APA-style plain-text references |
| `<topic>_refs.json` | Machine-readable JSON reference list |

---

## 🤖 IBM Granite Capabilities Used

| Capability | Prompt Type | Output |
|---|---|---|
| **Summarisation** | Abstract → 2–3 sentence summary | Per-paper summary |
| **Synthesis** | Multi-paper context → narrative | Literature review paragraph |
| **Hypothesis generation** | Research context → numbered list | 3–5 testable hypotheses |
| **Section drafting** | Notes → formal academic prose | Paper section text |
| **Key findings extraction** | Abstract → bullet points | Structured findings |
| **Executive summary** | Query + paper count → summary | Report introduction |

---

## 📜 License

Apache 2.0 — see LICENSE file.

---

*Built for the AICTE Research Agent Challenge — IBM Cloud Lite + IBM Granite mandatory.*
