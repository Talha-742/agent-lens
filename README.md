<div align="center">

```
 █████╗  ██████╗ ███████╗███╗   ██╗████████╗██╗     ███████╗███╗   ██╗███████╗
██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝██║     ██╔════╝████╗  ██║██╔════╝
███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ██║     █████╗  ██╔██╗ ██║███████╗
██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ██║     ██╔══╝  ██║╚██╗██║╚════██║
██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ███████╗███████╗██║ ╚████║███████║
╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚══════╝╚═╝  ╚═══╝╚══════╝
```

### 🔍 AI-Powered LLM Discovery for Agentic Workflows

*Describe your workflow. Get the perfect model — instantly.*

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-API_Backend-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-5-7952B3?style=for-the-badge&logo=bootstrap&logoColor=white)](https://getbootstrap.com)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![uv](https://img.shields.io/badge/uv-Package_Manager-7c3aed?style=for-the-badge)](https://github.com/astral-sh/uv)

<br/>

![AgentLens Demo](https://placehold.co/900x420/1e293b/94a3b8?text=AgentLens+%E2%80%94+Screenshot+Preview)

> **No OpenAI key required.** AgentLens runs 100% locally using [Ollama](https://ollama.com),  
> searches the web via DuckDuckGo, and delivers structured LLM recommendations in seconds.

</div>

---

## 📖 Table of Contents

- [What is AgentLens?](#-what-is-agentlens)
- [Features](#-features)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Running the App](#-running-the-app)
- [How It Works](#-how-it-works)
- [Example Interaction](#-example-interaction)
- [Output Format](#-output-format)
- [Tech Stack](#-tech-stack)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🤔 What is AgentLens?

**AgentLens** is a local-first, AI-powered LLM discovery assistant. You describe the agentic workflow you're building — in plain English — and AgentLens searches the web for up-to-date model information, reasons over it using a local LLM, and returns a ranked, structured list of the best models for your use case.

Think of it as your **AI engineering advisor** that reads the internet so you don't have to.

```
You: "I'm building a marketing automation agent with A/B testing, 
      report generation, and audience segmentation."

AgentLens: Here are 7 ranked LLMs for your workflow, with tool calling 
           support, cost tier, context window, and suitability scores.
```

It runs entirely on your machine — **zero cloud API costs, zero data leaks.**

---

## ✨ Features

| Feature | Description |
|---|---|
| 🌐 **Live Web Search** | DuckDuckGo search fetches real-time LLM benchmarks, releases & pricing |
| 🧠 **Local LLM Reasoning** | Qwen3 (via Ollama) analyses search results and produces structured recommendations |
| 🏠 **Local Model Discovery** | Automatically lists all your installed Ollama models with metadata |
| 📊 **Comparison Table** | Side-by-side comparison: provider, parameters, cost, tool support, suitability score |
| 🃏 **Model Cards** | Rich cards with key features, tool-calling details, and workflow fit explanation |
| 🧪 **In-App Model Testing** | Send a test prompt to any local Ollama model directly from the UI |
| 🔋 **Workflow Scoring** | Local models are heuristically scored for relevance to your described workflow |
| 🕘 **Search History** | Sidebar keeps track of your queries within the session |
| ⚙️ **Status Dashboard** | Live Ollama connection status and active model info in the sidebar |
| 💯 **Zero Cloud Dependency** | No OpenAI key, no external API costs — fully local inference |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AgentLens Pipeline                       │
└─────────────────────────────────────────────────────────────────┘

  User Query
      │
      ▼
┌──────────────────┐   REST/JSON   ┌──────────────────────┐
│  HTML/CSS/JS     │──────────────▶│   api.py (Flask)     │
│  Bootstrap 5     │               │                      │
│  frontend/       │◀──────────────│  POST /api/recommend │
│  index.html      │               │  GET  /api/local-models
└──────────────────┘               │  POST /api/test-model│
                                   │  GET  /api/status    │
                                   └──────────┬───────────┘
                                              │
                              ┌───────────────▼───────────────┐
                              │        agent_core.py          │
                              │                               │
                              │  1. DDGS Web Search ──▶ DuckDuckGo
                              │  2. Build context   ◀──────────┘
                              │  3. Call Ollama                │
                              │     (OpenAI SDK)  ──▶ Ollama  │
                              │  4. Parse JSON    ◀───────────┘
                              └───────────────┬───────────────┘
                                              │
                              ┌───────────────▼───────────────┐
                              │       ollama_utils.py         │
                              │  • List local models ──▶ Ollama API
                              │  • Score for workflow          │
                              │  • Test model output ◀────────┘
                              └───────────────────────────────┘
```

**Key design decision:** The OpenAI Python SDK is used to talk to Ollama's OpenAI-compatible endpoint (`http://localhost:11434/v1`). This means you get the clean OpenAI SDK interface without needing an OpenAI API key — just point `base_url` at Ollama.

---

## 📁 Project Structure

```
agentlens/
│
├── frontend/
│   └── index.html          # 🖥️  Full UI — HTML, CSS, Bootstrap 5, vanilla JS
├── api.py                  # 🌐  Flask REST API — bridges frontend ↔ backend
├── agent_core.py           # 🧠  Core engine: DDGS search + Ollama inference + JSON parsing
├── ollama_utils.py         # 🏠  Local Ollama model utilities: list, score, test
├── config.py               # ⚙️  All settings, constants, and environment loading
├── .env                    # 🔐  Local config (Ollama URL + model name) — gitignored
├── .env.example            # 📋  Safe-to-commit example env file
└── requirements.txt        # 📦  Python dependencies
```

---

## 🧰 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.10+** — [python.org](https://python.org)
- **uv** (package manager) — [astral.sh/uv](https://github.com/astral-sh/uv)
- **Ollama** — [ollama.com](https://ollama.com)
- A pulled Ollama model (e.g. `qwen3.5:cloud`)

To install Ollama and pull the model:

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the model used by AgentLens
ollama pull qwen3.5:cloud
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/agentlens.git
cd agentlens
```

### 2. Create and activate a virtual environment

```bash
uv venv --python 3.10
source .venv/bin/activate        # Linux / macOS
# or
.venv\Scripts\activate           # Windows
```

### 3. Install all dependencies

```bash
uv add openai ollama flask flask-cors python-dotenv pandas duckduckgo-search
```

Or restore from `requirements.txt`:

```bash
uv pip install -r requirements.txt
```

---

## ⚙️ Configuration

Copy the example env file and adjust if needed:

```bash
cp .env.example .env
```

Your `.env` file:

```env
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=qwen3.5:cloud
```

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama's OpenAI-compatible endpoint |
| `OLLAMA_MODEL` | `qwen3.5:cloud` | The model used for LLM reasoning |

> **Tip:** You can swap `OLLAMA_MODEL` for any model you have installed — e.g. `qwen2.5:72b`, `llama3.1:8b`, `mistral:7b`.

---

## ▶️ Running the App

### Step 1 — Start Ollama

```bash
ollama serve
```

> Skip this if Ollama is already running as a system service.

### Step 2 — Launch AgentLens

```bash
python api.py
```

The app will open at `http://localhost:5000`.

---

## 🔬 How It Works

AgentLens follows a clean 5-step pipeline for every query:

```
Step 1 │ DESCRIBE   You type your agentic workflow in plain English
       │
Step 2 │ SEARCH     Two targeted DuckDuckGo queries fetch up-to-date
       │            LLM benchmark data, release notes, and pricing
       │
Step 3 │ REASON     Qwen3 (running locally via Ollama) reads the search
       │            context and your query, then returns a structured
       │            JSON object with 5–8 ranked model recommendations
       │
Step 4 │ ENRICH     Your locally installed Ollama models are listed,
       │            scored, and displayed alongside cloud recommendations
       │
Step 5 │ DISPLAY    Results render as model cards, tabbed views,
       │            a comparison table, and an interactive test console
```

### Why OpenAI SDK + Ollama?

Ollama exposes an OpenAI-compatible REST API at `/v1`. By setting `base_url=http://localhost:11434/v1` and `api_key="ollama"` (a dummy value), the full OpenAI Python SDK works with zero code changes — giving you clean, future-proof client code without cloud lock-in.

---

## 💬 Example Interaction

**User Query:**
> *"I'm building a marketing agentic flow that automates campaign creation, audience targeting, A/B test analysis, and report generation. Recommend the best LLM models."*

**AgentLens Response:**

```
Workflow Analysis
─────────────────────────────────────────────────────────────────
This workflow requires strong tool/function calling for multi-step
orchestration, large context handling for campaign documents, and
reliable structured output for A/B analysis and reports.

Key Requirements: Tool Calling • Long Context • Structured Output
                  Reasoning • Cost Efficiency

┌─────────────────────────────────────────────────────────────────┐
│  #1  GPT-4o                    Score: 9.4/10  ☁️ Cloud          │
│  #2  Claude 3.5 Sonnet         Score: 9.1/10  ☁️ Cloud          │
│  #3  Qwen2.5-72B-Instruct      Score: 8.8/10  🏠 Local/Both     │
│  #4  Llama 3.1 405B            Score: 8.5/10  🏠 Local/Both     │
│  #5  Gemini 1.5 Pro            Score: 8.3/10  ☁️ Cloud          │
│  #6  Mistral Large             Score: 7.9/10  ☁️ Cloud          │
│  #7  DeepSeek-V3               Score: 7.6/10  🏠 Local/Both     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📋 Output Format

Each recommended model is displayed with the following structured fields:

| Field | Example |
|---|---|
| **Rank** | `#1` |
| **Model Name** | `Qwen2.5-72B-Instruct` |
| **Provider** | `Alibaba / Ollama` |
| **Parameters** | `72B` |
| **Description** | `Alibaba's flagship open-source model with strong reasoning…` |
| **Key Features** | `Tool Calling`, `128K Context`, `Multilingual`, `Free` |
| **Tool Calling Support** | `Yes` / `No` / `Partial` |
| **Tool Calling Details** | `Supports parallel function calling with structured outputs` |
| **Cost Tier** | `Free` / `Low` / `Medium` / `High` |
| **Context Window** | `128K tokens` |
| **Suitability Score** | `9.1 / 10` |
| **Suitability Reason** | `Ideal for agentic workflows requiring reliable tool use…` |
| **Model Type** | `Cloud` / `Local` / `Both` |

---

## 🧱 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | [HTML / CSS / Bootstrap 5](https://getbootstrap.com) | Responsive web UI — no framework, no build step |
| **JavaScript** | Vanilla JS (`fetch` / `async-await`) | API calls, dynamic rendering, session history |
| **API Layer** | [Flask](https://flask.palletsprojects.com) + [Flask-CORS](https://flask-cors.readthedocs.io) | REST API bridging frontend and Python backend |
| **LLM Client** | [OpenAI Python SDK](https://github.com/openai/openai-python) | Calls Ollama's OpenAI-compatible endpoint |
| **Local Inference** | [Ollama](https://ollama.com) | Runs LLMs locally |
| **Reasoning Model** | `qwen3.5:cloud` | Analyses web results & generates recommendations |
| **Web Search** | [duckduckgo-search](https://github.com/deedy5/duckduckgo_search) | Real-time LLM data retrieval |
| **Data** | [Pandas](https://pandas.pydata.org) | Comparison tables & DataFrames |
| **Config** | [python-dotenv](https://github.com/theskumar/python-dotenv) | Environment variable management |
| **Package Manager** | [uv](https://github.com/astral-sh/uv) | Fast Python dependency management |

---

## 🗺️ Roadmap

- [ ] **Streaming responses** — stream model output token by token in the UI
- [ ] **Model benchmarks panel** — pull LMSYS Chatbot Arena and Open LLM Leaderboard scores
- [ ] **Workflow templates** — pre-built query templates for common agentic patterns
- [ ] **Export to JSON/CSV** — download recommendation reports
- [ ] **Multi-model comparison mode** — run the same test prompt across multiple local models side-by-side
- [ ] **Persistent search history** — save sessions across app restarts
- [ ] **Cost calculator** — estimate monthly API cost based on token usage

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

```bash
# Fork the repo, then:
git clone https://github.com/Talha-742/agent-lens.git
cd agentlens
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Create a feature branch
git checkout -b feature/your-feature-name

# Make your changes, then open a Pull Request
```

Please follow [PEP 8](https://pep8.org/) for Python code style and keep commits focused and descriptive.

<div align="center">

Built with 🔍 curiosity and ☕ caffeine

**AgentLens** — *because choosing the right LLM shouldn't require a research paper.*

</div>