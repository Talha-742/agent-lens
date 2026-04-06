"""
config.py — Settings & Constants for AgentLens
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ─── Ollama / OpenAI-compatible endpoint ──────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY: str  = "ollama"          # dummy key — required by OpenAI SDK
OLLAMA_MODEL: str    = os.getenv("OLLAMA_MODEL", "qwen3.5:cloud")

# ─── Web Search ───────────────────────────────────────────────────────────────
MAX_SEARCH_RESULTS: int = 8              # DDGS results per query
SEARCH_TIMEOUT: int     = 15            # seconds

# ─── App Meta ─────────────────────────────────────────────────────────────────
APP_TITLE    = "AgentLens 🔍"
APP_SUBTITLE = "AI-Powered LLM Discovery for Agentic Workflows"
APP_VERSION  = "1.0.0"

# ─── Cost tier colour mapping (for UI badges) ─────────────────────────────────
COST_COLORS = {
    "Free":   "#22c55e",   # green
    "Low":    "#84cc16",   # lime
    "Medium": "#f59e0b",   # amber
    "High":   "#ef4444",   # red
    "Unknown":"#94a3b8",   # slate
}

# ─── Suitability score colour thresholds ──────────────────────────────────────
def score_color(score: float) -> str:
    if score >= 8.5:
        return "#22c55e"
    elif score >= 6.5:
        return "#f59e0b"
    else:
        return "#ef4444"
