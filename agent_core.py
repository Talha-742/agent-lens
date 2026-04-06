"""
agent_core.py — LLM recommendation engine
  • Web search via DuckDuckGo (DDGS)
  • LLM inference via Ollama, accessed through the OpenAI-compatible SDK
"""
from __future__ import annotations

import json
import re
import time
from typing import Optional

import pandas as pd
from ddgs import DDGS
from openai import OpenAI

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_API_KEY,
    OLLAMA_MODEL,
    MAX_SEARCH_RESULTS,
    SEARCH_TIMEOUT,
)

# ─── OpenAI SDK client pointed at local Ollama ────────────────────────────────
client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)


# ─── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are AgentLens, a world-class AI engineering advisor that helps developers \
select the best LLM models for their agentic AI workflows.

You will receive:
1. The user's agentic workflow description.
2. Fresh web search snippets about current LLM models.

Your job: analyse the workflow, then recommend 5–8 LLM models (mix of cloud + open-source).

━━━ OUTPUT RULES ━━━
• Respond with ONLY a single valid JSON object — no markdown fences, no preamble, no commentary.
• Do NOT wrap in ```json ... ```.
• Strictly follow the schema below.

━━━ JSON SCHEMA ━━━
{
  "workflow_analysis": "<2-3 sentence analysis of the workflow's key LLM requirements>",
  "key_requirements": ["requirement 1", "requirement 2", "..."],
  "recommendations": [
    {
      "rank": 1,
      "model_name": "<Model Name>",
      "provider": "<Provider>",
      "parameters": "<e.g. 72B | Unknown (proprietary)>",
      "description": "<1-2 sentence summary of the model>",
      "key_features": ["feature 1", "feature 2", "feature 3"],
      "tool_calling_support": "<Yes | No | Partial>",
      "tool_calling_details": "<Brief note on function/tool-calling capability>",
      "cost_tier": "<Free | Low | Medium | High>",
      "context_window": "<e.g. 128K tokens>",
      "suitability_score": 8.5,
      "suitability_reason": "<Why this model fits the workflow>",
      "model_type": "<Cloud | Local | Both>",
      "best_for": "<One-liner on ideal use case>"
    }
  ],
  "search_sources": ["<url or title 1>", "<url or title 2>"]
}

━━━ FEW-SHOT EXAMPLE RECOMMENDATION ━━━
{
  "rank": 1,
  "model_name": "Qwen2.5-72B-Instruct",
  "provider": "Alibaba / Ollama",
  "parameters": "72B",
  "description": "Alibaba's flagship open-source instruction model with excellent reasoning and native tool calling.",
  "key_features": ["Strong tool/function calling", "128K context window", "Multilingual", "Free to run locally"],
  "tool_calling_support": "Yes",
  "tool_calling_details": "Supports OpenAI-style function calling with parallel tool use",
  "cost_tier": "Free",
  "context_window": "128K tokens",
  "suitability_score": 9.1,
  "suitability_reason": "Ideal for agentic workflows requiring reliable tool use, long context, and zero cost",
  "model_type": "Local",
  "best_for": "Complex multi-step agents with tool orchestration"
}
"""


# ─── Web Search ───────────────────────────────────────────────────────────────

def _build_search_queries(user_query: str) -> list[str]:
    """Generate 2 targeted DDGS queries from the user's workflow description."""
    return [
        f"best LLM models {user_query} agentic AI 2025 tool calling",
        f"open source LLM {user_query} benchmark comparison 2025",
    ]


def search_web(user_query: str) -> tuple[list[dict], str]:
    """
    Run DDGS searches and return (raw_results, formatted_context).
    """
    queries = _build_search_queries(user_query)
    all_results: list[dict] = []

    try:
        with DDGS() as ddgs:
            for q in queries:
                results = list(ddgs.text(q, max_results=MAX_SEARCH_RESULTS // 2))
                all_results.extend(results)
                time.sleep(0.3)   # polite delay
    except Exception as e:
        return [], f"[Search unavailable: {e}]"

    # De-duplicate by URL
    seen_urls: set[str] = set()
    unique: list[dict] = []
    for r in all_results:
        url = r.get("href", "")
        if url not in seen_urls:
            seen_urls.add(url)
            unique.append(r)

    formatted = "\n\n".join(
        f"[{i+1}] {r.get('title','')}\n{r.get('body','')}\nURL: {r.get('href','')}"
        for i, r in enumerate(unique[:MAX_SEARCH_RESULTS])
    )
    return unique[:MAX_SEARCH_RESULTS], formatted


# ─── LLM Inference via Ollama (OpenAI SDK) ────────────────────────────────────

def _strip_think_tags(text: str) -> str:
    """Remove <think>…</think> blocks emitted by reasoning models like Qwen3."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _parse_json_response(raw: str) -> dict:
    """
    Robustly extract a JSON object from the model response.
    Handles markdown fences, leading/trailing text, and think-blocks.
    """
    # 1. Strip think tags
    text = _strip_think_tags(raw)

    # 2. Remove markdown code fences if present
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    # 3. Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 4. Try extracting the first {...} block
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from model response:\n{text[:500]}")


def call_ollama(user_query: str, search_context: str) -> dict:
    """
    Call Ollama (via OpenAI SDK) with the web search context and user query.
    Returns the parsed recommendation dict.
    """
    user_message = (
        f"Agentic Workflow Description:\n{user_query}\n\n"
        f"Web Search Results (use these for up-to-date model info):\n{search_context}\n\n"
        "Now respond with the JSON recommendation object only."
    )

    response = client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.3,
    )

    raw = response.choices[0].message.content or ""
    return _parse_json_response(raw)


# ─── Public API ───────────────────────────────────────────────────────────────

def get_recommendations(user_query: str) -> dict:
    """
    Full pipeline:
      1. DDGS web search
      2. Ollama inference with search context
      3. Return structured result dict

    Returns dict with keys:
      - workflow_analysis, key_requirements, recommendations,
        search_sources, search_results (raw DDGS list), error (if any)
    """
    raw_results, search_context = search_web(user_query)

    try:
        parsed = call_ollama(user_query, search_context)
    except Exception as e:
        return {
            "error": str(e),
            "workflow_analysis": "",
            "key_requirements": [],
            "recommendations": [],
            "search_sources": [],
            "search_results": raw_results,
        }

    parsed["search_results"] = raw_results
    parsed.setdefault("error", None)
    return parsed


def recommendations_to_dataframe(recommendations: list[dict]) -> pd.DataFrame:
    """Convert the recommendations list into a display-ready Pandas DataFrame."""
    rows = []
    for r in recommendations:
        rows.append({
            "Rank":              r.get("rank", ""),
            "Model":             r.get("model_name", ""),
            "Provider":          r.get("provider", ""),
            "Parameters":        r.get("parameters", ""),
            "Tool Support":      r.get("tool_calling_support", ""),
            "Cost Tier":         r.get("cost_tier", ""),
            "Context Window":    r.get("context_window", ""),
            "Suitability Score": r.get("suitability_score", ""),
            "Type":              r.get("model_type", ""),
        })
    return pd.DataFrame(rows)


def check_openai_client_status() -> dict:
    """
    Verify that the Ollama endpoint is reachable via the OpenAI SDK.
    """
    try:
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        return {
            "connected": True,
            "message": f"✅ Ollama endpoint connected ({len(model_ids)} model(s) visible)",
            "models": model_ids,
        }
    except Exception as e:
        return {
            "connected": False,
            "message": f"❌ Cannot reach Ollama at {OLLAMA_BASE_URL}: {e}",
            "models": [],
        }
