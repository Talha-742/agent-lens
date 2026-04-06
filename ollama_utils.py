"""
ollama_utils.py — Local Ollama model utilities for AgentLens
"""
from __future__ import annotations

import ollama
from typing import Optional


# ─── List locally installed models ────────────────────────────────────────────

def list_local_models() -> list[dict]:
    """
    Return a list of all locally installed Ollama models with enriched metadata.
    Each dict contains: name, family, parameters, quantization, size_gb, modified.
    """
    try:
        response = ollama.list()
        models = []
        for m in response.models:
            details = m.details if m.details else {}
            name = m.model or "unknown"

            # Parse human-friendly size
            size_bytes = m.size or 0
            size_gb = round(size_bytes / (1024 ** 3), 2)

            models.append({
                "name":          name,
                "family":        getattr(details, "family", "unknown") or "unknown",
                "parameters":    getattr(details, "parameter_size", "unknown") or "unknown",
                "quantization":  getattr(details, "quantization_level", "unknown") or "unknown",
                "size_gb":       size_gb,
                "format":        getattr(details, "format", "unknown") or "unknown",
                "modified":      str(m.modified_at)[:10] if m.modified_at else "unknown",
            })
        return models
    except Exception as e:
        return []


def get_ollama_status() -> dict:
    """
    Check if Ollama is running and return a status dict.
    """
    try:
        models = list_local_models()
        return {
            "running": True,
            "model_count": len(models),
            "message": f"✅ Ollama running — {len(models)} model(s) installed",
        }
    except Exception as e:
        return {
            "running": False,
            "model_count": 0,
            "message": f"❌ Ollama not reachable: {e}",
        }


def test_model(model_name: str, prompt: str) -> str:
    """
    Send a test prompt to a locally installed Ollama model and return the response.
    Strips <think>…</think> blocks that reasoning models emit.
    """
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.7},
        )
        raw = response.message.content or ""
        return _strip_think_tags(raw)
    except Exception as e:
        return f"Error testing model: {e}"


def score_local_model_for_workflow(model: dict, workflow_keywords: list[str]) -> tuple[float, str]:
    """
    Heuristically score a local Ollama model for a given workflow.
    Returns (score 0-10, reason string).
    """
    name_lower = (model["name"] + model["family"]).lower()
    score = 5.0
    reasons = []

    keyword_map = {
        "code":      ["code", "coder", "starcoder", "deepseek", "codellama"],
        "reasoning": ["qwen", "deepseek", "llama", "mistral", "phi"],
        "math":      ["math", "qwen", "deepseek"],
        "vision":    ["vision", "llava", "bakllava", "minicpm"],
        "chat":      ["chat", "instruct", "llama", "mistral", "qwen", "gemma"],
        "tool":      ["qwen", "llama", "mistral", "hermes", "functionary"],
        "fast":      ["phi", "gemma", "qwen", "tinyllama"],
        "large":     ["70b", "72b", "405b", "110b"],
    }

    for wk in workflow_keywords:
        for category, tags in keyword_map.items():
            if wk.lower() in category:
                if any(t in name_lower for t in tags):
                    score = min(10.0, score + 1.0)
                    reasons.append(f"Matches {category} workloads")

    # Bonus for larger parameter models
    params = model["parameters"].lower()
    if any(p in params for p in ["70b", "72b", "34b", "32b"]):
        score = min(10.0, score + 1.5)
        reasons.append("Large parameter count — higher capability")
    elif any(p in params for p in ["13b", "14b", "8b", "7b"]):
        score = min(10.0, score + 0.5)
        reasons.append("Mid-size model — good capability/speed balance")

    reason = "; ".join(set(reasons)) if reasons else "General purpose model"
    return round(score, 1), reason


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models."""
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
