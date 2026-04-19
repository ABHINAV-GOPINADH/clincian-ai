"""LLM Configuration — Ollama (Local) + Mistral + HuggingFace"""

import os
from crewai import LLM
from aether.config.settings import settings

# ─────────────────────────────────────────
# 🟢 OLLAMA — Local LLMs
# ─────────────────────────────────────────

manager_llm = LLM(
    model=f"ollama/{settings.ollama_model}",
    base_url=settings.ollama_base_url,
    temperature=0.3,  # Low — for orchestration
)

agent_llm = LLM(
    model=f"ollama/{settings.ollama_model}",
    base_url=settings.ollama_base_url,
    temperature=0.5,  # Balanced — for specialist tasks
)

structured_llm = LLM(
    model=f"ollama/{settings.ollama_model}",
    base_url=settings.ollama_base_url,
    temperature=0.2,  # Lowest — for JSON outputs
)

def get_llm(temperature: float = 0.5) -> LLM:
    """Create an Ollama LLM with custom temperature."""
    return LLM(
        model=f"ollama/{settings.ollama_model}",
        base_url=settings.ollama_base_url,
        temperature=temperature,
    )

def get_manager_llm() -> LLM:
    return manager_llm

def get_agent_llm() -> LLM:
    return agent_llm

def get_structured_llm() -> LLM:
    return structured_llm


# ─────────────────────────────────────────
# 🔵 MISTRAL — Cloud LLM
# ─────────────────────────────────────────

def get_mistral() -> LLM:
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("❌ MISTRAL_API_KEY is not set in environment.")
    return LLM(
        model="mistral/mistral-large-latest",
        api_key=api_key,
        temperature=0.01,
        max_tokens=1024,
    )


# ─────────────────────────────────────────
# 🟡 HUGGINGFACE — Gemma 4 Clinical LLM
# ─────────────────────────────────────────

def get_strict_clinical_llm() -> LLM:
    """
    Lazy-loaded HuggingFace Gemma LLM.
    Call this function explicitly — do NOT call at module level.
    """
    hf_token = os.getenv("HUGGINGFACE_API_KEY")

    if not hf_token:
        raise ValueError("❌ HUGGINGFACE_API_KEY is empty! Check your .env file.")

    print(f"✅ HuggingFace token loaded. Starts with: {hf_token[:5]}...")

    return LLM(
        model="huggingface/Qwen/Qwen2.5-7B-Instruct",  # ✅ Verify exact model ID on HF Hub
        task="text-generation",
        max_tokens=1024,
        temperature=0.01,
        timeout=120,
        api_key=hf_token,
    )