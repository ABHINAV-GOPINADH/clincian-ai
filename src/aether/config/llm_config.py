"""LLM Configuration for Ollama (Local LLM)"""

import os
from crewai import LLM
from aether.config.settings import settings
from crewai import Agent

# Manager LLM - For orchestration (lower temperature)
manager_llm = LLM(
    model=f"ollama/{settings.ollama_model}",
    base_url=settings.ollama_base_url,
    temperature=0.3,
)

# Agent LLM - For specialist tasks (balanced temperature)
agent_llm = LLM(
    model=f"ollama/{settings.ollama_model}",
    base_url=settings.ollama_base_url,
    temperature=0.5,
)

# Structured LLM - For JSON outputs (lowest temperature)
structured_llm = LLM(
    model=f"ollama/{settings.ollama_model}",
    base_url=settings.ollama_base_url,
    temperature=0.2,
)

def get_strict_clinical_llm():
    hf_token = os.getenv("HUGGINGFACE_API_KEY") 
    
    # Let's verify the token is actually loading!
    if not hf_token:
        raise ValueError("❌ HUGGINGFACE_API_KEY is empty! The .env file is not loading.")
    else:
        print(f"✅ Token loaded successfully! Starts with: {hf_token[:5]}...")
    llm= LLM(
        model="huggingface/Qwen/Qwen2.5-7B-Instruct",
        task="text-generation",
        max_tokens=1024,
        temperature=0.01,
        repetition_penalty=1.03,
        timeout=120,
        api_key=settings.huggingface_api_key,
    )
    return llm
strict_gemma_llm = get_strict_clinical_llm()

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