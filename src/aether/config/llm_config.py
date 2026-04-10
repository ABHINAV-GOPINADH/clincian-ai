"""LLM Configuration for Ollama (Local LLM)"""

from crewai import LLM
from aether.config.settings import settings

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