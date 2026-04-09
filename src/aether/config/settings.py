from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')
    
    # LLM Configuration
    anthropic_api_key: str
    claude_model: str = "claude-opus-4-20250514"
    
    # Vector Store
    pinecone_api_key: str
    pinecone_index_name: str = "nice-ng97-guidelines"
    pinecone_environment: str = "us-east-1-aws"
    
    # OpenAI for embeddings
    openai_api_key: str
    
    # Logging
    log_level: str = "INFO"


settings = Settings()