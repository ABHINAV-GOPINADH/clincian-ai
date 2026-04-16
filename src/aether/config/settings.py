from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env', 
        env_file_encoding='utf-8',
        extra='ignore'  # Ignore extra fields in .env
    )
    
    # LLM Configuration - Ollama
    ollama_model: str = "llama3.2:3b"
    ollama_base_url: str = "http://localhost:11434"
    
    # Vector Store
    pinecone_api_key: str = ""
    pinecone_index_name: str = "nice-ng97-guidelines"
    pinecone_environment: str = "us-east-1"  # Changed from us-east-1-aws
    
    # Logging
    log_level: str = "INFO"

    # Hugging Face API Key
    huggingface_api_key: str = ""
    
    # Embedding dimension for local embeddings
    embedding_dimension: int = 384  # Changed to 384 for all-MiniLM-L6-v2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Validate only Pinecone key (Ollama doesn't need API key)
        if not self.pinecone_api_key:
            raise ValueError(
                "PINECONE_API_KEY not found in .env file. "
                "Sign up at https://www.pinecone.io/ to get a free API key"
            )


settings = Settings()