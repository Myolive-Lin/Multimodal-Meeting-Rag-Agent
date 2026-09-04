from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    frame_interval: float = 5.0
    whisper_model_size: str = "base"
    language: str = 'zh'
    device: str = 'cpu'
    app_name: str = "enterprise-agent"
    llm_model: str
    postgres_url: str
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rag_directory: str = "metting_notes"
    model_url: str
    token_api_key: str
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str ="https://cloud.langfuse.com"


settings = Settings()
