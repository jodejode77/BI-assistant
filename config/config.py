import os
from pathlib import Path
from typing import Optional
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    from pydantic import BaseSettings
    SettingsConfigDict = dict
from pydantic import Field, AliasChoices


class Settings(BaseSettings):
    telegram_token: str = Field(..., alias="TELEGRAM_BOT_TOKEN")
    telegram_admin_ids: list[int] = Field(default_factory=list, alias="TELEGRAM_ADMIN_IDS")
    
    db_host: str = Field("localhost", alias="DB_HOST")
    db_port: int = Field(5432, alias="DB_PORT")
    db_name: str = Field("home_credit", alias="DB_NAME")
    db_user: str = Field("postgres", alias="DB_USER")
    db_password: str = Field("postgres", alias="DB_PASSWORD")
    
    openai_api_key: Optional[str] = Field(None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, alias="ANTHROPIC_API_KEY")
    gemini_api_key: Optional[str] = Field(None, alias="GEMINI_API_KEY")
    mistral_api_key: Optional[str] = Field(None, alias="MISTRAL_API_KEY")
    llm_provider: str = Field("gemini", alias="LLM_PROVIDER")
    llm_model: str = Field("gemini-pro", alias="LLM_MODEL")
    
    embedding_model: str = Field("all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    chroma_persist_dir: str = Field("./chroma_db", alias="CHROMA_PERSIST_DIR")
    faiss_index_path: str = Field("./faiss_index", alias="FAISS_INDEX_PATH")
    rag_chunk_size: int = Field(1000, alias="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(200, alias="RAG_CHUNK_OVERLAP")
    rag_top_k: int = Field(5, alias="RAG_TOP_K")
    
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    log_file: str = Field("logs/bot.log", alias="LOG_FILE")
    
    max_query_time: int = Field(30, alias="MAX_QUERY_TIME")
    max_rows_return: int = Field(100, alias="MAX_ROWS_RETURN")
    cache_ttl: int = Field(3600, alias="CACHE_TTL")
    
    data_dir: Path = Field(Path("./data"), alias="DATA_DIR")
    temp_dir: Path = Field(Path("./temp"), alias="TEMP_DIR")
    
    @property
    def database_url(self) -> str:
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        populate_by_name=True
    )


settings = Settings()
