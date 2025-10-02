
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    embedder_model_name: str = "langtech-innovation/mRoBERTA_retrieval"
    index_dir: str = "data/index"
    vector_backend: str = "hnsw"
    hnsw_m: int = 16
    hnsw_ef_construct: int = 200
    hnsw_ef_query: int = 50
    hnsw_space: str = "cosine"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    require_index: bool = False
    auto_build_from_pickle: str = ""
    auto_build_limit: int = 0
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
