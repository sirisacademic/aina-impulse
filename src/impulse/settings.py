from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

# !!!! These values are overriden by .env in the project root !!!!

class Settings(BaseSettings):
    embedder_model_name: str = "nicolauduran45/mRoBERTA_retrieval-scientific_domain"
    index_dir: str = "data/index"
    vector_backend: str = "hnsw"
    hnsw_m: int = 16
    hnsw_ef_construct: int = 200
    hnsw_ef_query: int = 50
    hnsw_space: str = "cosine"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    html_port: int = 8080
    require_index: bool = False
    auto_build_from_pickle: str = ""
    auto_build_limit: int = 0
    
    # Query parser settings
    query_parser_model: str = "models/impulse-7b-tools-v3-merged"
    query_parser_quantize: Optional[str] = None
    query_parser_prompt: str = "data/training/salamandra_finetuning_prompt.txt"
    
    # NOTE: This is Pydantic's model_config: https://docs.pydantic.dev/2.0/usage/model_config/
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()


