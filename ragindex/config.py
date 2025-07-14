from pydantic import BaseModel
from typing import Optional

class LightRAGConfig(BaseModel):
    """LightRAG設定クラス"""
    storage_dir: str = "./storage"
    model_name: str = "gpt-4.1"
    llm_max_token_size: int = 8192
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    emb_max_token_size: int = 2048
    chunk_max_tokens: int = 8192
    parallel_num: int = 3
    max_depth: int = 30
    merge_score_threshold: float = 0.9
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
