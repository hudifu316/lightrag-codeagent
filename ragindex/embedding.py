import threading
from transformers import AutoModel, AutoTokenizer
from lightrag.llm.hf import hf_embed

class EmbeddingModelManager:
    """埋め込みモデルの管理クラス"""
    _models = {}
    _lock = threading.Lock()

    @classmethod
    def get_model_and_tokenizer(cls, model_name: str):
        if model_name not in cls._models:
            with cls._lock:
                if model_name not in cls._models:
                    model = AutoModel.from_pretrained(model_name)
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    cls._models[model_name] = (model, tokenizer)
        return cls._models[model_name]

def embedding_func_factory(model_name: str):
    def embedding_func(texts):
        model, tokenizer = EmbeddingModelManager.get_model_and_tokenizer(model_name)
        return hf_embed(texts, tokenizer=tokenizer, embed_model=model)
    return embedding_func
