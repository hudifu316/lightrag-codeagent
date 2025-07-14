
import os
import asyncio
from typing import AsyncIterator, Optional
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from pydantic import Field
from ragindex.config import LightRAGConfig
from ragindex.openai_client import openai_complete_func
from ragindex.embedding import embedding_func_factory
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

class LightRAGQueryTool(BaseTool):
    """LightRAGを使用して質問応答を行うツール"""

    name: str = "lightrag_query"
    description: str = """
    インデックス化された知識グラフに対して質問を行い、回答を取得します。
    入力: 質問文（文字列）
    出力: 回答文
    """

    config: LightRAGConfig = Field(default_factory=LightRAGConfig)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, config: Optional[LightRAGConfig] = None, **kwargs):
        if config:
            kwargs["config"] = config
        super().__init__(**kwargs)
        self._validate_config()

    def _validate_config(self):
        """設定の検証"""
        api_key = self.config.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API キーが設定されていません。")

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """同期実行"""
        return asyncio.run(self._arun(query, run_manager))

    async def _arun(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """非同期実行"""
        try:
            # RAGを初期化
            rag = await self._initialize_rag()

            # クエリパラメータの設定
            query_param = QueryParam(
                mode="hybrid",
                top_k=5,
                conversation_history=[],
                history_turns=5
            )

            # クエリ実行
            response = await rag.aquery(query=query, param=query_param)
            # Handle async iterator (streaming) or string response
            if isinstance(response, AsyncIterator):
                result = []
                async for chunk in response:
                    result.append(chunk)
                return "".join(result)
            return response

        except Exception as e:
            return f"クエリエラー: {str(e)}"

    async def _initialize_rag(self) -> LightRAG:
        """LightRAGの初期化"""
        api_key = self.config.openai_api_key or os.getenv("OPENAI_API_KEY")
        base_url = self.config.openai_base_url or os.getenv("OPENAI_BASE_URL")
        assert api_key is not None, "OpenAI API キーが設定されていません。"

        # Pickle可能なLLM関数を作成
        def llm_func(prompt: str, system_prompt: str = "", **kwargs):
            return openai_complete_func(
                prompt, system_prompt,
                model_name=self.config.model_name,
                api_key=api_key,
                base_url=base_url or None,
                **kwargs
            )

        # Pickle可能な埋め込み関数を作成
        embedding_func = embedding_func_factory(self.config.embedding_model)

        rag = LightRAG(
            working_dir=self.config.storage_dir,
            llm_model_func=llm_func,
            llm_model_max_token_size=self.config.llm_max_token_size,
            embedding_func=EmbeddingFunc(
                func=embedding_func,
                max_token_size=self.config.emb_max_token_size,
                embedding_dim=self.config.embedding_dim
            )
        )

        await rag.initialize_storages()
        return rag
