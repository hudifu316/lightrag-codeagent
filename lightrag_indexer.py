import os
from ragindex.config import LightRAGConfig
from ragindex.document_index import DocumentIndexTool
from ragindex.query_tool import LightRAGQueryTool
from dotenv import load_dotenv

def create_ragindex_tools(config=None):
    if config is None:
        config = LightRAGConfig()
    return [DocumentIndexTool(config=config), LightRAGQueryTool(config=config)]

async def cleanup_resources():
    from ragindex.openai_client import OpenAIClientManager
    await OpenAIClientManager.close_all()

if __name__ == "__main__":
    load_dotenv()
    import asyncio
    # 設定例
    config = LightRAGConfig(
        storage_dir="./storage_dir",
        model_name="gpt-4.1",
        parallel_num=5,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_base_url=os.getenv("OPENAI_BASE_URL"),
    )
    tools = create_ragindex_tools(config)

    def example_usage():
        index_tool = tools[0]
        query_tool = tools[1]
        try:
            # インデックス化例
            result = index_tool._run("./read_dir")
            print(f"インデックス化結果: {result}")

            # 質問応答例
            answer = query_tool._run("UserBillController.warikan()の呼び出しシーケンス図をMermaid記法で出力してください")
            print(f"回答: {answer}")
            pass
        finally:
            asyncio.run(cleanup_resources())

    example_usage()
