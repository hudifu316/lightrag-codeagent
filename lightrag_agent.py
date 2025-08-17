import os
import re
import asyncio
from typing import List, Dict, Set, TypedDict

# LangGraph関連のインポート
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage

# OpenAI関連
from openai import OpenAI
from ragindex.config import LightRAGConfig
from ragindex.query_tool import LightRAGQueryTool  # 追加

# 自作モジュール
from ragagent.models import LayerType, FrameworkType, SourceFile
from ragagent.analyzer import LayerClassifier, FrameworkDetector

load_dotenv()

config = LightRAGConfig(
        storage_dir="./storage_dir",
        model_name="gpt-4.1",
        parallel_num=5,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_base_url=os.getenv("OPENAI_BASE_URL"),
        )

class GraphState(TypedDict):
    """LangGraphの状態管理"""
    messages: List[AnyMessage]
    workspace_path: str
    source_files: List[SourceFile]
    framework: FrameworkType
    dependency_graph: Dict[str, Set[str]]
    processing_order: List[str]
    design_documents: List[Dict]
    current_file_index: int
    output_path: str
    extensions: List[str]


# LangGraphのノード関数群
async def scan_files_node(state: GraphState) -> GraphState:
    """ファイルスキャンノード"""
    print("ファイルをスキャンしています...")

    files = []
    workspace_path = state["workspace_path"]
    extensions = state["extensions"]

    for root, dirs, filenames in os.walk(workspace_path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'vendor']]

        for filename in filenames:
            if any(filename.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    extension = os.path.splitext(filename)[1][1:]

                    source_file = SourceFile(
                        path=file_path,
                        content=content,
                        extension=extension,
                        layer=LayerType.UNKNOWN,
                        dependencies=set(),
                        framework=FrameworkType.UNKNOWN
                    )

                    files.append(source_file)
                except Exception as e:
                    print(f"ファイル読み込みエラー: {file_path} - {e}")

    print(f"{len(files)}個のファイルを発見しました")

    state["source_files"] = files
    state["messages"].append(AIMessage(content=f"{len(files)}個のファイルをスキャンしました"))

    return state


async def detect_framework_node(state: GraphState) -> GraphState:
    """フレームワーク検出ノード"""
    print("フレームワークを検出しています...")

    detector = FrameworkDetector()
    framework = detector.detect_framework(state["source_files"])

    state["framework"] = framework
    print(f"検出されたフレームワーク: {framework.value}")

    state["messages"].append(AIMessage(content=f"フレームワークを検出しました: {framework.value}"))

    return state

async def classify_layers_node(state: GraphState) -> GraphState:
    """レイヤ分類ノード"""
    print("レイヤを分類しています...")

    classifier = LayerClassifier()

    for file in state["source_files"]:
        file.layer = classifier.classify_layer(file)
        file.framework = state["framework"]

    # レイヤ別統計
    layer_stats = {}
    for file in state["source_files"]:
        layer = file.layer.value
        layer_stats[layer] = layer_stats.get(layer, 0) + 1

    stats_message = "レイヤ分類結果:\n" + "\n".join([f"- {layer}: {count}ファイル" for layer, count in layer_stats.items()])
    print(stats_message)

    state["messages"].append(AIMessage(content=stats_message))

    return state

async def analyze_dependencies_node(state: GraphState) -> GraphState:
    """依存関係分析ノード（LightRAG使用）"""
    print("依存関係を分析しています...")

    # if not state["rag_instance"]:
    #     print("LightRAGが利用できません。基本的な依存関係分析を実行します。")
    #     state["dependency_graph"] = {}
    #     state["processing_order"] = [f.path for f in state["source_files"]]
    #     return state

    try:
        dependency_graph = {}
        dependency_analyzer_tool = LightRAGQueryTool(config=config)

        for file in state["source_files"]:
            
            # LightRAGを使用して依存関係を分析
            query = f"ファイル '{file.path}' が依存している他のファイルやライブラリを特定してください。import文、require文、include文などを分析してください。"
            result = await dependency_analyzer_tool._arun(query)

            # 結果から依存関係を抽出（簡略化した実装）
            dependencies = set()
            for other_file in state["source_files"]:
                if other_file.path != file.path and other_file.path in result:
                    dependencies.add(other_file.path)

            dependency_graph[file.path] = dependencies
            file.dependencies = dependencies

        # 簡単な処理順序決定（実際のトポロジカルソートは複雑なので簡略化）
        processing_order = list(dependency_graph.keys())

        state["dependency_graph"] = dependency_graph
        state["processing_order"] = processing_order

        print("依存関係分析が完了しました")
        state["messages"].append(AIMessage(content="LightRAGを使用して依存関係を分析しました"))

    except Exception as e:
        print(f"依存関係分析エラー: {e}")
        # フォールバック
        state["dependency_graph"] = {}
        state["processing_order"] = [f.path for f in state["source_files"]]
        state["messages"].append(AIMessage(content=f"依存関係分析でエラーが発生しました: {e}"))

    return state

async def generate_design_docs_node(state: GraphState) -> GraphState:
    """設計書生成ノード"""
    print("設計書を生成しています...")

    generated_docs = []
    client = OpenAI(api_key=config.openai_api_key)
    # テンプレート定義
    templates = {
        LayerType.VIEW: ["入力項目", "出力項目", "操作項目", "入力チェック", "操作時処理概要", "画面遷移先"],
        LayerType.CONTROLLER: ["入力項目", "出力項目", "入力チェック", "処理概要", "例外処理"],
        LayerType.SERVICE: ["入力項目", "出力項目", "入力チェック", "処理概要", "例外処理"],
        LayerType.REPOSITORY: ["データ項目定義", "ER図", "データアクセス処理"],
        LayerType.DOMAIN: ["ドメインオブジェクト", "ビジネスルール", "不変条件"]
    }

    # 出力ディレクトリ作成
    output_path = state["output_path"]
    os.makedirs(output_path, exist_ok=True)
    dependency_analyzer_tool = LightRAGQueryTool(config=config)

    for i, file in enumerate(state["source_files"]):
        print(f"処理中 ({i+1}/{len(state['source_files'])}): {file.path}")

        try:
            # LightRAGを使用して依存関係を分析
            query = f"ファイル '{file.path}' の設計に関連する情報、依存関係、使用パターンを教えてください。"
            context = await dependency_analyzer_tool._arun(query)

            # プロンプト構築
            template_sections = templates.get(file.layer, ["概要", "機能", "実装詳細"])

            prompt = f"""
以下のソースコードを分析して、{file.layer.value}層の設計書をMarkdown形式で生成してください。

## ファイル情報
- パス: {file.path}
- フレームワーク: {state['framework'].value}
- レイヤ: {file.layer.value}

## 関連情報（LightRAGより）
{context}

## 分析対象コード
```
{file.content[:4000]}  # トークン制限のため切り詰め
```

## 出力形式
以下のセクションを含む設計書を作成してください：
{chr(10).join([f"- {section}" for section in template_sections])}

## 特別な指示
- クラス図、シーケンス図、ER図はMermaid記法で記述してください
- 具体的なコードの内容に基づいて分析してください
- {state['framework'].value}フレームワークの特徴を考慮してください
- 日本語で記述してください
"""

            # OpenAI APIで設計書生成
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "あなたは経験豊富なソフトウェアアーキテクトです。ソースコードを分析して詳細な設計書を作成してください。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,
                temperature=0.3
            )

            doc_content = response.choices[0].message.content

            # ファイル保存
            safe_filename = re.sub(r'[^\w\-_.]', '_', os.path.basename(file.path))
            doc_filename = f"{safe_filename}_{file.layer.value}_design.md"
            doc_path = os.path.join(output_path, doc_filename)

            with open(doc_path, 'w', encoding='utf-8') as f:
                f.write(doc_content or "")

            generated_docs.append({
                'source_file': file.path,
                'design_doc': doc_path,
                'layer': file.layer.value
            })

        except Exception as e:
            print(f"設計書生成エラー: {file.path} - {e}")

    state["design_documents"] = generated_docs
    state["messages"].append(AIMessage(content=f"{len(generated_docs)}個の設計書を生成しました"))

    return state

async def generate_summary_node(state: GraphState) -> GraphState:
    """サマリー生成ノード"""
    print("サマリーを生成しています...")

    docs = state["design_documents"]
    dependency_graph = state["dependency_graph"]
    framework = state["framework"]
    output_path = state["output_path"]

    summary_content = f"""# システム設計書サマリー

## 概要
- 分析対象: {state['workspace_path']}
- 検出フレームワーク: {framework.value}
- 生成文書数: {len(docs)}

## アーキテクチャ構成

### レイヤ別ファイル数
"""

    # レイヤ別統計
    layer_stats = {}
    for doc in docs:
        layer = doc['layer']
        layer_stats[layer] = layer_stats.get(layer, 0) + 1

    for layer, count in layer_stats.items():
        summary_content += f"- {layer}: {count}ファイル\n"

    # 依存関係図
    summary_content += "\n## システム依存関係図\n\n"
    summary_content += "```mermaid\n"
    summary_content += "graph TD\n"

    # 依存関係をMermaid形式で出力
    node_id_map = {}
    node_counter = 0

    for file_path in dependency_graph:
        if file_path not in node_id_map:
            node_id_map[file_path] = f"N{node_counter}"
            node_counter += 1

        file_name = os.path.basename(file_path)
        summary_content += f"    {node_id_map[file_path]}[{file_name}]\n"

    for file_path, deps in dependency_graph.items():
        for dep in deps:
            if dep in node_id_map:
                summary_content += f"    {node_id_map[file_path]} --> {node_id_map[dep]}\n"

    summary_content += "```\n"

    # 生成文書一覧
    summary_content += "\n## 生成文書一覧\n\n"
    for doc in docs:
        summary_content += f"- [{os.path.basename(doc['design_doc'])}]({doc['design_doc']}) - {doc['layer']}層\n"

    # サマリーファイル保存
    summary_path = os.path.join(output_path, "README.md")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_content)

    print(f"設計書生成完了! 出力先: {output_path}")
    state["messages"].append(AIMessage(content="サマリー文書を生成しました。処理が完了しました。"))

    return state

# LangGraphワークフロー構築
def create_workflow():
    """LangGraphワークフローを作成"""

    workflow = StateGraph(GraphState)
    dependency_analyzer_tool = LightRAGQueryTool(config=config)
    # ツールノードを追加
    tools = [dependency_analyzer_tool]
    tool_node = ToolNode(tools)

    # ノードを追加
    workflow.add_node("scan_files", scan_files_node)
    workflow.add_node("detect_framework", detect_framework_node)
    workflow.add_node("classify_layers", classify_layers_node)
    workflow.add_node("analyze_dependencies", analyze_dependencies_node)
    workflow.add_node("generate_design_docs", generate_design_docs_node)
    workflow.add_node("generate_summary", generate_summary_node)
    workflow.add_node("tools", tool_node)

    # エッジを追加（処理の流れを定義）
    workflow.set_entry_point("scan_files")
    workflow.add_edge("scan_files", "detect_framework")
    workflow.add_edge("detect_framework", "classify_layers")
    workflow.add_edge("classify_layers", "analyze_dependencies")
    workflow.add_edge("analyze_dependencies", "generate_design_docs")
    workflow.add_edge("generate_design_docs", "generate_summary")
    workflow.add_edge("generate_summary", END)

    return workflow.compile()

async def main():
    """メイン処理"""
    # parser = argparse.ArgumentParser(description='ソースコード設計書生成ツール（LangGraph + LightRAG版）')
    # parser.add_argument('workspace', help='分析対象のワークスペースパス')
    # parser.add_argument('--api-key', required=True, help='OpenAI APIキー')
    # parser.add_argument('--extensions', nargs='+', help='対象とする拡張子',
    #                    default=['.py', '.java', '.js', '.ts', '.php', '.rb', '.cs', '.html', '.jsx', '.vue'])
    # parser.add_argument('--output', '-o', help='出力ディレクトリ', default='design_documents')

    # args = parser.parse_args()

    workspace = "./read_dir"  # args.workspace
    output_path = "./output"  # args.output
    api_key = os.getenv("OPENAI_API_KEY")or""  # args.api_key
    extensions=['.py', '.java', '.js', '.ts']
    try:
        # LangGraphワークフローを作成
        app = create_workflow()

        # 初期状態を設定
        initial_state: GraphState = {
            "messages": [HumanMessage(content=f"ワークスペース {workspace} の設計書を生成してください")],
            "workspace_path": workspace,
            "source_files": [],
            "framework": FrameworkType.UNKNOWN,
            "dependency_graph": {},
            "processing_order": [],
            "design_documents": [],
            "current_file_index": 0,
            "output_path": output_path,
            "api_key": api_key,
            "extensions": extensions
        }
        print(app.get_graph().draw_mermaid())

        # ワークフローを実行
        async for output in app.astream(initial_state):
            for node_name, node_output in output.items():
                if "messages" in node_output and node_output["messages"]:
                    last_message = node_output["messages"][-1]
                    print(f"[{node_name}] {last_message.content}")

        return 0

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return 1

if __name__ == "__main__":
    asyncio.run(main())
