import json
import os
import re
import asyncio
from typing import List, Dict, Set, TypedDict, Any

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

# ドキュメント内のパスを絶対パスに変換するヘルパー関数
def convert_relative_paths_to_absolute(content):
    """
    ドキュメント内の相対パスを絶対パスに変換する
    例: ./read_dir/file.md → /read_dir/file.md
    """
    # Markdownリンクパターンを検出: [text](path)
    link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    
    def replace_path(match):
        link_text = match.group(1)
        path = match.group(2)
        
        # 既に絶対パスの場合はそのまま
        if path.startswith('/'):
            return match.group(0)
            
        # 相対パスの場合は絶対パスに変換
        if path.startswith('./'):
            # ./で始まる相対パスを処理
            absolute_path = '/' + path[2:]
            return f'[{link_text}]({absolute_path})'
        elif path.startswith('../'):
            # ../で始まる相対パスを処理
            parts = path.split('/')
            up_count = 0
            for part in parts:
                if part == '..':
                    up_count += 1
                else:
                    break
            
            # 上位ディレクトリに移動した後のパス
            remaining_path = '/'.join(parts[up_count:])
            # ワークスペースのルートからの絶対パス
            absolute_path = '/' + remaining_path
            return f'[{link_text}]({absolute_path})'
        else:
            # その他の相対パス (現在のディレクトリからの相対パス)
            absolute_path = '/' + path
            return f'[{link_text}]({absolute_path})'
    
    # リンクパターンを置換
    content = re.sub(link_pattern, replace_path, content)
    
    return content

class GraphState(TypedDict):
    """LangGraphの状態管理"""
    messages: List[AnyMessage]
    workspace_path: str
    source_files: List[SourceFile]
    framework: FrameworkType
    dependency_graph: Dict[str, Set[str]]
    processing_order: List[str]
    design_documents: List[Dict[str, Any]]
    business_flow_documents: List[Dict[str, Any]]  # 業務フロー単位の設計書
    current_file_index: int
    output_path: str
    extensions: List[str]
    entry_points: List[Dict[str, Any]]  # 入力ポイント（View、APIエンドポイントなど）

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

async def detect_entry_points_node(state: GraphState) -> GraphState:
    """入力ポイント（View/APIエンドポイント）検出ノード"""
    print("入力ポイント（エントリーポイント）を検出しています...")

    entry_points = []
    
    # ViewレイヤとControllerレイヤのファイルをフィルタリング
    view_controller_files = [
        file for file in state["source_files"] 
        if file.layer in [LayerType.VIEW, LayerType.CONTROLLER]
    ]
    
    dependency_analyzer_tool = LightRAGQueryTool(config=config)

    # 各ファイルに対してLightRAGを使用してエントリーポイントを検出
    for file in view_controller_files:
        try:
            query = f"""
            ファイル '{file.path}' を分析し、以下を特定してください：
            1. このファイルが外部からのリクエストを受け付けるエントリーポイントかどうか
            2. エントリーポイントである場合、どのような入力を受け付けるか（APIエンドポイント、画面入力など）
            3. このエントリーポイントが処理する業務機能や処理の概要
            
            エントリーポイントである場合のみ、以下の完全なJSONフォーマット（正確な構文）で回答してください：
            {{
                "is_entry_point": true,
                "endpoint": [
                    {{
                        "type": "restful_api",
                        "name": "example",
                        "path": "/api/example"
                    }},
                    {{
                        "type": "screen",
                        "name": "user_input",
                        "path": "/screens/user_input"
                    }}
                ]
            }}
            
            エントリーポイントでない場合は以下のJSONのみを返してください：
            {{
                "is_entry_point": false
            }}
            """
            
            result = await dependency_analyzer_tool._arun(query)
            print(f"ファイル '{file.path}' の分析結果を受信しました")
            
            # JSON部分を抽出するための正規表現
            json_pattern = r'({[\s\S]*})'
            json_match = re.search(json_pattern, result)
            
            if json_match:
                json_str = json_match.group(1)
                try:
                    # JSONを解析
                    parsed_result = json.loads(json_str)
                    print(f"ファイル '{file.path}' のエントリーポイント分析結果: {parsed_result}")
                    result = parsed_result
                except json.JSONDecodeError as json_err:
                    print(f"JSONパースエラー: {file.path} - {json_err}")
                    # エラーが発生した場合はエントリーポイントではないと判断
                    result = {"is_entry_point": False}
            else:
                print(f"JSON形式が見つかりません: {file.path}")
                result = {"is_entry_point": False}

            # エントリーポイントと判断された場合、リストに追加
            if "is_entry_point" in result and result["is_entry_point"]:
                entry_points.append({
                    'file_path': file.path,
                    'layer': file.layer.value,
                    'description': result
                })
        
        except Exception as e:
            print(f"エントリーポイント検出エラー: {file.path} - {e}")
    
    print(f"{len(entry_points)}個のエントリーポイントを検出しました")
    state["entry_points"] = entry_points
    state["messages"].append(AIMessage(content=f"{len(entry_points)}個のエントリーポイントを検出しました"))
    
    return state

async def analyze_dependencies_node(state: GraphState) -> GraphState:
    """依存関係分析ノード（LightRAG使用）"""
    print("依存関係を分析しています...")
    try:
        dependency_graph = {}
        dependency_analyzer_tool = LightRAGQueryTool(config=config)

        for file in state["source_files"]:
            
            # LightRAGを使用して依存関係を分析
            query = f"""
            ファイル '{file.path}' が依存している他のファイルやライブラリを特定してください。
            import文、require文、include文などを分析し、以下の形式で明確に回答してください：
            
            1. ファイルパスの完全リスト：依存しているファイルのパスを完全な形式（相対パスまたは絶対パス）で列挙
            2. ライブラリ依存関係：外部ライブラリやフレームワークの依存関係
            
            回答の最後に「依存ファイルパス：」という見出しの後に、完全なパスのリストだけを改行区切りでリストアップしてください。
            """
            result = await dependency_analyzer_tool._arun(query)

            # LightRAGの出力から依存関係を抽出する高度な実装
            dependencies = set()
            
            # 正規表現パターンでファイルパスを検出
            # [DC], [KG]などの参照タグの後のファイルパスを検出
            file_paths_from_refs = re.findall(r'\[(?:DC|KG|REF)\]\s+([./\w-]+\.\w+)', result)
            
            # 「依存ファイルパス：」セクション以降のパスを検出
            file_paths_from_section = []
            deps_section_match = re.search(r'依存ファイルパス：\s*\n([\s\S]*?)(?:\n\n|\Z)', result)
            if deps_section_match:
                file_paths_from_section = re.findall(r'([./\w-]+\.\w+)', deps_section_match.group(1))
                dependencies.update(file_paths_from_section)
            
            # 明示的なファイルパスパターン
            explicit_paths = re.findall(r'([./][\w/-]+\.\w+)', result)
            
            # すべての検出したパスを統合
            all_possible_paths = set(file_paths_from_refs + explicit_paths + file_paths_from_section)
            
            # ワークスペース内の実ファイルとマッチングして依存関係を特定
            for other_file in state["source_files"]:
                if other_file.path != file.path:
                    # 完全一致
                    if other_file.path in all_possible_paths:
                        dependencies.add(other_file.path)
                    else:
                        # ファイル名のみの一致も確認
                        other_filename = os.path.basename(other_file.path)
                        for possible_path in all_possible_paths:
                            if possible_path.endswith(other_filename):
                                dependencies.add(other_file.path)
                                break
                    
                    # クラス名でのマッチング（Javaなどの場合）
                    # ファイル名から拡張子を除いた部分をクラス名と仮定
                    other_classname = os.path.splitext(os.path.basename(other_file.path))[0]
                    if other_classname in result and len(other_classname) > 3:  # 短すぎる名前は除外
                        dependencies.add(other_file.path)

            dependency_graph[file.path] = dependencies
            file.dependencies = dependencies

        # 処理順序決定（トポロジカルソートの簡易実装）
        processing_order = []
        remaining = set(dependency_graph.keys())
        
        while remaining:
            # 依存されていないファイルを探す
            independent = []
            for file_path in remaining:
                if all(file_path not in deps for deps in dependency_graph.values()):
                    independent.append(file_path)
            
            # 依存されていないファイルがなければ循環参照がある
            if not independent:
                # 循環参照を解決するため、残りのファイルを適当な順序で追加
                processing_order.extend(list(remaining))
                break
            
            # 依存されていないファイルを処理順序に追加
            processing_order.extend(independent)
            
            # 処理済みのファイルを除去
            for file_path in independent:
                remaining.remove(file_path)
                # そのファイルが持つ依存関係も考慮から除外
                dependency_graph[file_path] = set()

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

async def generate_business_flow_docs_node(state: GraphState) -> GraphState:
    """業務フロー単位の設計書生成ノード"""
    print("業務フロー単位の設計書を生成しています...")
    
    generated_docs = []
    client = OpenAI(api_key=config.openai_api_key)
    output_path = state["output_path"]
    os.makedirs(os.path.join(output_path, "business_flows"), exist_ok=True)
    dependency_analyzer_tool = LightRAGQueryTool(config=config)
    
    # 各エントリーポイントごとに設計書を生成
    entry_points = state.get("entry_points", [])
    for i, entry_point in enumerate(entry_points):
        if not isinstance(entry_point, dict):
            continue
            
        file_path = entry_point.get("file_path", "unknown")
        print(f"業務フロー処理中 ({i+1}/{len(entry_points)}): {file_path}")
        
        try:
            # 1. エントリーポイントから呼び出される処理フローを追跡
            entry_file_path = file_path
            if not entry_file_path or entry_file_path == "unknown":
                continue
                
            # コールチェーンを特定するクエリ
            query = f"""
            ファイル '{entry_file_path}' の処理フローを追跡してください。
            以下の情報を特定します：
            
            1. このエントリーポイントが起点となる業務処理の概要
            2. 呼び出し順に従った処理フロー（コールチェーン）
            3. 関連するファイルとそのレイヤ
            4. 処理における主要なデータの流れ
            5. 例外処理パターン
            
            フロー内の各ステップと対応するファイルを特定してください。特に、Controllerからの呼び出し、Service層の処理、
            Repository層へのアクセス、Domain層でのビジネスロジック実行に注目してください。
            
            また、このフローで共有される主要なデータモデルの構造も特定してください。
            """
            
            flow_analysis = await dependency_analyzer_tool._arun(query)
            
            # 2. 詳細な処理フローの解析（関連ファイルの内容も考慮）
            related_files_query = f"""
            '{entry_file_path}' から始まる処理フローに関連するファイルを全て特定し、リストアップしてください。
            これには、直接または間接的に呼び出されるControllerやService、Repository、Domainなどの全てのレイヤが含まれます。
            可能な限り詳細にファイルパスを列挙してください。
            """
            
            related_files_result = await dependency_analyzer_tool._arun(related_files_query)
            
            description = entry_point.get("description", "")
            
            # 3. 業務フロー単位の設計書を生成
            prompt = f"""
            以下の情報を基に、'{entry_file_path}' から始まる業務フロー単位の詳細設計書をMarkdown形式で生成してください。

            ## エントリーポイント情報
            {description}

            ## 処理フロー分析
            {flow_analysis}

            ## 関連ファイル
            {related_files_result}

            ## 出力形式
            以下のセクションを含む設計書を作成してください：

            1. 業務フロー概要
               - 機能名
               - 処理概要
               - ユースケース

            2. 入出力定義
               - 入力データ仕様
               - 出力データ仕様
               - エラーパターン

            3. 処理フロー詳細
               - シーケンス図（Mermaid記法）
               - 各レイヤでの処理内容
               - 主要なビジネスロジック説明

            4. データモデル
               - 使用するエンティティ
               - データの流れ
               - ER図（関連する場合、Mermaid記法）

            5. 例外処理
               - エラーハンドリング
               - エッジケース

            ## 特別な指示
            - クラス図、シーケンス図、ER図はMermaid記法で記述してください
            - 各レイヤでの処理内容を具体的に説明してください
            - コードと設計の対応関係を明確にしてください
            - 日本語で記述してください
            """

            # OpenAI APIで設計書生成
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "あなたは経験豊富なソフトウェアアーキテクトです。ソースコードを分析して業務フロー単位の詳細設計書を作成してください。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,
                temperature=0.2
            )

            doc_content = response.choices[0].message.content
            
            # ドキュメント内の相対パスを絶対パスに変換
            doc_content = convert_relative_paths_to_absolute(doc_content)

            # ファイル名を生成（エントリーポイントのファイル名を基に）
            entry_filename = os.path.basename(entry_file_path)
            safe_filename = re.sub(r'[^\w\-_.]', '_', entry_filename)
            doc_filename = f"flow_{safe_filename}_design.md"
            doc_path = os.path.join(output_path, "business_flows", doc_filename)

            with open(doc_path, 'w', encoding='utf-8') as f:
                f.write(doc_content or "")

            generated_docs.append({
                'entry_point': entry_file_path,
                'design_doc': doc_path,
                'description': description
            })

        except Exception as e:
            print(f"業務フロー設計書生成エラー: {file_path} - {e}")

    state["business_flow_documents"] = generated_docs
    state["messages"].append(AIMessage(content=f"{len(generated_docs)}個の業務フロー設計書を生成しました"))

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
            
            # ドキュメント内の相対パスを絶対パスに変換
            doc_content = convert_relative_paths_to_absolute(doc_content)

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
    business_flow_docs = state.get("business_flow_documents", [])
    dependency_graph = state["dependency_graph"]
    framework = state["framework"]
    output_path = state["output_path"]

    summary_content = f"""# システム設計書サマリー

## 概要
- 分析対象: {state['workspace_path']}
- 検出フレームワーク: {framework.value}
- ファイル単位設計書数: {len(docs)}
- 業務フロー単位設計書数: {len(business_flow_docs)}

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

    # 業務フロー設計書一覧
    if business_flow_docs:
        summary_content += "\n## 業務フロー単位設計書一覧\n\n"
        for doc in business_flow_docs:
            entry_point = doc.get('entry_point', '')
            doc_path = doc.get('design_doc', '')
            if entry_point and doc_path:
                summary_content += f"- [{os.path.basename(doc_path)}]({doc_path}) - エントリーポイント: {os.path.basename(entry_point)}\n"

    # ファイル単位の設計書一覧（レイヤーごとの表形式）
    summary_content += "\n## ファイル単位設計書一覧\n\n"
    
    # レイヤーごとにドキュメントをグループ化
    layer_docs = {}
    for doc in docs:
        layer = doc['layer']
        if layer not in layer_docs:
            layer_docs[layer] = []
        layer_docs[layer].append(doc)
    
    # レイヤーごとに表形式で出力
    for layer, layer_doc_list in sorted(layer_docs.items()):
        summary_content += f"### {layer}層\n\n"
        summary_content += "| ファイル名 | 設計書リンク |\n"
        summary_content += "| -------- | ---------- |\n"
        for doc in layer_doc_list:
            source_file = os.path.basename(doc['source_file'])
            design_doc_name = os.path.basename(doc['design_doc'])
            summary_content += f"| {source_file} | [{design_doc_name}]({doc['design_doc']}) |\n"
        summary_content += "\n"

    # サマリーファイル保存
    summary_path = os.path.join(output_path, "README.md")
    
    # READMEのコンテンツ内のパスも絶対パスに変換
    summary_content = convert_relative_paths_to_absolute(summary_content)
    
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
    workflow.add_node("detect_entry_points", detect_entry_points_node)  # エントリーポイント検出ノード追加
    workflow.add_node("analyze_dependencies", analyze_dependencies_node)
    workflow.add_node("generate_design_docs", generate_design_docs_node)
    workflow.add_node("generate_business_flow_docs", generate_business_flow_docs_node)  # 業務フロー設計書生成ノード追加
    workflow.add_node("generate_summary", generate_summary_node)
    workflow.add_node("tools", tool_node)

    # エッジを追加（処理の流れを定義）
    workflow.set_entry_point("scan_files")
    workflow.add_edge("scan_files", "detect_framework")
    workflow.add_edge("detect_framework", "classify_layers")
    workflow.add_edge("classify_layers", "detect_entry_points")  # レイヤ分類後にエントリーポイント検出
    workflow.add_edge("detect_entry_points", "analyze_dependencies")
    workflow.add_edge("analyze_dependencies", "generate_design_docs")
    workflow.add_edge("generate_design_docs", "generate_business_flow_docs")  # ファイル単位設計書生成後に業務フロー設計書生成
    workflow.add_edge("generate_business_flow_docs", "generate_summary")
    workflow.add_edge("generate_summary", END)

    return workflow.compile()

async def main():

    workspace = "./read_dir"  # args.workspace
    output_path = "./output"  # args.output
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
            "business_flow_documents": [],
            "current_file_index": 0,
            "output_path": output_path,
            "extensions": extensions,
            "entry_points": []
        }

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
