import os
import asyncio
import textract
import numpy as np
from typing import Optional
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from pydantic import Field
from tree_sitter import Node, Parser, Language
import tree_sitter_python as tspython
import tree_sitter_cpp as tscpp
import tree_sitter_java as tsjava
from ragindex.config import LightRAGConfig
from ragindex.openai_client import openai_complete_func
from ragindex.embedding import embedding_func_factory, EmbeddingModelManager
from ragindex.utils import get_node_line_range
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

doc_ext_dict = {"text_file": ["txt", "md"], "binary_file": ["pdf", "csv", "doc"]}
max_depth = 30

python_definition_dict = {
    "class_definition": "identifier",
    "function_definition": "identifier",
}
py_lang = Language(tspython.language())

cpp_definition_dict = {
    "class_specifier": "name",
    "struct_specifier": "name",
    "function_declarator": "identifier",
}
cpp_lang = Language(tscpp.language())

java_definition_dict = {
    "class_declaration": "identifier",
    "method_declaration": "identifier",
    "interface_declaration": "identifier",
}
java_lang = Language(tsjava.language())

code_ext_dict = {
    "py": {"definition": python_definition_dict, "language": py_lang},
    "cpp": {"definition": cpp_definition_dict, "language": cpp_lang},
    "h": {"definition": cpp_definition_dict, "language": cpp_lang},
    "java": {"definition": java_definition_dict, "language": java_lang},
}


class DocumentIndexTool(BaseTool):
    """ドキュメントとコードをインデックス化するツール"""

    name: str = "document_index"
    description: str = """
    ドキュメントとコードファイルをインデックス化して知識グラフを構築します。
    入力: ディレクトリパス（文字列）
    出力: インデックス化の結果
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
        api_key = self.config.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API キーが設定されていません。config.openai_api_key または環境変数 OPENAI_API_KEY を設定してください。"
            )

    def _run(
        self,
        directory_path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        return asyncio.run(self._arun(directory_path, run_manager))

    async def _arun(
        self,
        directory_path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            # LightRAGの初期化
            rag = await self._initialize_rag()
            # 指定したディレクトリからドキュメントファイルとコードファイルを抽出
            doc_dict, code_dict = self._read_dir(directory_path)
            # ドキュメントのチャンク化、グラフ化処理を実行
            await self._doc_insert(rag, doc_dict)
            # コードのチャンク化、グラフ化処理を実行
            all_entitiy_name_list = await self._code_insert(rag, code_dict)
            # ドキュメントとコードのエンティティをマージ
            await self._merge_doc_and_code(rag, doc_dict, all_entitiy_name_list)
            # ストレージの終了処理
            await rag.finalize_storages()
            return f"インデックス化完了: ドキュメント {len(doc_dict)} 件, コードファイル {len(code_dict)} 件"
        except Exception as e:
            import traceback

            error_detail = traceback.format_exc()
            print(f"詳細なエラー情報:\n{error_detail}")
            return f"インデックス化エラー: {str(e)}\n詳細なエラー情報:\n{error_detail}"

    async def _initialize_rag(self) -> LightRAG:
        api_key = self.config.openai_api_key
        base_url = self.config.openai_base_url
        assert api_key is not None, "OpenAI API キーが設定されていません。"

        def llm_func(prompt: str, system_prompt: str = "", **kwargs):
            return openai_complete_func(
                prompt,
                system_prompt,
                model_name=self.config.model_name,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )

        embedding_func = embedding_func_factory(self.config.embedding_model)
        rag = LightRAG(
            working_dir=self.config.storage_dir,
            max_parallel_insert=self.config.parallel_num,
            llm_model_func=llm_func,
            max_total_tokens=self.config.llm_max_token_size,
            embedding_func=EmbeddingFunc(
                func=embedding_func,
                max_token_size=self.config.emb_max_token_size,
                embedding_dim=self.config.embedding_dim,
            ),
            llm_model_max_async=self.config.parallel_num,
            embedding_func_max_async=self.config.parallel_num,
        )
        await rag.initialize_storages()
        await initialize_pipeline_status()
        return rag

    def _read_dir(self, read_dir_path):
        print("=" * 50)
        print("処理予定ファイル")
        doc_dict = {}
        code_dict = {}
        allow_ext_set = set(sum(doc_ext_dict.values(), [])) | set(code_ext_dict.keys())
        for dir_path, _, file_name_list in os.walk(read_dir_path):
            for file_name in file_name_list:
                _, ext = os.path.splitext(file_name)
                if ext.lstrip(".") not in allow_ext_set:
                    continue
                file_path = os.path.join(dir_path, file_name)
                if ext.lstrip(".") in code_ext_dict:
                    with open(file_path, "rb") as file:
                        code_dict[file_path] = file.read()
                    print(f"コードファイル: {file_path}")
                elif ext.lstrip(".") in doc_ext_dict["text_file"]:
                    with open(file_path, "r", encoding="utf-8") as file:
                        doc_dict[file_path] = file.read()
                    print(f"テキストファイル: {file_path}")
                elif ext.lstrip(".") in doc_ext_dict["binary_file"]:
                    doc_dict[file_path] = textract.process(file_path)
                    print(f"バイナリファイル: {file_path}")
        print("=" * 50 + "\n")
        return doc_dict, code_dict

    async def _doc_insert(self, rag: LightRAG, doc_dict: dict):
        print("=" * 50)
        print("ドキュメントファイルのグラフ化")
        # ドキュメントのチャンク化、グラフ化処理
        await rag.ainsert(
            list(doc_dict.values()),
            file_paths=list(doc_dict.keys())
        )
        for doc_path in doc_dict.keys():
            print(f"処理完了：{doc_path}")
        print("=" * 50 + "\n")

    async def _code_insert(self, rag: LightRAG, code_dict: dict):
        print("=" * 50)
        print("コードファイルのグラフ化")

        async def process_file(code_path, file_content_bytes):
            file_name = os.path.basename(code_path)
            _, ext = os.path.splitext(file_name)
            language = code_ext_dict[ext.lstrip(".")]["language"]
            parser = Parser(language)
            tree = parser.parse(file_content_bytes)
            root_node = tree.root_node
            file_content_text = file_content_bytes.decode("utf-8")
            line_offset = 0
            line_offset_list = []
            line_list = file_content_text.splitlines()
            for line in line_list:
                line_offset_list.append(line_offset)
                line_offset += len(line.encode("utf-8")) + 1
            chunks = []
            entities = []
            relationships = []
            chunk_node_list = await self._create_chunks(root_node, file_content_bytes)
            definition_dict = code_ext_dict[ext.lstrip(".")]["definition"]
            for node, node_text in chunk_node_list:
                start_line, end_line = get_node_line_range(node, line_offset_list)
                source_id = f"file:{file_name}_line:{start_line}-{end_line}"
                chunks.append({"content": node_text, "source_id": source_id})
                chunk_entities, chunk_relationships = await self._create_graph(
                    node=node,
                    definition_dict=definition_dict,
                    file_content_bytes=file_content_bytes,
                    parent_definition_name="",
                    source_id=source_id,
                    file_name=file_name,
                    line_offset_list=line_offset_list,
                )
                entities += chunk_entities
                relationships += chunk_relationships
            await rag.ainsert_custom_kg(
                custom_kg={
                    "chunks": chunks,
                    "entities": entities,
                    "relationships": relationships,
                }
            )
            print(f"処理完了：{code_path}")
            return [entity["entity_name"] for entity in entities]

        all_entity_name_list = []
        file_item_list = list(code_dict.items())
        batch_size = self.config.parallel_num
        for batch_index in range(0, len(file_item_list), batch_size):
            batch_item_list = file_item_list[batch_index : batch_index + batch_size]
            batch_task_list = []
            for code_path, file_content_bytes in batch_item_list:
                task = asyncio.create_task(process_file(code_path, file_content_bytes))
                batch_task_list.append(task)
            batch_result_list = await asyncio.gather(*batch_task_list)
            for entity_name_list in batch_result_list:
                all_entity_name_list.extend(entity_name_list)
        print("=" * 50 + "\n")
        return all_entity_name_list

    async def _create_chunks(self, root_node: Node, file_content_bytes: bytes):
        # トークナイザーを取得
        _, tokenizer = EmbeddingModelManager.get_model_and_tokenizer(
            self.config.embedding_model
        )
        # ノード格納用キューの初期化
        task_queue = asyncio.Queue()
        # ルートノード直下のノードをキューに追加
        for child_node in root_node.children:
            await task_queue.put(child_node)

        chunk_node_list = []
        while not task_queue.empty():
            # キューからノードを取得
            current_node = await task_queue.get()
            # ノードに対応するコード部分を取得
            node_text = (
                file_content_bytes[current_node.start_byte : current_node.end_byte]
                .decode("utf-8")
                .strip()
            )
            if not node_text:
                continue
            tokens = await asyncio.to_thread(tokenizer.encode, node_text)
            # トークン数が設定値以下の場合、チャンクとして追加
            if self.config.chunk_max_tokens >= len(tokens):
                chunk_node_list.append((current_node, node_text))
            # トークン数が設定値を超える場合、子ノードをキューに追加
            else:
                for child_node in current_node.children:
                    await task_queue.put(child_node)
        return chunk_node_list

    async def _create_graph(
        self,
        node: Node,
        definition_dict: dict,
        file_content_bytes: bytes,
        parent_definition_name: str,
        source_id: str,
        file_name: str,
        line_offset_list: list,
    ):
        entities = []
        relationships = []
        task_queue = asyncio.Queue()
        await task_queue.put((node, parent_definition_name, 0))
        while not task_queue.empty():
            current_node, parent_definition_name, depth = await task_queue.get()
            node_text = (
                file_content_bytes[current_node.start_byte : current_node.end_byte]
                .decode("utf-8")
                .strip()
            )
            if not node_text:
                continue
            definition_name = parent_definition_name
            if current_node.type in definition_dict:
                start_line, end_line = get_node_line_range(
                    current_node, line_offset_list
                )
                search_queue = asyncio.Queue()
                for child in current_node.children:
                    await search_queue.put(child)
                entity_name = ""
                while not search_queue.empty():
                    search_node = await search_queue.get()
                    if search_node.type == definition_dict[current_node.type]:
                        definition_name = (
                            file_content_bytes[
                                search_node.start_byte : search_node.end_byte
                            ]
                            .decode("utf-8")
                            .strip()
                        )
                        entity_name = f"{file_name}:{definition_name}"
                        break
                    else:
                        for child in search_node.children:
                            await search_queue.put(child)
                if entity_name:
                    api_key = self.config.openai_api_key
                    base_url = self.config.openai_base_url
                    assert api_key is not None
                    assert base_url is not None
                    prompt = f"""# Instructions\n    Extract the important elements and processes from the program and create a brief summary statement described in natural language.\n\n    # Rules\n    - Create a summary statement using natural language, not the program.\n    - Output only a pure summary without any supplements or questions.\n\n    # Program\n    {node_text}\n\n    # Summary statement"""
                    description = await openai_complete_func(
                        prompt=prompt,
                        max_tokens=self.config.llm_max_token_size,
                        api_key=api_key,
                        base_url=base_url,
                    )
                    entities.append(
                        {
                            "entity_name": entity_name,
                            "entity_type": current_node.type,
                            "description": description,
                            "source_id": source_id,
                        }
                    )
                    if parent_definition_name:
                        relationships.append(
                            {
                                "src_id": f"{file_name}:{parent_definition_name}",
                                "tgt_id": entity_name,
                                "description": f"The {definition_name} of {parent_definition_name} located in lines {start_line} through {end_line}.",
                                "keywords": f"{parent_definition_name} {definition_name}",
                                "weight": 1.0,
                                "source_id": source_id,
                            }
                        )
            if depth < max_depth:
                for child_node in current_node.children:
                    await task_queue.put((child_node, definition_name, depth + 1))
        return entities, relationships

    async def _merge_doc_and_code(
        self, rag: LightRAG, doc_dict: dict, entity_name_list: list
    ):
        print("=" * 50)
        print("エンティティのマージ")
        all_entity_name = await rag.get_graph_labels()
        ast_entity_list = []
        doc_entity_list = []
        for entity_name in all_entity_name:
            entity = await rag.chunk_entity_relation_graph.get_node(entity_name)
            assert entity is not None, f"Entity {entity_name} not found"
            entity_file_path = entity.get("file_path")
            if entity_file_path:
                file_path_list = entity_file_path.split("<SEP>")
                if any(file_path in doc_dict for file_path in file_path_list):
                    doc_entity_list.append(
                        (entity.get("entity_id"), entity.get("description"))
                    )
            elif entity.get("entity_id") in entity_name_list:
                ast_entity_list.append(
                    (entity.get("entity_id"), entity.get("description"))
                )
        if not ast_entity_list:
            print("コードのエンティティが作成されていません")
        if not doc_entity_list:
            print("ドキュメントのエンティティが作成されていません")
        if not ast_entity_list or not doc_entity_list:
            print("マージをスキップ")
            print("=" * 50 + "\n")
            return
        embedding_func = rag.embedding_func
        assert isinstance(embedding_func, EmbeddingFunc), (
            "Embedding function is not set"
        )
        ast_name_embedding_array = await embedding_func(
            [ast_name.split(":", 1)[1] for ast_name, _ in ast_entity_list]
        )
        doc_name_embedding_array = await embedding_func(
            [doc_name for doc_name, _ in doc_entity_list]
        )
        ast_list_locks = asyncio.Lock()

        async def _process_doc_entity(doc_index, doc_name, doc_description):
            doc_name_embedding = doc_name_embedding_array[doc_index]
            extract_ast_list = []
            for ast_index, (ast_name, ast_description) in enumerate(ast_entity_list):
                ast_name_embedding = ast_name_embedding_array[ast_index]
                similarity = np.dot(ast_name_embedding, doc_name_embedding) / (
                    np.linalg.norm(ast_name_embedding)
                    * np.linalg.norm(doc_name_embedding)
                )
                if similarity >= self.config.merge_score_threshold:
                    extract_ast_list.append((ast_name, ast_description))
            if extract_ast_list:
                exist_ast_list = []
                async with ast_list_locks:
                    for ast_name, ast_description in extract_ast_list:
                        exist = await rag.chunk_entity_relation_graph.has_node(ast_name)
                        if exist:
                            exist_ast_list.append((ast_name, ast_description))
                    if exist_ast_list:
                        merge_description = ""
                        print("-" * 50)
                        print("マージ対象のエンティティ")
                        print(doc_name)
                        for exist_ast_index, (
                            exist_ast_name,
                            exist_ast_description,
                        ) in enumerate(exist_ast_list, 1):
                            merge_description += (
                                f"<SEP>{exist_ast_name}\n{exist_ast_description}"
                            )
                            print(f"{exist_ast_index}: {exist_ast_name}")
                        print("-" * 50 + "\n")
                        await rag.amerge_entities(
                            source_entities=[
                                doc_name,
                                *[
                                    exist_ast_name
                                    for exist_ast_name, _ in exist_ast_list
                                ],
                            ],
                            target_entity=doc_name,
                            target_entity_data={
                                "description": f"{doc_description}{merge_description}"
                            },
                        )

        batch_size = self.config.parallel_num
        for batch_index in range(0, len(doc_entity_list), batch_size):
            batch_doc_entity_list = doc_entity_list[
                batch_index : batch_index + batch_size
            ]
            batch_task_list = []
            for i, (doc_name, doc_description) in enumerate(batch_doc_entity_list):
                doc_index = batch_index + i
                task = asyncio.create_task(
                    _process_doc_entity(doc_index, doc_name, doc_description)
                )
                batch_task_list.append(task)
            await asyncio.gather(*batch_task_list)
        print("=" * 50 + "\n")
