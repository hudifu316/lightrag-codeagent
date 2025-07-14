# lightrag-codeagent

lightragを使ってソースコードおよびドキュメントの依存関係を分析し、設計情報をMarkdownで出力するエージェントです。

### 💣️注意
**勉強がてら実装しているのでうまく動かないかもしれません。コードは参考セクションにある記事等を参考にして一部生成AI（GPT-4.1、Claude Sonnet 4）を使って自動生成しています**

## 🚀Usage

1. 必要なライブラリをインストール
    ```bash
    pip install -r requirements.txt
    ```
1. `.env`ファイルを作成し、以下の環境変数を設定
    ```env
    OPENAI_API_KEY=your_openai_api_key
    ```
1. 解析したプロジェクトを`read_dir`配下にコピー
1. indexerを動かしてRAGインスタンスを作成
    ```bash
    python main.py
    ```
1. ragagentを動かして設計情報を取得
    ```bash
    python ragagent.py
    ```

## 参考
https://github.com/HKUDS/LightRAG
https://zenn.dev/yumefuku/articles/llm-code-graphrag