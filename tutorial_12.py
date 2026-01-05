# %% [markdown]
# # 第12回演習：Retrieval Augmented Generation (RAG) の構築
#
# 本演習では，これまでの講義で学んだ大規模言語モデル (LLM) の知識を活かし，
# 講義資料 (`week12_2025.pdf`) の内容に基づいて質問に回答する **RAG (Retrieval Augmented Generation)** システムを構築します．
#
# **本演習の目的:**
# 1. 未学習のデータ（講義資料）を LLM に扱わせる手法としての RAG を理解する．
# 2. PDF からのテキスト抽出，ベクトル化，検索，そして生成という RAG の一連のパイプラインを実装する．
# 3. Google Colab の無料枠 (T4 GPU) で動作する軽量かつ高性能なモデル (`Qwen2.5-1.5B`) を体験する．
#
# ---

# %% [markdown]
# ## 0. 準備: ライブラリのインストールと環境設定
#
# RAG の構築に必要なライブラリをインストールします．
#
# - `pypdf`: PDF ファイルからテキストを抽出するために使用します．
# - `sentence-transformers`: テキストをベクトル化（埋め込み表現に変換）するために使用します．
# - `transformers`, `accelerate`, `bitsandbytes`: LLM をロードし，高速に推論するために使用します．
# - `langchain` (今回は簡易実装のため使いませんが、発展的な実装には便利です)

# %%
!pip install uv
!uv pip install --system -q pypdf sentence-transformers transformers accelerate bitsandbytes

# %% [markdown]
# 必要なライブラリをインポートし，デバイス（GPU）の設定を行います．

# %%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from pypdf import PdfReader
import numpy as np
import textwrap

# デバイスの設定 (GPUが使えるならcuda, 使えなければcpu)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# %% [markdown]
# ## 1. データ読み込み: 講義資料 (PDF) の準備
#
# Google Colab で実行する場合，講義資料 (`week12_2025.pdf`) をアップロードする必要があります．
# 以下のセルを実行して，ファイルをアップロードしてください．

# %%
import os
from google.colab import files

# ファイルが見つからない場合はアップロードを促す
pdf_filename = "week12_2025.pdf"
if not os.path.exists(pdf_filename):
    print(f"'{pdf_filename}' が見つかりません．アップロードしてください．")
    uploaded = files.upload()
    # アップロードされたファイル名を所得（複数可だが今回は1つと仮定）
    for k in uploaded.keys():
        pdf_filename = k
        print(f"アップロード完了: {pdf_filename}")
else:
    print(f"'{pdf_filename}' は既に存在します．")

# %% [markdown]
# ## 2. テキスト抽出
#
# アップロードされたPDFを読み込み，テキストデータを抽出します．
# 機械にとって扱いやすいように，意味のまとまり（チャンク）ごとに分割する処理も重要ですが，
# 今回はシンプルにページごと，あるいは一定の文字数で分割して扱います．

# %%
# PDFファイルのパス (Colab等の環境に合わせて適宜変更してください)
pdf_path = pdf_filename

def load_pdf_text(path):
    try:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return ""

pdf_content = load_pdf_text(pdf_path)
print(f"抽出された文字数: {len(pdf_content)} 文字")
print("--- 先頭500文字プレビュー ---")
print(pdf_content[:500])

# %% [markdown]
# ### テキストのチャンキング (分割)
#
# 長いテキストをそのまま扱うと，埋め込みモデルの入力制限を超えたり，検索精度が落ちたりします．
# ここでは，簡易的に **300文字** 程度のチャンクに分割し，少しずつオーバーラップさせます．

# %%
def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks

chunks = chunk_text(pdf_content)
print(f"チャンク数: {len(chunks)}")
print(f"チャンク例: {chunks[0]}")

# %% [markdown]
# ## 3. 検索 (Retrieval) の準備: 埋め込みモデルのロードとデータベース構築
#
# テキストの意味をベクトル（数値の列）に変換する「埋め込みモデル (Embedding Model)」を使用します．
# 今回は日本語にも対応しており，高性能な `intfloat/multilingual-e5-large` を使用します．
#
# **注意:** `e5` モデルは，検索対象の文章には `passage: `，クエリには `query: ` という接頭辞（プレフィックス）をつけることが推奨されています．

# %%
# 埋め込みモデルのロード
embed_model_name = "intfloat/multilingual-e5-large"
print(f"Loading embedding model: {embed_model_name}...")
embedder = SentenceTransformer(embed_model_name, device=device)

# ドキュメント（チャンク）のベクトル化
# prefix "passage: " を付与して埋め込む
passage_chunks = ["passage: " + c for c in chunks]
corpus_embeddings = embedder.encode(passage_chunks, convert_to_tensor=True)

print("Vector database creation complete.")
print(f"Embedding shape: {corpus_embeddings.shape}")

# %% [markdown]
# ## 4. 生成 (Generation) の準備: LLM のロード
#
# 検索した情報を元に回答を生成する LLM をロードします．
# Google Colab の無料枠でも動作し，かつ日本語性能が高い **Qwen2.5-1.5B-Instruct** を使用します．
# これは 15億パラメータという小規模なモデルですが，非常に高い性能を持っています．
#
# ※ Hugging Face のアカウント認証は不要なモデルです．

# %%
# LLMのロード
llm_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
print(f"Loading LLM: {llm_model_name}...")

tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
model = AutoModelForCausalLM.from_pretrained(
    llm_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# %% [markdown]
# ## 5. RAG システムの実装
#
# これまでの要素を組み合わせて，RAG の関数を作成します．
#
# **処理の流れ:**
# 1. ユーザーの質問 (Query) を受け取る．
# 2. 質問をベクトル化する (`query: ` を付与)．
# 3. 事前に作成したデータベースから，質問に類似したチャンク（講義資料の一部）を検索する．
# 4. 検索されたチャンクを「参考情報」としてプロンプトに組み込む．
# 5. LLM にプロンプトを渡し，回答を生成させる．

# %%
def retrieve_relevant_chunks(query, k=3):
    """
    クエリに関連するチャンクを上位k個検索する関数
    """
    query_embedding = embedder.encode(f"query: {query}", convert_to_tensor=True)

    # コサイン類似度を計算
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]

    # 上位k個のインデックスを取得
    top_results = torch.topk(cos_scores, k=k)

    relevant_chunks = []
    for score, idx in zip(top_results.values, top_results.indices):
        # チャンクのリストから元のテキストを取得 (passage: を除去して表示)
        original_text = chunks[idx]
        relevant_chunks.append(original_text)

    return relevant_chunks

def generate_rag_answer(query):
    """
    RAG を用いて回答を生成する関数
    """
    # 1. 検索
    relevant_contexts = retrieve_relevant_chunks(query)
    context_str = "\n\n".join(relevant_contexts)

    # 2. プロンプト作成
    # 文脈を与えて回答させるためのテンプレート
    prompt_template = f"""以下は講義資料からの抜粋です。この内容に基づいて、ユーザーの質問に日本語で答えてください。
資料に書かれていないことは「資料には記載がありません」と答えてください。

[参考資料]
{context_str}

[質問]
{query}

[回答]
"""

    # チャット形式のフォーマットに変換
    messages = [
        {"role": "system", "content": "あなたは親切なAIアシスタントです。与えられた参考資料に基づいて、正確に質問に答えてください。"},
        {"role": "user", "content": prompt_template}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 3. 生成
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response, relevant_contexts

def generate_no_rag_answer(query):
    """
    RAG を使用せずに（文脈なしで）回答を生成する関数
    """
    messages = [
        {"role": "system", "content": "あなたは親切なAIアシスタントです。ユーザーの質問に日本語で答えてください。"},
        {"role": "user", "content": query}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# %% [markdown]
# ## 6. 実行と確認
#
# 実際に講義内容について質問してみましょう．
# ここでは，PDFの内容（と想定されるもの）について質問を投げかけます．

# %%
# 質問の例
questions = [
    "今回の講義の主要なテーマは何ですか？",
    "大規模言語モデルの課題について、資料ではどのように述べられていますか？",
    "次回の演習内容は決まっていますか？",
    "LLMのマルチモーダル化とは具体的にどのようなことですか？",
    "AIエージェントが研究もできるとはどういう意味ですか？",
    "高度なLLMの活用方法として、どのような事例が紹介されていますか？"
]

print("=== RAG vs No-RAG 比較デモ開始 ===")

for q in questions:
    print(f"\nQ: {q}")
    print("=" * 60)

    # RAGなし
    print("[No-RAG] 回答生成中...")
    no_rag_answer = generate_no_rag_answer(q)
    print("A (No-RAG):")
    print(textwrap.fill(no_rag_answer, width=80))
    print("-" * 30)

    # RAGあり
    print("[RAG] 回答生成中...")
    rag_answer, contexts = generate_rag_answer(q)
    print("A (RAG):")
    print(textwrap.fill(rag_answer, width=80))

    print("\n[参照した情報の抜粋]")
    for i, ctx in enumerate(contexts):
        print(f"({i+1}) {ctx[:50]}...") # 長いので先頭だけ表示

# %% [markdown]
# ## 7. 発展課題（オプション）
#
# 余裕がある人は，以下の改良に挑戦してみてください．
#
# 1. **チャンク分割の工夫**: `RecursiveCharacterTextSplitter` (LangChain) などを使って，文章の区切りを意識した分割にする．
# 2. **検索精度の向上**: 検索結果を LLM に渡す前に，本当にその文書が質問に関係あるかを判定する (Reranking)．
# 3. **マルチモーダル化**: PDF に含まれる図表を画像として認識し，画像の内容も踏まえて回答する (GPT-4o系やGemini系などのAPIが必要，もしくはVLMを使用)．

# %% [markdown]
# ## 8. 演習のまとめ
#
# 本演習では，**RAG (Retrieval Augmented Generation)** の基本的な仕組みと実装方法を学びました．
#
# 1. **RAGの有効性**:
#    - 大規模言語モデル (LLM) は強力ですが，学習していない最新情報や専門知識（今回の講義資料など）については正確に答えられません．
#    - 外部知識を検索してプロンプトに加えることで，**ハルシネーション（もっともらしいウソ）** を抑制し，正確な回答を生成できるようになることを確認しました．
#
# 2. **RAGの構成要素**:
#    - **ドキュメントローダー**: PDFなどの非構造化データからテキストを抽出します．
#    - **エンべディング（埋め込み）**: テキストをベクトル化し，意味的な検索を可能にします．
#    - **ベクトルデータベース**: ベクトル化したデータを保持し，高速に検索します（今回は簡易的にメモリ上で処理）．
#    - **生成モデル (LLM)**: 検索結果を元に自然な回答を生成します．
#
# 今回構築したシステムは簡易的なものですが，企業内のドキュメント検索や，特定ドメインのチャットボットなど，実社会で広く応用されている技術の基礎となります．
