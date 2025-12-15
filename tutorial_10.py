"""
MDA入門 2025年度 第10回演習（修正版）
大規模言語モデル (LLM) とプロンプティング

演習内容:
1. プロンプティングの実践（Zero-shot / Few-shot）
2. In-Context Learningの体験
3. Chain-of-Thought (CoT) Promptingの実験

使用モデル（優先順位順）:
1. Qwen/Qwen2.5-0.5B-Instruct（最優先）
2. llm-jp/llm-jp-3-1.8b-instruct
3. rinna/japanese-gpt2-medium
4. cyberagent/open-calm-small（フォールバック）

使用データセット:
- JGLUE/MARC-ja（日本語商品レビュー）
- tyqiangz/multilingual-sentiments（多言語感情分析）
"""

# ============================================
# 環境セットアップ
# ============================================

print("=" * 70)
print("環境セットアップ中...")
print("=" * 70)

# 必要なライブラリのインストール
#!pip install -q transformers torch datasets accelerate sentencepiece

import gc
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

warnings.filterwarnings("ignore")

# 日本語フォントの設定
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# シード固定（再現性のため）
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_prediction(prompt, generator, tokenizer):
    try:
        output = generator(
            prompt,
            max_new_tokens=5, # 答えだけ欲しいので短くて良い
            num_return_sequences=1,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id,
            truncation=True
        )
        # プロンプト以降のテキストを取得
        generated = output[0]["generated_text"][len(prompt):].strip()
        # 最初の改行や空白までを取得
        return generated.split('\n')[0].strip()
    except Exception as e:
        return "Error"


def build_prompt(examples, target_text, randomize_labels=False):
    prompt = "以下のテキストの感情を「ポジティブ」「ネガティブ」「中立」で分類してください。\n\n"

    # ラベルの候補
    labels = ["ポジティブ", "ネガティブ", "中立"]

    for i, ex in enumerate(examples):
        # テキストとラベルの取得（データ構造に対応）
        if isinstance(ex, dict):
             txt = ex.get("text", ex.get("review", ex.get("sentence", "")))
             lbl = ex.get("label", ex.get("sentiment", ""))
        else:
             txt = str(list(ex.values())[0])
             lbl = "unknown"

        txt = str(txt)[:80] # 長すぎるとコンテキスト長圧迫するのでカット

        if randomize_labels:
            # ラベルをランダムに選択
            lbl = random.choice(labels)

        prompt += f"例{i+1}:\nテキスト: {txt}...\n感情: {lbl}\n\n"

    prompt += f"テキスト: {target_text}\n感情:"
    return prompt


def main():
    set_seed(42)

    print("\n✓ セットアップ完了")
    print("=" * 70)
    print("MDA入門 第10回演習: 大規模言語モデルとプロンプティング")
    print("=" * 70)

    # ============================================
    # 公開LLMモデルの読み込み（複数モデルを順次試行）
    # ============================================

    print("\n" + "=" * 70)
    print("公開LLMモデルの読み込み")
    print("=" * 70)

    # 試行するモデルのリスト（優先順位順）
    model_candidates = [
        {
            "name": "Qwen/Qwen2.5-0.5B-Instruct",
            "params": "500M",
            "description": "Alibaba開発、多言語対応、Instructionチューニング済み",
        },
        {
            "name": "llm-jp/llm-jp-3-1.8b-instruct",
            "params": "1.8B",
            "description": "国立情報学研究所開発、日本語特化",
        },
        {
            "name": "rinna/japanese-gpt2-medium",
            "params": "330M",
            "description": "日本語GPT-2、実績豊富",
        },
        {
            "name": "cyberagent/open-calm-small",
            "params": "160M",
            "description": "サイバーエージェント開発、超軽量",
        },
    ]

    # デバイスの確認
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n使用デバイス: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPUメモリ: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )

    # モデルのロード試行
    model = None
    tokenizer = None
    model_name = None
    loaded_model_info = None

    print("\n" + "-" * 70)
    print("モデルのロード試行（優先順位順）")
    print("-" * 70)

    for i, candidate in enumerate(model_candidates, 1):
        model_name = candidate["name"]
        print(f"\n【試行 {i}/4】{model_name}")
        print(f"  パラメータ数: {candidate['params']}")
        print(f"  特徴: {candidate['description']}")
        print("  読み込み中...", end="")

        try:
            # トークナイザーの読み込み
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # モデルの読み込み（メモリ効率化）
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto",
                low_cpu_mem_usage=True,
            )

            # パラメータ数の確認
            num_params = sum(p.numel() for p in model.parameters()) / 1e6

            print(" ✓ 成功！")
            print(f"  実際のパラメータ数: {num_params:.0f}M")

            loaded_model_info = candidate
            break  # 成功したらループを抜ける

        except Exception as e:
            print(" ✗ 失敗")
            print(f"  エラー: {str(e)[:100]}...")

            # メモリをクリア
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

            model = None
            tokenizer = None

            if i < len(model_candidates):
                print("  → 次の候補モデルを試します...")
            else:
                print("  → すべてのモデルのロードに失敗しました")

    # ロード結果の確認
    if model is None or tokenizer is None:
        raise RuntimeError(
            "❌ すべてのモデルのロードに失敗しました。Google Colabの設定を確認してください。"
        )

    print("\n" + "=" * 70)
    print(f"✓ 使用モデル: {model_name}")
    print(f"  パラメータ数: {loaded_model_info['params']}")
    print(f"  特徴: {loaded_model_info['description']}")
    print("=" * 70)

    # テキスト生成用のパイプライン作成
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )  # device=0 if device == "cuda" else -1,

    print("\n✓ テキスト生成パイプラインの準備完了")

    # ============================================
    # 公開データセットの読み込み
    # ============================================

    print("\n" + "=" * 70)
    print("公開データセットの読み込み")
    print("=" * 70)

    # データセット候補（優先順位順）
    dataset_candidates = [
        {
            "name": "tyqiangz/multilingual-sentiments",
            "config": "japanese",
            "description": "多言語感情分析データセット（日本語）",
        },
        {
            "name": "shunk031/JGLUE",
            "config": "MARC-ja",
            "description": "日本語商品レビュー（Amazon）",
        },
    ]

    dataset = None
    dataset_info = None

    # サンプルデータ
    if dataset is None:
        print("代替として手動サンプルデータを使用します\n")

        # 代替サンプルデータ（日本語感情分析）
        dataset = [
            {"text": "この商品は最高です！買って良かったです。", "label": "positive"},
            {"text": "期待外れでした。品質が悪すぎます。", "label": "negative"},
            {"text": "値段の割には普通です。可もなく不可もなく。", "label": "neutral"},
            {
                "text": "素晴らしい製品です。友人にも勧めたいと思います。",
                "label": "positive",
            },
            {"text": "サービスが最悪でした。二度と利用しません。", "label": "negative"},
            {"text": "機能は良いですが、デザインがイマイチです。", "label": "neutral"},
            {"text": "期待以上の性能でした。大満足です！", "label": "positive"},
            {"text": "説明と実物が全然違う。詐欺レベルです。", "label": "negative"},
            {"text": "この価格なら妥当だと思います。", "label": "neutral"},
            {"text": "感動しました。本当に買って良かったです。", "label": "positive"},
        ]

        dataset_info = {
            "name": "手動サンプルデータ",
            "description": "日本語商品レビュー感情分析",
        }

        print(f"✓ サンプルデータ作成完了（{len(dataset)}件）")

    else:
        print(f"\n✓ データセット読み込み完了: {dataset_info['name']}")

        # データセットのサンプル表示
        print("\n【サンプルデータ】")
        sample = dataset[0] if isinstance(dataset, list) else dataset[0]
        for key, value in sample.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"  {key}: {value[:100]}...")
            else:
                print(f"  {key}: {value}")

    # ============================================
    # 演習1: プロンプティングの基礎実践
    # ============================================

    print("\n\n" + "=" * 70)
    print("演習1: プロンプティングの基礎実践")
    print("=" * 70)

    """
    この演習では、プロンプトの設計が出力にどう影響するかを学びます。
    - Zero-shot prompting（例題なし）
    - Few-shot prompting（例題あり）
    の違いを実際に確認します。
    """

    # --- タスク1: テキストの感情分析 ---
    print("\n" + "-" * 70)
    print("タスク1: テキストの感情分析")
    print("-" * 70)

    # テスト用のテキストを選択
    if isinstance(dataset, list):
        test_text = dataset[0]["text"]
        test_label = dataset[0]["label"]
    else:
        test_idx = 50
        sample = dataset[test_idx]
        # データセットの構造に応じてテキストとラベルを取得
        if "text" in sample:
            test_text = sample["text"]
        elif "sentence" in sample:
            test_text = sample["sentence"]
        elif "review" in sample:
            test_text = sample["review"]
        else:
            test_text = str(list(sample.values())[0])

        if "label" in sample:
            test_label = sample["label"]
        elif "sentiment" in sample:
            test_label = sample["sentiment"]
        else:
            test_label = "unknown"

    print("\n【テストテキスト】")
    print(f"テキスト: {test_text}")
    print(f"正解ラベル: {test_label}")

    # Zero-shot prompting
    print("\n【実験1】Zero-shot Prompting")
    print("-" * 50)

    zero_shot_prompt = f"""以下のテキストの感情を「ポジティブ」「ネガティブ」「中立」のいずれかで分類してください。

テキスト: {test_text}

感情:"""

    print("プロンプト:")
    print(zero_shot_prompt[:150] + "...")

    # テキスト生成
    try:
        output_zero = generator(
            zero_shot_prompt,
            max_new_tokens=20,
            num_return_sequences=1,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
            if tokenizer.eos_token_id
            else tokenizer.pad_token_id,
            truncation=True,
        )

        generated_text = output_zero[0]["generated_text"]
        summary_zero = generated_text[len(zero_shot_prompt) :].strip()

        print("\n生成された分類結果:")
        print(f"{summary_zero[:50]}")

    except Exception as e:
        print(f"⚠️ 生成エラー: {e}")
        summary_zero = "（生成失敗）"

    # 結果表示
    print("\n" + "=" * 70)
    print("【Zero-shot 結果】")
    print("=" * 70)
    print(f"正解:       {test_label}")
    print(f"予測:       {summary_zero[:30]}")

    # ============================================
    # 演習2: In-Context Learning (ICL) の実験
    # ============================================

    print("\n\n" + "=" * 70)
    print("演習2: In-Context Learning (ICL) の実験")
    print("=" * 70)

    print("""
Few-shot ICLにおける以下の要素の影響を実験・比較します：
1. 標準的なFew-shot（正解ラベル、固定順序）
2. ラベルのランダム化（事例のラベルをランダムに入れ替え）
3. 事例順序の変更（事例の提示順をシャッフル）
""")

    # 評価用データの準備
    n_shots = 3
    if isinstance(dataset, list):
        # 手動データの場合
        icl_examples = dataset[:n_shots]      # 例示用
        icl_test_data = dataset[n_shots:n_shots+5]    # テスト用（最大5件）
    else:
        # HFデータセットの場合
        icl_examples = [dataset[i] for i in range(n_shots)]
        icl_test_data = [dataset[i] for i in range(n_shots, n_shots+5)]

    # --- 実験設定 ---
    conditions = [
        {"name": "標準Few-shot", "random_label": False, "shuffle_order": False},
        {"name": "ラベルランダム", "random_label": True, "shuffle_order": False},
        {"name": "順序変更",     "random_label": False, "shuffle_order": True},
    ]

    # 実験ループ
    results = {c["name"]: [] for c in conditions}

    print(f"\n評価データ数: {len(icl_test_data)}件 (各条件で推論を実行)")
    print("-" * 60)

    for idx, sample in enumerate(icl_test_data):
        # テストテキストの取得
        if isinstance(sample, dict):
            test_text = sample.get("text", sample.get("review", sample.get("sentence", "")))
            true_label = sample.get("label", sample.get("sentiment", ""))
        else:
            test_text = str(list(sample.values())[0])
            true_label = "unknown"

        print(f"\nテストケース {idx+1}: {test_text[:30]}... (正解: {true_label})")

        for cond in conditions:
            current_examples = list(icl_examples) # コピー

            # 順序変更
            if cond["shuffle_order"]:
                random.shuffle(current_examples)

            # プロンプト作成
            prompt = build_prompt(
                current_examples,
                test_text,
                randomize_labels=cond["random_label"]
            )

            # 推論
            pred = generate_prediction(prompt, generator, tokenizer)

            # 結果保存
            # 簡易正解判定（文字列として正解ラベルが含まれているか）
            is_correct = False
            if isinstance(true_label, str) and true_label in pred:
                is_correct = True

            results[cond["name"]].append(is_correct)

            mark = "○" if is_correct else "✗"
            print(f"  [{cond['name']}] 予測: {pred} ({mark})")

    # 集計結果
    print("\n" + "=" * 60)
    print("【実験結果集計（正解率）】")
    print("=" * 60)
    for name, res in results.items():
        if len(res) > 0:
            acc = sum(res) / len(res) * 100
            print(f"{name}: {acc:.1f}% ({sum(res)}/{len(res)})")
        else:
             print(f"{name}: データなし")

    print("\n💡 考察:")
    print("- ICLでは「ラベルの正確さ」よりも「タスクの形式（フォーマット）」や「入力分布」が重要であるという研究結果があります（Min et al., 2022など）。")
    print("- そのため、ラベルをランダムにしても性能が大きく落ちないことがあります。")
    print("- 一方で、提示順序（Recency Biasなど）は性能に影響を与える場合があります。")

    # ============================================
    # 演習3: Chain-of-Thought (CoT) Prompting
    # ============================================

    print("\n\n" + "=" * 70)
    print("演習3: Chain-of-Thought (CoT) Prompting")
    print("=" * 70)

    print("""
思考過程を例題に含めることで、推論能力が向上することを確認します。
タスク: 簡単な算数の文章題
""")

    # --- 算数文章題のサンプル ---
    math_problems = [
        {
            "question": "太郎は5個のりんごを持っていました。3個もらいました。今何個ありますか？",
            "answer": "8個",
            "reasoning": "最初に5個持っていて、3個もらったので、5 + 3 = 8個です。",
        },
        {
            "question": "花子は10個のみかんを持っていました。4個食べました。今何個ありますか？",
            "answer": "6個",
            "reasoning": "最初に10個持っていて、4個食べたので、10 - 4 = 6個です。",
        },
        {
            "question": "クラスに30人の生徒がいます。そのうち12人が男子です。女子は何人ですか？",
            "answer": "18人",
            "reasoning": "全体が30人で、男子が12人なので、女子は 30 - 12 = 18人です。",
        },
    ]

    # テスト問題
    test_problem = {
        "question": "本屋で500円の本を2冊と300円の雑誌を1冊買いました。合計金額はいくらですか？",
        "answer": "1300円",
    }

    print("\n【テスト問題】")
    print(f"問題: {test_problem['question']}")
    print(f"正解: {test_problem['answer']}")

    # --- 実験1: 通常のFew-shot ---
    print("\n" + "-" * 70)
    print("実験1: 通常のFew-shot（思考過程なし）")
    print("-" * 70)

    standard_prompt = "以下の算数の問題を解いてください。\n\n"

    # 例題（答えのみ）
    for i, example in enumerate(math_problems[:2], 1):
        standard_prompt += f"問題: {example['question']}\n"
        standard_prompt += f"答え: {example['answer']}\n\n"

    # テスト問題
    standard_prompt += f"問題: {test_problem['question']}\n答え:"

    print(f"プロンプト:\n{standard_prompt}")

    try:
        output_standard = generator(
            standard_prompt,
            max_new_tokens=30,
            num_return_sequences=1,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
            if tokenizer.eos_token_id
            else tokenizer.pad_token_id,
            truncation=True,
        )

        result_standard = output_standard[0]["generated_text"][
            len(standard_prompt) :
        ].strip()
        print(f"\n生成された答え: {result_standard[:50]}")

    except Exception as e:
        print(f"⚠️ 生成エラー: {e}")
        result_standard = "エラー"

    # --- 実験2: CoT Few-shot ---
    print("\n" + "-" * 70)
    print("実験2: CoT Few-shot（思考過程あり）")
    print("-" * 70)

    cot_prompt = "以下の算数の問題を、計算過程を示しながら解いてください。\n\n"

    # 例題（思考過程付き）
    for i, example in enumerate(math_problems[:2], 1):
        cot_prompt += f"問題: {example['question']}\n"
        cot_prompt += f"考え方: {example['reasoning']}\n"
        cot_prompt += f"答え: {example['answer']}\n\n"

    # テスト問題
    cot_prompt += f"問題: {test_problem['question']}\n考え方:"

    print(f"プロンプト:\n{cot_prompt}")

    try:
        output_cot = generator(
            cot_prompt,
            max_new_tokens=50,
            num_return_sequences=1,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
            if tokenizer.eos_token_id
            else tokenizer.pad_token_id,
            truncation=True,
        )

        result_cot = output_cot[0]["generated_text"][len(cot_prompt) :].strip()
        print(f"\n生成された思考過程と答え: {result_cot[:100]}...")

    except Exception as e:
        print(f"⚠️ 生成エラー: {e}")
        result_cot = "エラー"

    # --- 実験3: Zero-shot CoT ---
    print("\n" + "-" * 70)
    print("実験3: Zero-shot CoT（「段階的に考えましょう」を追加）")
    print("-" * 70)

    zero_shot_cot_prompt = f"""以下の算数の問題を解いてください。段階的に考えましょう。

問題: {test_problem["question"]}

考え方:"""

    print(f"プロンプト:\n{zero_shot_cot_prompt}")

    try:
        output_zero_cot = generator(
            zero_shot_cot_prompt,
            max_new_tokens=50,
            num_return_sequences=1,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
            if tokenizer.eos_token_id
            else tokenizer.pad_token_id,
            truncation=True,
        )

        result_zero_cot = output_zero_cot[0]["generated_text"][
            len(zero_shot_cot_prompt) :
        ].strip()
        print(f"\n生成された思考過程: {result_zero_cot[:100]}...")

    except Exception as e:
        print(f"⚠️ 生成エラー: {e}")
        result_zero_cot = "エラー"

    # 結果比較
    print("\n" + "=" * 70)
    print("【CoT効果の比較】")
    print("=" * 70)
    print(f"正解: {test_problem['answer']}")
    print(f"\n通常Few-shot:    {result_standard[:50]}")
    print(f"CoT Few-shot:    {result_cot[:50]}")
    print(f"Zero-shot CoT:   {result_zero_cot[:50]}")
    print("\n💡 CoTでは思考過程を明示することで、より正確な推論が可能になります")



    # ============================================
    # プロンプトエンジニアリングのベストプラクティス
    # ============================================

    print("\n\n" + "=" * 70)
    print("プロンプトエンジニアリングのベストプラクティス")
    print("=" * 70)

    best_practices = """
【1. 明確で具体的な指示】
✓ タスクを明確に定義する
✓ 曖昧な表現を避ける
✗ 「感情を教えて」→ ✓ 「感情をポジティブ/ネガティブ/中立で分類」

【2. 出力形式の指定】
✓ 期待する出力の形式を明示する
✓ 箇条書き、JSON、単語のみなど
✗ 形式指定なし → ✓ 「感情のみを1単語で答えてください」

【3. 役割の設定】
✓ 「あなたは〜の専門家です」と役割を与える
✓ タスクに適した文脈を提供
例: 「あなたは感情分析の専門家です」

【4. 例題の提供（Few-shot）】
✓ 入力と期待される出力の例を示す
✓ 多様な例を2-5個程度
✓ 例のバランスを考慮

【5. 制約条件の明示】
✓ 文字数、使用語彙、トーンなどを指定
例: 「100文字以内で」「専門用語を避けて」

【6. 段階的思考の促進（CoT）】
✓ 「段階的に考えましょう」を追加
✓ 複雑な推論タスクで効果的
✓ 例題に思考過程を含める

【7. 否定形より肯定形】
✗ 「〜を含めないでください」
✓ 「〜のみを含めてください」
"""

    print(best_practices)


    # ============================================
    # 参考リソース
    # ============================================

    print("\n\n" + "=" * 70)
    print("参考リソース - さらに学ぶために")
    print("=" * 70)

    resources = """
【公式ドキュメント】
• OpenAI Prompt Engineering Guide
  https://platform.openai.com/docs/guides/prompt-engineering
  - プロンプトエンジニアリングの公式ガイド

【技術解説】
• Prompt Engineering by Lilian Weng
  https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/
  - 包括的な技術解説とベストプラクティス

【学習リソース】
• Prompt Engineering Guide (GitHub)
  https://github.com/dair-ai/Prompt-Engineering-Guide
  - 論文、チュートリアル、実例のコレクション
  - 多言語対応（日本語あり）

• Awesome ChatGPT Prompts
  https://github.com/f/awesome-chatgpt-prompts
  - 様々なタスクのプロンプト例集
  - コミュニティで共有されたベストプラクティス

【データセット】
• Hugging Face Datasets
  https://huggingface.co/datasets
  - 公開されている多様なデータセット
  - 日本語データセットも豊富

【モデル】
• Hugging Face Model Hub
  https://huggingface.co/models
  - APIキー不要の公開モデルが多数
  - 日本語対応モデルも充実
"""

    print(resources)

    # ============================================
    # まとめ
    # ============================================

    print("\n\n" + "=" * 70)
    print("演習のまとめ")
    print("=" * 70)

    summary = f"""
【本日学んだこと】

1️⃣ プロンプティングの基礎
   ✓ Zero-shot の動作確認
   ✓ プロンプト設計の重要性

2️⃣ In-Context Learning (ICL)の性質
   ✓ Few-shotによりモデルがタスク形式を学習
   ✓ ラベルの正確さよりも「形式」や「分布」が重要（Label Chaos）
   ✓ 事例の提示順序が結果に影響する可能性

3️⃣ Chain-of-Thought Prompting
   ✓ 思考過程を明示することで推論能力向上
   ✓ 複雑なタスクで特に効果的

【使用したモデル】
モデル: {model_name}
特徴: {loaded_model_info["description"]}

【使用したデータセット】
データセット: {dataset_info["name"]}
説明: {dataset_info["description"]}

【重要なポイント】
• プロンプトは「指示書」- 具体的で明確に
• ICLは不思議な性質を持つ（ラベルが間違っていても動くことがある）
• CoTは「思考の見える化」- 推論タスクで強力

【次のステップ】
• 様々なタスクでプロンプトを設計してみる
• モデルサイズによる挙動の違いを確かめる
• 最新のプロンプト技術を学ぶ
"""

    print(summary)

if __name__ == "__main__":
    main()
