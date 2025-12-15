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
print("\n【実験1-1】Zero-shot Prompting")
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

# Few-shot prompting
print("\n\n【実験1-2】Few-shot Prompting（例題付き）")
print("-" * 50)

# 例題を作成
if isinstance(dataset, list):
    examples = dataset[1:4]
else:
    examples = [dataset[i] for i in [10, 20, 30]]

few_shot_prompt = "以下の例を参考に、テキストの感情を「ポジティブ」「ネガティブ」「中立」で分類してください。\n\n"

# 例題を追加
for i, example in enumerate(examples[:3], 1):
    if isinstance(dataset, list):
        ex_text = example["text"]
        ex_label = example["label"]
    else:
        if "text" in example:
            ex_text = example["text"]
        elif "sentence" in example:
            ex_text = example["sentence"]
        elif "review" in example:
            ex_text = example["review"]
        else:
            ex_text = str(list(example.values())[0])[:100]

        if "label" in example:
            ex_label = example["label"]
        elif "sentiment" in example:
            ex_label = example["sentiment"]
        else:
            ex_label = "ポジティブ" if i % 2 == 0 else "ネガティブ"

    few_shot_prompt += f"例{i}:\nテキスト: {ex_text[:80]}...\n感情: {ex_label}\n\n"

# テスト文を追加
few_shot_prompt += f"テキスト: {test_text}\n\n感情:"

print("プロンプト（例題3つ付き）:")
print(few_shot_prompt[:200] + "...")

# テキスト生成
try:
    output_few = generator(
        few_shot_prompt,
        max_new_tokens=20,
        num_return_sequences=1,
        temperature=0.3,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
        if tokenizer.eos_token_id
        else tokenizer.pad_token_id,
        truncation=True,
    )

    generated_text_few = output_few[0]["generated_text"]
    summary_few = generated_text_few[len(few_shot_prompt) :].strip()

    print("\n生成された分類結果:")
    print(f"{summary_few[:50]}")

except Exception as e:
    print(f"⚠️ 生成エラー: {e}")
    summary_few = "（生成失敗）"

# 比較
print("\n" + "=" * 70)
print("【比較結果】")
print("=" * 70)
print(f"正解:       {test_label}")
print(f"Zero-shot:  {summary_zero[:30]}")
print(f"Few-shot:   {summary_few[:30]}")
print(
    "\n💡 Few-shotでは例題から形式やカテゴリを学習し、より適切な分類になる傾向があります"
)

# ============================================
# 演習2: In-Context Learningの体験
# ============================================

print("\n\n" + "=" * 70)
print("演習2: In-Context Learningの体験")
print("=" * 70)

print("""
例題の数を変えることで性能がどう変化するかを観察します。
タスク: テキストの感情分類
""")

# --- 実験: 例題数を変えて分類 ---
print("\n" + "-" * 70)
print("実験: 例題数と分類精度の関係")
print("-" * 70)


def create_sentiment_prompt(num_examples, test_text, dataset):
    """感情分類用プロンプトを作成"""
    prompt = "以下のテキストの感情を「ポジティブ」「ネガティブ」「中立」で分類してください。\n\n"

    if num_examples > 0:
        # 例題を追加
        examples_to_use = min(num_examples, 5)
        if isinstance(dataset, list):
            examples = dataset[:examples_to_use]
        else:
            examples = [dataset[i * 10] for i in range(examples_to_use)]

        for i, example in enumerate(examples, 1):
            if isinstance(dataset, list):
                ex_text = example["text"][:60]
                ex_label = example["label"]
            else:
                if "text" in example:
                    ex_text = example["text"][:60]
                elif "review" in example:
                    ex_text = example["review"][:60]
                else:
                    ex_text = str(list(example.values())[0])[:60]

                if "label" in example:
                    ex_label = example["label"]
                else:
                    ex_label = "ポジティブ" if i % 2 == 0 else "ネガティブ"

            prompt += f"テキスト: {ex_text}...\n感情: {ex_label}\n\n"

    # テスト文を追加
    prompt += f"テキスト: {test_text}\n感情:"
    return prompt


# テスト用テキスト（新しいサンプル）
if isinstance(dataset, list):
    test_sample = dataset[5]
    icl_test_text = test_sample["text"]
    icl_test_label = test_sample["label"]
else:
    icl_test_idx = 100
    icl_sample = dataset[icl_test_idx]
    if "text" in icl_sample:
        icl_test_text = icl_sample["text"]
    elif "review" in icl_sample:
        icl_test_text = icl_sample["review"]
    else:
        icl_test_text = "この製品は期待以上の性能でした。"

    if "label" in icl_sample:
        icl_test_label = icl_sample["label"]
    else:
        icl_test_label = "ポジティブ"

print("\n【テストテキスト】")
print(f"テキスト: {icl_test_text}")
print(f"正解: {icl_test_label}")

# Zero-shot（例題0個）
print("\n【Zero-shot（例題なし）】")
prompt_0shot = create_sentiment_prompt(0, icl_test_text, dataset)

print(f"プロンプト:\n{prompt_0shot[:150]}...")

try:
    output = generator(
        prompt_0shot,
        max_new_tokens=15,
        num_return_sequences=1,
        temperature=0.3,
        pad_token_id=tokenizer.eos_token_id
        if tokenizer.eos_token_id
        else tokenizer.pad_token_id,
        truncation=True,
    )
    result_0shot = output[0]["generated_text"][len(prompt_0shot) :].strip()
    print(f"予測感情: {result_0shot[:30]}")
except Exception as e:
    print(f"⚠️ 生成エラー: {e}")
    result_0shot = "エラー"

# Few-shot（例題3個）
print("\n【Few-shot（例題3個）】")
prompt_3shot = create_sentiment_prompt(3, icl_test_text, dataset)

print(f"プロンプト（例題あり）:\n{prompt_3shot[:200]}...")

try:
    output = generator(
        prompt_3shot,
        max_new_tokens=15,
        num_return_sequences=1,
        temperature=0.3,
        pad_token_id=tokenizer.eos_token_id
        if tokenizer.eos_token_id
        else tokenizer.pad_token_id,
        truncation=True,
    )
    result_3shot = output[0]["generated_text"][len(prompt_3shot) :].strip()
    print(f"予測感情: {result_3shot[:30]}")
except Exception as e:
    print(f"⚠️ 生成エラー: {e}")
    result_3shot = "エラー"

print("\n" + "=" * 70)
print("【結果比較】")
print("=" * 70)
print(f"正解:         {icl_test_label}")
print(f"Zero-shot予測: {result_0shot[:30]}")
print(f"Few-shot予測:  {result_3shot[:30]}")
print("\n💡 例題があることで、モデルはタスクの形式をより正確に理解できます")

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
# 演習4: プロンプトエンジニアリングの実践
# ============================================

print("\n\n" + "=" * 70)
print("演習4: プロンプトエンジニアリングの実践")
print("=" * 70)

print("""
効果的なプロンプト設計の原則を実践します。
同じタスクでも、プロンプトの書き方で結果が大きく変わることを確認します。
""")

# タスク用のテキスト
prac_text = "雰囲気は良かったですが、料理が冷めていて残念でした。"

print("\n【テストテキスト】")
print(f"テキスト: {prac_text}")

# パターン1: 曖昧な指示
print("\n" + "-" * 70)
print("パターン1: 曖昧な指示")
print("-" * 70)

vague_prompt = f"感情を教えて。\n\n{prac_text}\n\n感情:"
print(f"プロンプト:\n{vague_prompt}")

try:
    output_vague = generator(
        vague_prompt,
        max_new_tokens=20,
        temperature=0.3,
        pad_token_id=tokenizer.eos_token_id
        if tokenizer.eos_token_id
        else tokenizer.pad_token_id,
        truncation=True,
    )
    result_vague = output_vague[0]["generated_text"][len(vague_prompt) :].strip()
    print(f"\n結果: {result_vague[:30]}")
except Exception as e:
    result_vague = "エラー"
    print(f"⚠️ エラー: {e}")

# パターン2: 明確な指示 + 形式指定
print("\n" + "-" * 70)
print("パターン2: 明確な指示 + 出力形式の指定")
print("-" * 70)

clear_prompt = f"""以下のテキストの感情を「ポジティブ」「ネガティブ」「中立」のいずれかで分類してください。
感情のみを答え、説明は不要です。

テキスト: {prac_text}

感情:"""

print(f"プロンプト:\n{clear_prompt}")

try:
    output_clear = generator(
        clear_prompt,
        max_new_tokens=10,
        temperature=0.3,
        pad_token_id=tokenizer.eos_token_id
        if tokenizer.eos_token_id
        else tokenizer.pad_token_id,
        truncation=True,
    )
    result_clear = output_clear[0]["generated_text"][len(clear_prompt) :].strip()
    print(f"\n結果: {result_clear[:20]}")
except Exception as e:
    result_clear = "エラー"
    print(f"⚠️ エラー: {e}")

# パターン3: 役割設定 + Few-shot + 形式指定
print("\n" + "-" * 70)
print("パターン3: 役割設定 + Few-shot + 形式指定")
print("-" * 70)

best_prompt = """あなたは感情分析の専門家です。以下のテキストの感情を「ポジティブ」「ネガティブ」「中立」で分類してください。

例:
テキスト: このレストランの料理は最高でした！また来たいです。
感情: ポジティブ

テキスト: サービスが悪くて二度と行きたくないです。
感情: ネガティブ

テキスト: 料理は美味しいけど、値段が高すぎます。
感情: 中立

"""
best_prompt += f"テキスト: {prac_text}\n感情:"

print(f"プロンプト（一部省略）:\n{best_prompt[:200]}...")

try:
    output_best = generator(
        best_prompt,
        max_new_tokens=10,
        temperature=0.3,
        pad_token_id=tokenizer.eos_token_id
        if tokenizer.eos_token_id
        else tokenizer.pad_token_id,
        truncation=True,
    )
    result_best = output_best[0]["generated_text"][len(best_prompt) :].strip()
    print(f"\n結果: {result_best[:20]}")
except Exception as e:
    result_best = "エラー"
    print(f"⚠️ エラー: {e}")

# 比較表示
print("\n" + "=" * 70)
print("【プロンプト設計による結果の違い】")
print("=" * 70)
print(f"パターン1（曖昧）:          {result_vague[:30]}")
print(f"パターン2（明確+形式）:     {result_clear[:30]}")
print(f"パターン3（役割+Few+形式）: {result_best[:30]}")
print("\n💡 プロンプトが具体的で構造化されているほど、期待する出力が得られます")

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
   ✓ Zero-shot vs Few-shotの違いと効果
   ✓ プロンプト設計が出力品質に大きく影響
   ✓ 明確な指示と形式指定の重要性

2️⃣ In-Context Learning
   ✓ 例題から学習する能力
   ✓ 例題の数と質が性能に影響
   ✓ 適切な例題選択の重要性

3️⃣ Chain-of-Thought Prompting
   ✓ 思考過程を明示することで推論能力向上
   ✓ 複雑なタスクで特に効果的
   ✓ Zero-shot CoTでも効果あり

4️⃣ プロンプトエンジニアリングのベストプラクティス
   ✓ 役割設定、形式指定、制約条件の明示
   ✓ 段階的思考の促進
   ✓ 適切な例題の提供

【使用したモデル】
モデル: {model_name}
特徴: {loaded_model_info["description"]}

【使用したデータセット】
データセット: {dataset_info["name"]}
説明: {dataset_info["description"]}

【重要なポイント】
• プロンプトは「指示書」- 具体的で明確に
• 例題は「お手本」- 多様性とバランスを考慮
• CoTは「思考の見える化」- 推論タスクで強力
• 実験と改善のサイクルが重要

【次のステップ】
• 様々なタスクでプロンプトを設計してみる
• 公開データセットで評価実験を行う
• 最新のプロンプト技術を学ぶ（参考リソース参照）
• 独自のプロンプトテンプレートを構築する
"""

print(summary)
