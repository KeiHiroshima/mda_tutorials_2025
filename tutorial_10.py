"""
MDA入門 2025年度 第10回演習
大規模言語モデル (LLM) とプロンプティング

演習内容:
1. プロンプティングの実践（Zero-shot / Few-shot）
2. In-Context Learningの体験
3. Chain-of-Thought (CoT) Promptingの実験

使用モデル:
- rinna/japanese-gpt2-medium (日本語GPT-2、APIキー不要)
- cyberagent/open-calm-small (日本語LLM、APIキー不要)

使用データセット:
- livedoor ニュースコーパス (公開日本語ニュースデータ)
- JGLUEベンチマーク (日本語自然言語理解タスク)
"""

# ============================================
# 環境セットアップ
# ============================================

print("=" * 70)
print("環境セットアップ中...")
print("=" * 70)

# 必要なライブラリのインストール
#!pip install -q transformers torch datasets sentencepiece fugashi ipadic

import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
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
# モデルの準備
# ============================================

print("\n" + "=" * 70)
print("公開LLMモデルの読み込み")
print("=" * 70)

# 軽量な日本語LLMモデルを使用（APIキー不要）
model_name = "rinna/japanese-gpt2-medium"  # 約330Mパラメータ

print(f"\nモデル: {model_name}")
print("※ このモデルは無料で利用可能な公開モデルです（APIキー不要）")
print("読み込み中...")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # GPUが利用可能ならGPUを使用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print("✓ モデルの読み込み完了")
    print(f"  デバイス: {device}")
    print(f"  パラメータ数: 約{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")

except Exception as e:
    print(f"\n⚠️ エラー: {e}")
    print("別のモデルを試します...")
    model_name = "cyberagent/open-calm-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"✓ 代替モデル {model_name} の読み込み完了")

# テキスト生成用のパイプライン作成
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if device == "cuda" else -1,
)

print("\n✓ テキスト生成パイプラインの準備完了")

# ============================================
# 公開データセットの読み込み
# ============================================

print("\n" + "=" * 70)
print("公開データセットの読み込み")
print("=" * 70)

print("\nlivedoor ニュースコーパスを使用します")
print("※ 日本語のニュース記事（9カテゴリ）の公開データセット")

try:
    # livedoor ニュースコーパスの読み込み
    dataset = load_dataset("livedoor_news_corpus", split="train")

    print("✓ データセット読み込み完了")
    print(f"  総記事数: {len(dataset)}")
    print(f"  カテゴリ: {set(dataset['category'])}")

    # サンプルデータの表示
    sample = dataset[0]
    print("\n【サンプル記事】")
    print(f"カテゴリ: {sample['category']}")
    print(f"タイトル: {sample['title']}")
    print(f"本文（抜粋）: {sample['text'][:100]}...")

except Exception as e:
    print(f"\n⚠️ livedoorデータセット読み込みエラー: {e}")
    print("代替として手動でサンプルデータを作成します...")

    # 代替サンプルデータ
    dataset = [
        {
            "category": "sports",
            "title": "日本代表が劇的勝利",
            "text": "サッカー日本代表は昨日の試合で3-2の逆転勝利を収めた。後半ロスタイムの決勝ゴールでスタジアムは大歓声に包まれた。",
        },
        {
            "category": "technology",
            "title": "新型AIチップ発表",
            "text": "大手半導体メーカーが次世代AIチップを発表した。性能は前世代比で5倍に向上し、消費電力は半分に削減されている。",
        },
        {
            "category": "entertainment",
            "title": "人気映画が興行収入記録更新",
            "text": "公開中の話題作が週末興行収入で歴代最高記録を更新した。全国の映画館で満席が続出している。",
        },
        {
            "category": "economy",
            "title": "株価が急上昇",
            "text": "東京株式市場で日経平均株価が大幅に上昇した。好調な企業業績を受けて投資家心理が改善している。",
        },
        {
            "category": "technology",
            "title": "量子コンピュータで新記録",
            "text": "研究チームが量子コンピュータを使った計算で新記録を達成した。従来のスーパーコンピュータでは不可能だった規模の計算に成功した。",
        },
    ]
    print(f"✓ サンプルデータ作成完了（{len(dataset)}件）")

# ============================================
# 演習1: プロンプティングの基礎実践
# ============================================

print("\n\n" + "=" * 70)
print("演習1: プロンプティングの基礎実践")
print("=" * 70)

print("""
この演習では、プロンプトの設計が出力にどう影響するかを学びます。
- Zero-shot prompting（例題なし）
- Few-shot prompting（例題あり）
の違いを実際に確認します。
""")

# --- タスク1: ニュース記事の要約 ---
print("\n" + "-" * 70)
print("タスク1: ニュース記事の要約")
print("-" * 70)

# テスト用の記事を選択
if isinstance(dataset, list):
    test_article = dataset[0]
else:
    test_article = dataset[100]

print("\n【元記事】")
print(f"カテゴリ: {test_article['category']}")
print(f"タイトル: {test_article['title']}")
if isinstance(dataset, list):
    print(f"本文: {test_article['text']}")
else:
    print(f"本文: {test_article['text'][:200]}...")

# Zero-shot prompting
print("\n【実験1-1】Zero-shot Prompting")
print("-" * 50)

zero_shot_prompt = f"""以下のニュース記事を1文で要約してください。

記事: {test_article["text"][:200] if not isinstance(dataset, list) else test_article["text"]}

要約:"""

print("プロンプト:")
print(zero_shot_prompt[:150] + "...")

# テキスト生成
try:
    output_zero = generator(
        zero_shot_prompt,
        max_new_tokens=50,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        truncation=True,
    )

    generated_text = output_zero[0]["generated_text"]
    # プロンプト部分を除去して生成部分のみ抽出
    summary_zero = generated_text[len(zero_shot_prompt) :].strip()

    print("\n生成された要約:")
    print(f"{summary_zero[:100]}...")

except Exception as e:
    print(f"⚠️ 生成エラー: {e}")
    summary_zero = "（生成失敗）"

# Few-shot prompting
print("\n\n【実験1-2】Few-shot Prompting（例題付き）")
print("-" * 50)

# 例題を作成
if isinstance(dataset, list):
    examples = dataset[1:3]
else:
    examples = [dataset[i] for i in [10, 20]]

few_shot_prompt = "以下の例を参考に、ニュース記事を1文で要約してください。\n\n"

# 例題を追加
for i, example in enumerate(examples, 1):
    ex_text = (
        example["text"][:150] if not isinstance(dataset, list) else example["text"]
    )
    few_shot_prompt += f"例{i}:\n記事: {ex_text}\n要約: {example['title']}\n\n"

# テスト記事を追加
test_text = (
    test_article["text"][:200]
    if not isinstance(dataset, list)
    else test_article["text"]
)
few_shot_prompt += f"記事: {test_text}\n\n要約:"

print("プロンプト（例題2つ付き）:")
print(few_shot_prompt[:200] + "...")

# テキスト生成
try:
    output_few = generator(
        few_shot_prompt,
        max_new_tokens=50,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        truncation=True,
    )

    generated_text_few = output_few[0]["generated_text"]
    summary_few = generated_text_few[len(few_shot_prompt) :].strip()

    print("\n生成された要約:")
    print(f"{summary_few[:100]}...")

except Exception as e:
    print(f"⚠️ 生成エラー: {e}")
    summary_few = "（生成失敗）"

# 比較
print("\n" + "=" * 70)
print("【比較結果】")
print("=" * 70)
print(f"Zero-shot: {summary_zero[:80]}...")
print(f"Few-shot:  {summary_few[:80]}...")
print("\n💡 Few-shotでは例題から形式や長さを学習し、より適切な要約になる傾向があります")

# ============================================
# 演習2: In-Context Learningの体験
# ============================================

print("\n\n" + "=" * 70)
print("演習2: In-Context Learningの体験")
print("=" * 70)

print("""
例題の数を変えることで性能がどう変化するかを観察します。
タスク: ニュース記事のカテゴリ分類
""")

# --- データ準備 ---
print("\n" + "-" * 70)
print("データ準備")
print("-" * 70)

# カテゴリごとにサンプルを選択
if isinstance(dataset, list):
    categories = list(set([item["category"] for item in dataset]))
    category_samples = {}
    for cat in categories:
        category_samples[cat] = [item for item in dataset if item["category"] == cat]
else:
    categories = list(set(dataset["category"]))
    category_samples = {}
    for cat in categories[:5]:  # 最初の5カテゴリのみ使用
        samples = [item for item in dataset if item["category"] == cat]
        category_samples[cat] = samples[:10]  # 各カテゴリ10件まで

print(f"使用カテゴリ: {list(category_samples.keys())}")
print(f"各カテゴリのサンプル数: {[len(v) for v in category_samples.values()]}")

# テスト用記事を選択
test_category = list(category_samples.keys())[0]
if isinstance(dataset, list):
    test_sample = category_samples[test_category][0]
else:
    test_sample = category_samples[test_category][-1]

print("\n【テスト記事】")
print(f"正解カテゴリ: {test_sample['category']}")
print(f"タイトル: {test_sample['title']}")

# --- 実験: 例題数を変えて分類 ---
print("\n" + "-" * 70)
print("実験: 例題数と分類精度の関係")
print("-" * 70)


def create_classification_prompt(num_examples, test_text, category_samples):
    """分類用プロンプトを作成"""
    prompt = "以下のニュース記事をカテゴリに分類してください。\n"
    prompt += f"カテゴリ: {', '.join(category_samples.keys())}\n\n"

    if num_examples > 0:
        # 例題を追加
        examples_added = 0
        for cat, samples in category_samples.items():
            for sample in samples[:num_examples]:
                if examples_added >= num_examples * len(category_samples):
                    break
                ex_text = sample["text"][:100] if "text" in sample else sample["title"]
                prompt += f"記事: {ex_text}...\nカテゴリ: {sample['category']}\n\n"
                examples_added += 1

    # テスト記事を追加
    prompt += f"記事: {test_text}...\nカテゴリ:"
    return prompt


# Zero-shot（例題0個）
print("\n【Zero-shot（例題なし）】")
test_text = test_sample["text"][:100] if "text" in test_sample else test_sample["title"]
prompt_0shot = create_classification_prompt(0, test_text, category_samples)

print(f"プロンプト:\n{prompt_0shot[:200]}...")

try:
    output = generator(
        prompt_0shot,
        max_new_tokens=10,
        num_return_sequences=1,
        temperature=0.3,
        pad_token_id=tokenizer.eos_token_id,
        truncation=True,
    )
    result_0shot = output[0]["generated_text"][len(prompt_0shot) :].strip()
    print(f"予測カテゴリ: {result_0shot[:20]}")
except Exception as e:
    print(f"⚠️ 生成エラー: {e}")
    result_0shot = "エラー"

# Few-shot（例題2個）
print("\n【Few-shot（例題2個ずつ）】")
prompt_2shot = create_classification_prompt(2, test_text, category_samples)

print(f"プロンプト（例題あり）:\n{prompt_2shot[:250]}...")

try:
    output = generator(
        prompt_2shot,
        max_new_tokens=10,
        num_return_sequences=1,
        temperature=0.3,
        pad_token_id=tokenizer.eos_token_id,
        truncation=True,
    )
    result_2shot = output[0]["generated_text"][len(prompt_2shot) :].strip()
    print(f"予測カテゴリ: {result_2shot[:20]}")
except Exception as e:
    print(f"⚠️ 生成エラー: {e}")
    result_2shot = "エラー"

print("\n" + "=" * 70)
print("【結果比較】")
print("=" * 70)
print(f"正解カテゴリ:  {test_sample['category']}")
print(f"Zero-shot予測: {result_0shot[:20]}")
print(f"Few-shot予測:  {result_2shot[:20]}")
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
        pad_token_id=tokenizer.eos_token_id,
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
        pad_token_id=tokenizer.eos_token_id,
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
        pad_token_id=tokenizer.eos_token_id,
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

# タスク: テキスト分類（感情分析）
sentiment_samples = [
    {
        "text": "このレストランの料理は最高でした！また来たいです。",
        "sentiment": "ポジティブ",
    },
    {"text": "サービスが悪くて二度と行きたくないです。", "sentiment": "ネガティブ"},
    {"text": "料理は美味しいけど、値段が高すぎます。", "sentiment": "中立"},
]

test_text_sentiment = "雰囲気は良かったですが、料理が冷めていて残念でした。"

print("\n【テストテキスト】")
print(f"テキスト: {test_text_sentiment}")

# パターン1: 曖昧な指示
print("\n" + "-" * 70)
print("パターン1: 曖昧な指示")
print("-" * 70)

vague_prompt = f"感情を教えて。\n\n{test_text_sentiment}\n\n感情:"
print(f"プロンプト:\n{vague_prompt}")

try:
    output_vague = generator(
        vague_prompt,
        max_new_tokens=20,
        temperature=0.3,
        pad_token_id=tokenizer.eos_token_id,
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

テキスト: {test_text_sentiment}

感情:"""

print(f"プロンプト:\n{clear_prompt}")

try:
    output_clear = generator(
        clear_prompt,
        max_new_tokens=10,
        temperature=0.3,
        pad_token_id=tokenizer.eos_token_id,
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
best_prompt += f"テキスト: {test_text_sentiment}\n感情:"

print(f"プロンプト（一部省略）:\n{best_prompt[:200]}...")

try:
    output_best = generator(
        best_prompt,
        max_new_tokens=10,
        temperature=0.3,
        pad_token_id=tokenizer.eos_token_id,
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
# 発展課題
# ============================================

print("\n\n" + "=" * 70)
print("発展課題（任意）")
print("=" * 70)

advanced_exercises = """
以下の課題に挑戦してみましょう:

【課題A】独自タスクでのプロンプト最適化
1. 自分で興味のあるタスクを選ぶ（翻訳、要約、分類など）
2. 様々なプロンプトパターンを試す
3. 最も効果的なプロンプトを見つける
4. 改善のポイントをまとめる

【課題B】Few-shot例題の選択実験
1. 同じタスクで異なる例題セットを用意
2. 例題の数（1, 3, 5個など）を変えて比較
3. 例題の質（多様性、バランス）が結果に与える影響を分析

【課題C】CoTの効果検証
1. 複数の推論タスクを用意
2. 通常のFew-shot vs CoT Few-shotで比較
3. どのようなタスクでCoTが特に効果的か考察

【課題D】プロンプトテンプレート作成
1. 汎用的に使えるプロンプトテンプレートを設計
2. 複数のタスクで試して有効性を検証
3. テンプレートをドキュメント化

【課題E】公開データセットでの評価
1. livedoorニュースコーパスの別カテゴリで分類精度を測定
2. 異なるプロンプト戦略で精度を比較
3. 結果を定量的に分析
"""

print(advanced_exercises)

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

summary = """
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

print("\n" + "=" * 70)
print("演習終了 - お疲れ様でした！")
print("=" * 70)
print("\n💡 プロンプトエンジニアリングはLLMを効果的に活用する鍵です")
print("   実践を通じて、自分なりのノウハウを蓄積していきましょう！")
