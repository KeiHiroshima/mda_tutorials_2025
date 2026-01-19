
# 必要なライブラリのインストール
# pip install transformers torch accelerate

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_name):
    """
    指定されたモデルとトークナイザーをロードする関数
    """
    print(f"Loading model: {model_name}...")
    try:
        # device_map="auto" でGPUがあれば自動的に使用する (accelerateの機能)
        # torch_dtype=torch.float16 でメモリ使用量を減らし高速化
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        print(f"Failed to load model {model_name}: {e}")
        return None, None

def generate_text(prompt, model, tokenizer, max_length=100):
    """
    LLMにプロンプトを渡して出力を生成する関数
    """
    if model is None or tokenizer is None:
        return "Model or Tokenizer not loaded."

    print("Generating text...")

    # 入力をトークナイズ
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # テキスト生成（時間計測）
    start_time = time.time()
    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id
        )
    end_time = time.time()

    print(f"Inference time: {end_time - start_time:.4f} seconds")

    # 出力をデコード
    output = tokenizer.decode(tokens[0], skip_special_tokens=True)
    return output

if __name__ == "__main__":
    # 使用するモデルの候補 (コメントアウトを外して切り替え)

    # 1. CyberAgent OpenCALM Small (軽量・高速)
    model_name = "cyberagent/open-calm-1b"

    # 3. Qwen2.5-1.5B (比較的新しいモデル)
    # model_name = "Qwen/Qwen2.5-1.5B"

    # モデルとトークナイザーのロード
    model, tokenizer = load_model_and_tokenizer(model_name)

    if model and tokenizer:
        # プロンプトの入力
        input_variable = "横浜国立大学の学長は誰ですか？"
        print(f"Input: {input_variable}")

        # 出力の生成
        output_text = generate_text(input_variable, model, tokenizer)

        print("-" * 30)
        print("Output:")
        print(output_text)
        print("-" * 30)
