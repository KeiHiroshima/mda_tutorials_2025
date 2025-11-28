"""
========================================
第二回講義実習：データ品質とAI性能の関係
========================================

このノートブックでは、以下を学びます：
1. データの「量」がAI性能に与える影響
2. データの「質」がAI性能に与える影響
3. "Garbage In, Garbage Out" の原則

所要時間：約30-40分
"""

# ============================================
# セクション1: Google Colabの動作確認
# ============================================
print("=" * 50)
print("🔧 Google Colabの環境を確認します")
print("=" * 50)

import sys

import torch

# Pythonバージョン確認
print(f"\n📌 Pythonバージョン: {sys.version.split()[0]}")

# PyTorchバージョン確認
print(f"📌 PyTorchバージョン: {torch.__version__}")

# GPU利用可能か確認
if torch.cuda.is_available():
    print(f"✅ GPU利用可能: {torch.cuda.get_device_name(0)}")
    print(
        f"   GPUメモリ: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )
    device = "cuda"
else:
    print("⚠️  GPUが利用できません。CPUで実行します。")
    print(
        "   （Colab画面右上のメニュー > ランタイムのタイプを変更 > T4 GPUを選択してください）"
    )
    device = "cpu"

print("\n環境確認完了！次のセルに進んでください。\n")

# ============================================
# セクション2: 必要なライブラリのインストール
# ============================================
print("=" * 50)
print("📦 必要なライブラリをインストールします（1-2分かかります）")
print("=" * 50)

# PyTorch Lightningのインストール
#!pip install pytorch-lightning torchmetrics -q

print("✅ インストール完了！\n")

# ============================================
# セクション3: ライブラリのインポート
# ============================================
print("=" * 50)
print("📚 ライブラリを読み込みます")
print("=" * 50)

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import japanize_matplotlib

warnings.filterwarnings("ignore")

# 日本語フォント設定（グラフ用）
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

print("✅ ライブラリの読み込み完了！\n")

# ============================================
# セクション4: データセットの準備
# ============================================
print("=" * 50)
print("🖼️  画像データセット (CIFAR-10) を準備します")
print("=" * 50)
print("\nCIFAR-10とは：")
print("- 10種類の物体（飛行機、車、鳥、猫など）の小さな画像")
print("- 訓練用: 50,000枚、テスト用: 10,000枚")
print("- 画像サイズ: 32×32ピクセル（とても小さい！）\n")

# データの前処理（画像を数値に変換）
transform = transforms.Compose(
    [
        transforms.ToTensor(),  # 画像を数値の配列に変換
        transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        ),  # 数値を-1〜1の範囲に正規化
    ]
)

# データセットのダウンロード
print("データをダウンロード中...")
train_dataset_full = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

# クラス名（物体の種類）
class_names = ["飛行機", "車", "鳥", "猫", "鹿", "犬", "蛙", "馬", "船", "トラック"]

print("\n✅ データセット準備完了！")
print(f"   訓練データ: {len(train_dataset_full)}枚")
print(f"   テストデータ: {len(test_dataset)}枚")

# サンプル画像を表示
print("\n📸 データセットのサンプルを表示します：")
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle("CIFAR-10 データセットのサンプル画像", fontsize=14, fontweight="bold")

for idx, ax in enumerate(axes.flat):
    img, label = train_dataset_full[idx]
    # 画像を表示用に変換（-1〜1 → 0〜1）
    img = img / 2 + 0.5
    ax.imshow(img.permute(1, 2, 0))
    ax.set_title(f"{class_names[label]}", fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.show()

plt.savefig("cifar10_samples.png")

breakpoint()


# ============================================
# セクション5: AIモデルの定義
# ============================================
print("=" * 50)
print("🤖 AIモデル（畳み込みニューラルネットワーク）を定義します")
print("=" * 50)
print("\nこのモデルは画像を見て、10種類の物体のどれかを当てるAIです。")
print("人間の脳の神経回路を模倣した仕組みで動いています。\n")


class SimpleCNN(pl.LightningModule):
    """
    シンプルな畳み込みニューラルネットワーク

    仕組み（簡単に）：
    1. 画像から特徴を抽出（エッジ、色、形など）
    2. 特徴を組み合わせて物体を認識
    3. 10種類のどれかを予測
    """

    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()

        # 畳み込み層（画像から特徴を抽出）
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # 第1層：32個の特徴を抽出
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 第2層：64個の特徴を抽出
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)  # 第3層：さらに特徴を深掘り

        # プーリング層（画像を小さくする）
        self.pool = nn.MaxPool2d(2, 2)

        # 全結合層（特徴から最終判断）
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)  # 10種類に分類

        # ドロップアウト（過学習を防ぐ）
        self.dropout = nn.Dropout(0.8)

        # 学習の進捗を記録
        self.train_acc_history = []
        self.val_acc_history = []

    def forward(self, x):
        """画像を入力して予測を出力"""
        # 畳み込み + 活性化 + プーリング
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 → 16x16
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 → 8x8
        x = self.pool(F.relu(self.conv3(x)))  # 8x8 → 4x4

        # 平坦化（画像を1次元に）
        x = x.view(-1, 64 * 4 * 4)

        # 全結合層
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def training_step(self, batch, batch_idx):
        """訓練時の処理"""
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        # 精度を計算
        acc = (y_hat.argmax(dim=1) == y).float().mean()

        # ログに記録
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """検証時の処理"""
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        # 精度を計算
        acc = (y_hat.argmax(dim=1) == y).float().mean()

        # ログに記録
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        """最適化手法の設定"""
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


breakpoint()


# ============================================
# セクション6: データ品質を調整する関数
# ============================================
print("=" * 50)
print("🔧 データの「量」と「質」を調整する機能を準備します")
print("=" * 50)


def create_dataset_with_quality(
    dataset,
    data_size_ratio: float = 1.0,
    noise_ratio: float = 0.0,
    label_error_ratio: float = 0.0,
    seed: int = 42,
):
    """
    データセットの量と質を調整する関数

    引数:
        dataset: 元のデータセット
        data_size_ratio: 使用するデータの割合（0.0〜1.0）
            例: 0.1 = 10%のデータのみ使用, 1.0 = 全データ使用
        noise_ratio: ノイズを追加する割合（0.0〜1.0）
            例: 0.3 = 30%の画像にノイズを追加
        label_error_ratio: ラベルを間違える割合（0.0〜1.0）
            例: 0.1 = 10%のラベルをランダムに変更
        seed: 乱数シード（再現性のため）

    戻り値:
        調整されたデータセット
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # データ量の調整
    total_size = len(dataset)
    use_size = int(total_size * data_size_ratio)
    indices = np.random.choice(total_size, use_size, replace=False)

    print("\n📊 データセット調整:")
    print(f"   - 元のサイズ: {total_size}枚")
    print(f"   - 使用サイズ: {use_size}枚 ({data_size_ratio * 100:.0f}%)")
    print(f"   - ノイズ追加: {noise_ratio * 100:.0f}%の画像")
    print(f"   - ラベルエラー: {label_error_ratio * 100:.0f}%のラベル")

    # カスタムデータセットクラス
    class QualityAdjustedDataset(Dataset):
        def __init__(self, base_dataset, indices, noise_ratio, label_error_ratio):
            self.base_dataset = base_dataset
            self.indices = indices
            self.noise_ratio = noise_ratio
            self.label_error_ratio = label_error_ratio

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            real_idx = self.indices[idx]
            img, label = self.base_dataset[real_idx]

            # ノイズの追加（画質を悪くする）
            if np.random.random() < self.noise_ratio:
                noise = torch.randn_like(img) * 0.5  # ランダムノイズ
                img = img + noise
                img = torch.clamp(img, -1, 1)  # 範囲を-1〜1に制限

            # ラベルエラー（間違ったラベルをつける）
            if np.random.random() < self.label_error_ratio:
                label = np.random.randint(0, 10)  # ランダムなラベルに変更

            return img, label

    adjusted_dataset = QualityAdjustedDataset(
        dataset, indices, noise_ratio, label_error_ratio
    )

    return adjusted_dataset


print("\n✅ データ調整機能の準備完了！\n")

breakpoint()


# ============================================
# セクション7: モデル訓練関数
# ============================================
print("=" * 50)
print("🏋️ モデルを訓練する関数を準備します")
print("=" * 50)


def train_model(
    train_dataset,
    test_dataset,
    experiment_name: str,
    max_epochs: int = 10,
    batch_size: int = 128,
):
    """
    モデルを訓練して精度を返す関数

    引数:
        train_dataset: 訓練用データセット
        test_dataset: テスト用データセット
        experiment_name: 実験名（グラフ表示用）
        max_epochs: 訓練回数
        batch_size: 一度に処理する画像数

    戻り値:
        最終的なテスト精度
    """

    print(f"\n{'=' * 50}")
    print(f"🚀 実験開始: {experiment_name}")
    print(f"{'=' * 50}")

    # データローダーの作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
    )

    # モデルの作成
    model = SimpleCNN()

    # トレーナーの設定
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=False,
    )

    # 訓練開始
    print(f"\n⏳ 訓練中... (エポック数: {max_epochs})")
    trainer.fit(model, train_loader, test_loader)

    # テストデータで最終評価
    print("\n📊 最終評価中...")
    results = trainer.validate(model, test_loader)

    final_accuracy = results[0]["val_acc"]

    print("\n✅ 実験完了！")
    print(f"   最終テスト精度: {final_accuracy * 100:.2f}%")

    return final_accuracy


print("\n✅ 訓練関数の準備完了！\n")

breakpoint()


# ============================================
# セクション8: 実験1 - データ量の影響
# ============================================
print("\n" + "=" * 50)
print("🔬 実験1: データの「量」が性能に与える影響")
print("=" * 50)
print("\n【問い】")
print("データが少ない場合と多い場合で、AIの精度はどう変わるでしょうか？\n")

# 実験設定
data_ratios = [0.05, 0.1, 0.2, 0.5, 1.0]  # 5%, 10%, 20%, 50%, 100%
results_quantity = []

print("これから5つの実験を行います（各2-3分）:\n")

for ratio in data_ratios:
    print(f"\n{'─' * 50}")
    print(f"📌 データ量: {ratio * 100:.0f}% ({int(len(train_dataset_full) * ratio)}枚)")

    # データセットの準備
    train_data = create_dataset_with_quality(
        train_dataset_full,
        data_size_ratio=ratio,
        noise_ratio=0.0,  # ノイズなし
        label_error_ratio=0.0,  # ラベルエラーなし
    )

    # モデルの訓練
    accuracy = train_model(
        train_data,
        test_dataset,
        experiment_name=f"データ量 {ratio * 100:.0f}%",
        max_epochs=5,  # 時間短縮のため5エポック
        batch_size=128,
    )

    results_quantity.append({"ratio": ratio, "accuracy": accuracy})

# 結果の可視化
print("\n" + "=" * 50)
print("📈 実験1の結果をグラフで表示")
print("=" * 50)

plt.figure(figsize=(10, 6))
ratios = [r["ratio"] * 100 for r in results_quantity]
accuracies = [r["accuracy"] * 100 for r in results_quantity]

plt.plot(ratios, accuracies, marker="o", linewidth=2, markersize=10, color="#2E86AB")
plt.xlabel("データ量 (%)", fontsize=12, fontweight="bold")
plt.ylabel("テスト精度 (%)", fontsize=12, fontweight="bold")
plt.title("実験1: データ量とAI精度の関係", fontsize=14, fontweight="bold")
plt.grid(True, alpha=0.3)
plt.xticks(ratios)

# 各点に値を表示
for ratio, acc in zip(ratios, accuracies):
    plt.annotate(
        f"{acc:.1f}%",
        xy=(ratio, acc),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        fontsize=10,
        fontweight="bold",
    )

plt.tight_layout()
plt.savefig("data_quantity_vs_accuracy.png")
plt.show()

print("\n💡 考察ポイント:")
print("   - データ量が増えると精度は上がりましたか？")
print("   - どのくらいのデータ量から精度が安定しますか？")
print("   - 少ないデータでも実用的な精度は得られましたか？\n")


breakpoint()


# ============================================
# セクション9: 実験2 - データ品質の影響
# ============================================
print("\n" + "=" * 50)
print("🔬 実験2: データの「質」が性能に与える影響")
print("=" * 50)
print("\n【問い】")
print("データにノイズやエラーがある場合、AIの精度はどう変わるでしょうか？\n")

# 実験設定
quality_settings = [
    {"noise": 0.0, "label_error": 0.0, "name": "完璧なデータ"},
    {"noise": 0.3, "label_error": 0.0, "name": "ノイズ30%"},
    {"noise": 0.0, "label_error": 0.1, "name": "ラベルエラー10%"},
    {"noise": 0.3, "label_error": 0.1, "name": "ノイズ30% + ラベルエラー10%"},
]

results_quality = []

print("これから4つの実験を行います（各2-3分）:\n")

for setting in quality_settings:
    print(f"\n{'─' * 50}")
    print(f"📌 実験: {setting['name']}")

    # データセットの準備（全データを使用）
    train_data = create_dataset_with_quality(
        train_dataset_full,
        data_size_ratio=0.2,  # 時間短縮のため20%使用
        noise_ratio=setting["noise"],
        label_error_ratio=setting["label_error"],
    )

    # 品質が悪いデータのサンプルを表示（最初の実験のみ）
    if setting["noise"] > 0 or setting["label_error"] > 0:
        if results_quality == []:  # 最初の悪いデータの実験時のみ
            print("\n📸 品質の悪いデータのサンプル:")
            fig, axes = plt.subplots(1, 5, figsize=(12, 3))
            fig.suptitle(
                f"{setting['name']} のサンプル", fontsize=12, fontweight="bold"
            )

            for idx, ax in enumerate(axes):
                img, label = train_data[idx]
                img = img / 2 + 0.5  # 表示用に正規化
                img = torch.clamp(img, 0, 1)  # 範囲を0-1に制限
                ax.imshow(img.permute(1, 2, 0))
                ax.set_title(f"{class_names[label]}", fontsize=9)
                ax.axis("off")

            plt.tight_layout()
            plt.show()

    # モデルの訓練
    accuracy = train_model(
        train_data,
        test_dataset,
        experiment_name=setting["name"],
        max_epochs=5,
        batch_size=128,
    )

    results_quality.append(
        {
            "name": setting["name"],
            "noise": setting["noise"],
            "label_error": setting["label_error"],
            "accuracy": accuracy,
        }
    )

# 結果の可視化
print("\n" + "=" * 50)
print("📈 実験2の結果をグラフで表示")
print("=" * 50)

plt.figure(figsize=(12, 6))
names = [r["name"] for r in results_quality]
accuracies = [r["accuracy"] * 100 for r in results_quality]
colors = ["#06D6A0", "#FFD166", "#EF476F", "#AB0011"]

bars = plt.bar(names, accuracies, color=colors, edgecolor="black", linewidth=1.5)
plt.ylabel("テスト精度 (%)", fontsize=12, fontweight="bold")
plt.title("実験2: データ品質とAI精度の関係", fontsize=14, fontweight="bold")
plt.ylim(0, 100)
plt.grid(True, axis="y", alpha=0.3)

# 各バーに値を表示
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 1,
        f"{acc:.1f}%",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig("data_quality_vs_accuracy.png")
plt.show()

print("\n💡 考察ポイント:")
print("   - ノイズがあると精度はどのくらい下がりましたか？")
print("   - ラベルエラーの影響は大きいですか？")
print("   - 複数の問題が組み合わさるとどうなりますか？\n")

breakpoint()


# ============================================
# セクション10: 総合結果とまとめ
# ============================================
print("\n" + "=" * 50)
print("📋 実験結果のまとめ")
print("=" * 50)

print("\n【実験1: データ量の影響】")
print("-" * 50)
for r in results_quantity:
    print(f"  データ量 {r['ratio'] * 100:>5.0f}% → 精度 {r['accuracy'] * 100:>5.2f}%")

print("\n【実験2: データ品質の影響】")
print("-" * 50)
for r in results_quality:
    print(f"  {r['name']:<30} → 精度 {r['accuracy'] * 100:>5.2f}%")

# 最良と最悪の比較
best_quality = max(results_quality, key=lambda x: x["accuracy"])
worst_quality = min(results_quality, key=lambda x: x["accuracy"])

print("\n" + "=" * 50)
print("🎯 重要な発見")
print("=" * 50)

print("\n1️⃣ データ量の影響:")
print(f"   - 最小データ(5%): {results_quantity[0]['accuracy'] * 100:.1f}%")
print(f"   - 最大データ(100%): {results_quantity[-1]['accuracy'] * 100:.1f}%")
print(
    f"   - 差: {(results_quantity[-1]['accuracy'] - results_quantity[0]['accuracy']) * 100:.1f}ポイント"
)

print("\n2️⃣ データ品質の影響:")
print(f"   - 最良({best_quality['name']}): {best_quality['accuracy'] * 100:.1f}%")
print(f"   - 最悪({worst_quality['name']}): {worst_quality['accuracy'] * 100:.1f}%")
print(
    f"   - 差: {(best_quality['accuracy'] - worst_quality['accuracy']) * 100:.1f}ポイント"
)

print("\n" + "=" * 50)
print("💡 「Garbage In, Garbage Out」の原則")
print("=" * 50)
print("""
今日の実験から学んだこと:

✅ データの「量」は重要
   → でも、ある程度以上あれば効果は頭打ち

✅ データの「質」は非常に重要
   → ノイズやエラーは精度を大きく下げる

✅ どんなに優れたAIモデルでも...
   → 悪いデータからは良い結果は生まれない

📌 結論:
   AIプロジェクトで最も重要なのは、
   「良質なデータを集めること」

   モデルの改良より、データの改善が先！
""")

print("\n" + "=" * 50)
print("🎓 実習完了！お疲れ様でした！")
print("=" * 50)
print("\n次回の講義では、統計的思考の基礎を学びます。")
print("今日学んだデータの重要性を忘れずに！\n")
