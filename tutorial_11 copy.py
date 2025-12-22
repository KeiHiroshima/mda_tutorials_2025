"""
AIモデルを支えるハードウェア - 演習資料
MDA入門 第12回

本演習では以下の内容を扱います:
1. ハードウェア情報の確認
2. CPU vs GPU 速度比較実験
3. メモリ使用量の観察
4. 並列計算の効果実感

必要なライブラリ:
- numpy
- cupy (GPUがある場合)
- matplotlib
- torch
- psutil
"""

import subprocess
import time
import warnings

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# 日本語フォントの設定（必要に応じて）
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "DejaVu Sans"]

GPU_AVAILABLE = True

print("\nライブラリのインポートが完了しました")
print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")

# =====================================
# 演習1: ハードウェア情報の確認
# =====================================

print("\n" + "=" * 70)
print("演習1: ハードウェア情報の確認")
print("=" * 70)

print(
    """
この演習では、現在使用している計算環境のハードウェア情報を確認します。
- GPU情報
- CPU情報
- メモリ情報
- PyTorchのデバイス設定
"""
)

# --- GPU情報の確認 ---
print("\n【GPU情報】")
print("-" * 70)

if torch.cuda.is_available():
    print("✓ CUDA is available")
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ Number of GPUs: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        gpu_memory = torch.cuda.get_device_properties(i).total_memory
        print(f"  Total Memory: {gpu_memory / 1e9:.2f} GB")

    # nvidia-smiコマンドを実行
    print("\n【nvidia-smi 出力】")
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=True
        )
        print(result.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("nvidia-smi コマンドが利用できません")

    device = torch.device("cuda")
    print(f"\n使用デバイス: {device} (GPU)")
else:
    print("✗ CUDA is not available")
    print("CPUモードで実行します")
    device = torch.device("cpu")
    print(f"\n使用デバイス: {device} (CPU)")

# --- CPU情報の確認 ---
print("\n【CPU情報】")
print("-" * 70)

# lscpuコマンドを実行（Linux/Mac）
print("【CPU詳細情報】")
try:
    result = subprocess.run(
        ["lscpu"], capture_output=True, text=True, check=True, timeout=5
    )
    # 重要な情報のみ抽出
    for line in result.stdout.split("\n"):
        if any(
            keyword in line
            for keyword in [
                "Model name",
                "CPU(s)",
                "Thread(s)",
                "Core(s)",
                "Socket(s)",
                "CPU MHz",
            ]
        ):
            print(line)
except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
    # Windowsまたはコマンド失敗時はpythonで取得
    print(f"CPU コア数: {psutil.cpu_count(logical=False)} (物理)")
    print(f"CPU スレッド数: {psutil.cpu_count(logical=True)} (論理)")
    print(f"CPU 使用率: {psutil.cpu_percent(interval=1)}%")

# --- メモリ情報の確認 ---
print("\n【メモリ情報】")
print("-" * 70)

ram = psutil.virtual_memory()
print(f"Total RAM: {ram.total / 1e9:.2f} GB")
print(f"Available RAM: {ram.available / 1e9:.2f} GB")
print(f"Used RAM: {ram.used / 1e9:.2f} GB")
print(f"RAM Usage: {ram.percent}%")

# GPU Memory
if torch.cuda.is_available():
    print("\nGPU Memory (Device 0):")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")


# =====================================
# 演習2: CPU vs GPU 速度比較実験
# =====================================

print("\n" + "=" * 70)
print("演習2: CPU vs GPU 速度比較実験")
print("=" * 70)

if not GPU_AVAILABLE:
    print("\n⚠️ CuPyが利用できないため、この演習はスキップされます")
    print("Google Colabなどのクラウド環境で実行してください")
else:
    print(
        """
この演習では、CPUとGPUで同じ行列積の計算を行い、速度を比較します。
- NumPy (CPU) と CuPy (GPU) を使用
- 行列サイズを変えて実験
- 計算時間と高速化率を測定
"""
    )

    # 行列サイズのリスト
    sizes = [100, 500, 1000, 2000, 5000]
    cpu_times = []
    gpu_times = []
    speedups = []

    print("\n行列積の計算速度を測定中...")
    print("-" * 70)

    for n in sizes:
        print(f"\n行列サイズ: {n} × {n}")

        # --- CPU (NumPy) ---
        A_cpu = np.random.rand(n, n).astype(np.float32)
        B_cpu = np.random.rand(n, n).astype(np.float32)

        start = time.time()
        C_cpu = A_cpu @ B_cpu
        cpu_time = time.time() - start
        cpu_times.append(cpu_time)
        print(f"  CPU (NumPy):  {cpu_time:.4f} 秒")

        # --- GPU (CuPy) ---
        A_gpu = cp.random.rand(n, n).astype(cp.float32)
        B_gpu = cp.random.rand(n, n).astype(cp.float32)

        # ウォームアップ（初回は遅いことがあるため）
        _ = A_gpu @ B_gpu
        cp.cuda.Stream.null.synchronize()

        start = time.time()
        C_gpu = A_gpu @ B_gpu
        cp.cuda.Stream.null.synchronize()  # GPU計算の完了を待つ
        gpu_time = time.time() - start
        gpu_times.append(gpu_time)
        print(f"  GPU (CuPy):   {gpu_time:.4f} 秒")

        # 高速化率
        speedup = cpu_time / gpu_time
        speedups.append(speedup)
        print(f"  高速化率:     {speedup:.1f}x")

        # 結果の検証（計算結果が正しいか確認）
        diff = np.abs(C_cpu - cp.asnumpy(C_gpu)).max()
        print(f"  最大誤差:     {diff:.2e}")

    # --- 結果の可視化 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左図: 計算時間の比較
    ax = axes[0]
    x_pos = np.arange(len(sizes))
    width = 0.35

    ax.bar(x_pos - width / 2, cpu_times, width, label="CPU (NumPy)", color="steelblue")
    ax.bar(x_pos + width / 2, gpu_times, width, label="GPU (CuPy)", color="coral")

    ax.set_xlabel("Matrix Size", fontsize=12)
    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.set_title(
        "CPU vs GPU: Matrix Multiplication Time", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{s}×{s}" for s in sizes])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_yscale("log")  # 対数スケール

    # 右図: 高速化率
    ax = axes[1]
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(sizes)))
    bars = ax.bar(range(len(sizes)), speedups, color=colors, edgecolor="black")

    ax.set_xlabel("Matrix Size", fontsize=12)
    ax.set_ylabel("Speedup (times)", fontsize=12)
    ax.set_title("GPU Speedup over CPU", fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f"{s}×{s}" for s in sizes])
    ax.grid(True, alpha=0.3, axis="y")

    # 各棒に数値を表示
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{speedup:.1f}x",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig("exercise2_cpu_gpu_comparison.png", dpi=150, bbox_inches="tight")
    print("\n図を 'exercise2_cpu_gpu_comparison.png' として保存しました")
    plt.show()

    # --- 結果のサマリー ---
    print("\n【実験結果サマリー】")
    print("-" * 70)
    print(
        f"最小高速化率: {min(speedups):.1f}x (行列サイズ: {sizes[speedups.index(min(speedups))]}×{sizes[speedups.index(min(speedups))]})"
    )
    print(
        f"最大高速化率: {max(speedups):.1f}x (行列サイズ: {sizes[speedups.index(max(speedups))]}×{sizes[speedups.index(max(speedups))]})"
    )
    print(
        "\n💡 考察: 行列サイズが小さいときはGPUのオーバーヘッド（データ転送時間等）が"
    )
    print("   相対的に大きくなるため、高速化率が低くなります。")
    print("   行列サイズが大きくなるほど、GPUの並列計算能力が発揮され、")
    print("   CPUに対して圧倒的に高速になります。")


# =====================================
# 演習3: メモリ使用量の観察
# =====================================

print("\n" + "=" * 70)
print("演習3: メモリ使用量の観察")
print("=" * 70)

print(
    """
この演習では、ニューラルネットワークのパラメータ数とメモリ使用量の関係を観察します。
- 簡単なモデルでパラメータ数を計算
- モデルサイズとメモリの関係を推定
- 大規模モデル（GPT-3等）のメモリ要件を理解
"""
)


# --- シンプルな全結合ニューラルネットワーク ---
class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def count_parameters(model):
    """モデルのパラメータ数をカウント"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_memory(num_params, dtype="float32"):
    """パラメータ数からメモリ使用量を推定"""
    bytes_per_param = {"float32": 4, "float16": 2, "int8": 1}
    bytes_size = bytes_per_param.get(dtype, 4)
    memory_bytes = num_params * bytes_size
    return memory_bytes / 1e6  # MB単位で返す


# --- モデルサイズを変えて実験 ---
configs = [
    ("小", 100, 256, 10),
    ("中", 1000, 1024, 100),
    ("大", 10000, 4096, 1000),
]

print("\n【モデルサイズとメモリ使用量】")
print("-" * 70)

results = []

for name, input_size, hidden_size, output_size in configs:
    model = SimpleModel(input_size, hidden_size, output_size)
    total, trainable = count_parameters(model)

    # メモリ使用量を推定
    memory_params = estimate_memory(total, "float32")
    memory_with_grads = memory_params * 2  # 勾配も保存するため

    results.append(
        {
            "name": name,
            "config": f"{input_size}→{hidden_size}→{output_size}",
            "params": total,
            "memory_params": memory_params,
            "memory_grads": memory_with_grads,
        }
    )

    print(f"\nモデルサイズ: {name}")
    print(f"  構成:             {input_size} → {hidden_size} → {output_size}")
    print(f"  パラメータ数:     {total:,}")
    print(f"  メモリ (重みのみ): {memory_params:.2f} MB")
    print(f"  メモリ (勾配込み): {memory_with_grads:.2f} MB")

# --- 実際のGPUメモリ使用量を測定（GPUがある場合） ---
if torch.cuda.is_available():
    print("\n【実際のGPUメモリ使用量】")

    for name, input_size, hidden_size, output_size in configs:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = SimpleModel(input_size, hidden_size, output_size).to(device)
        input_tensor = torch.randn(1, input_size).to(device)

        # Forward pass
        output = model(input_tensor)

        # メモリ使用量
        allocated = torch.cuda.memory_allocated() / 1e6
        reserved = torch.cuda.memory_reserved() / 1e6
        peak = torch.cuda.max_memory_allocated() / 1e6

        print(f"\nモデルサイズ: {name}")
        print(f"  Allocated: {allocated:.2f} MB")
        print(f"  Reserved:  {reserved:.2f} MB")
        print(f"  Peak:      {peak:.2f} MB")

# --- 大規模モデルのメモリ推定 ---
print("\n【有名な大規模モデルのメモリ要件】")
print("-" * 70)

large_models = {
    "ResNet-50": 25.6e6,
    "BERT-Base": 110e6,
    "GPT-2": 1.5e9,
    "GPT-3": 175e9,
    "GPT-4 (推定)": 1.7e12,
}

print(
    f"{'モデル名':<20} {'パラメータ数':<15} {'メモリ (FP32)':<15} {'メモリ (FP16)':<15}"
)
print("-" * 70)

for name, params in large_models.items():
    memory_fp32 = params * 4 / 1e9  # GB
    memory_fp16 = params * 2 / 1e9  # GB
    print(
        f"{name:<20} {params / 1e9:>10.1f}B {memory_fp32:>12.1f} GB {memory_fp16:>12.1f} GB"
    )

print("\n💡 考察:")
print("   - GPT-3クラスのモデルは、重みだけで700GB（FP32）必要")
print("   - 学習時は勾配やオプティマイザの状態も保存するため、さらに2-4倍必要")
print("   - そのため、複数のGPUに分散して配置する必要がある")
print("   - FP16やInt8量子化により、メモリ使用量を削減可能")

# --- 可視化 ---
fig, ax = plt.subplots(figsize=(10, 6))

model_names = list(large_models.keys())
params_billions = [large_models[name] / 1e9 for name in model_names]
memory_fp32_gb = [p * 4 / 1e9 for p in [large_models[name] for name in model_names]]

x_pos = np.arange(len(model_names))
bars = ax.bar(x_pos, memory_fp32_gb, color="steelblue", edgecolor="black")

ax.set_xlabel("Model", fontsize=12)
ax.set_ylabel("Memory (GB, FP32)", fontsize=12)
ax.set_title(
    "Memory Requirements for Large Language Models", fontsize=14, fontweight="bold"
)
ax.set_xticks(x_pos)
ax.set_xticklabels(model_names, rotation=15, ha="right")
ax.set_yscale("log")
ax.grid(True, alpha=0.3, axis="y")

# 各棒にパラメータ数を表示
for i, (bar, params) in enumerate(zip(bars, params_billions)):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{params:.1f}B",
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.tight_layout()
plt.savefig("exercise3_memory_usage.png", dpi=150, bbox_inches="tight")
print("\n図を 'exercise3_memory_usage.png' として保存しました")
plt.show()

# =====================================
# 演習4: 並列計算の効果実感
# =====================================

print("\n" + "=" * 70)
print("演習4: 並列計算の効果実感")
print("=" * 70)

print(
    """
この演習では、並列化しやすい処理と並列化しにくい処理を比較します。
- 並列化可能: 要素ごとの独立した演算
- 並列化困難: 前の結果に依存する再帰的な計算
"""
)


def parallel_friendly(n):
    """並列化しやすい処理: 要素ごとの積（各計算が独立）"""
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)

    start = time.time()
    C = A * B  # 要素ごとの積（並列化可能）
    elapsed = time.time() - start

    return elapsed


def parallel_unfriendly(n):
    """並列化しにくい処理: Fibonacci数列（前の結果に依存）"""
    result = np.zeros(n)
    result[0] = 1
    result[1] = 1

    start = time.time()
    for i in range(2, n):
        result[i] = result[i - 1] + result[i - 2]  # 前の結果に依存
    elapsed = time.time() - start

    return elapsed


# --- 実験 ---
sizes = [1000, 5000, 10000, 50000]  # , 100000

parallel_times = []
sequential_times = []
ratios = []

print("\n【並列化可能 vs 並列化困難な処理の比較】")
print("-" * 70)
print(f"{'サイズ':<12} {'並列化可能 (ms)':<18} {'逐次処理 (ms)':<18} {'比率':<10}")
print("-" * 70)

for n in sizes:
    t_parallel = parallel_friendly(n) * 1000  # ミリ秒に変換
    t_sequential = parallel_unfriendly(n) * 1000
    ratio = t_sequential / t_parallel if t_parallel > 0 else 0

    parallel_times.append(t_parallel)
    sequential_times.append(t_sequential)
    ratios.append(ratio)

    print(f"{n:<12} {t_parallel:<18.2f} {t_sequential:<18.2f} {ratio:<10.1f}x")

# --- 可視化 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左図: 実行時間の比較
ax = axes[0]
x_pos = np.arange(len(sizes))
width = 0.35

ax.bar(
    x_pos - width / 2,
    parallel_times,
    width,
    label="Parallel-friendly",
    color="green",
    alpha=0.7,
)
ax.bar(
    x_pos + width / 2,
    sequential_times,
    width,
    label="Sequential",
    color="red",
    alpha=0.7,
)

ax.set_xlabel("Data Size", fontsize=12)
ax.set_ylabel("Time (milliseconds)", fontsize=12)
ax.set_title(
    "Execution Time: Parallel-friendly vs Sequential", fontsize=14, fontweight="bold"
)
ax.set_xticks(x_pos)
ax.set_xticklabels([f"{s:,}" for s in sizes], rotation=15, ha="right")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis="y")
ax.set_yscale("log")

# 右図: 実行時間の比率
ax = axes[1]
ax.plot(sizes, ratios, "o-", linewidth=2, markersize=8, color="purple")

ax.set_xlabel("Data Size", fontsize=12)
ax.set_ylabel("Time Ratio (Sequential / Parallel)", fontsize=12)
ax.set_title("Performance Gap", fontsize=14, fontweight="bold")
ax.grid(True, alpha=0.3)

# データポイントに値を表示
for i, (size, ratio) in enumerate(zip(sizes, ratios)):
    ax.text(size, ratio, f"{ratio:.1f}x", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("exercise4_parallelization.png", dpi=150, bbox_inches="tight")
print("\n図を 'exercise4_parallelization.png' として保存しました")
plt.show()

print("\n💡 考察:")
print("   - 要素ごとの演算（A * B）は各要素が独立しているため、")
print("     NumPyは内部で自動的に並列化（SIMD命令等）を行います")
print("   - Fibonacci数列のような再帰的な計算は、前の結果に依存するため、")
print("     並列化が困難で、逐次的に計算する必要があります")
print("   - GPUはこのような「大量の独立した計算」を得意としています")

print("\n" + "=" * 70)
print("演習4 完了")
print("=" * 70)

# =====================================
# 演習のまとめ
# =====================================

print("\n" + "=" * 70)
print("演習のまとめ")
print("=" * 70)

print(
    """
本演習で学んだこと:

1. ハードウェア情報の確認
   - GPU、CPU、メモリの仕様を確認
   - 自分が使っている計算環境を理解する重要性

2. CPU vs GPU 速度比較
   - GPUは大規模な行列演算で圧倒的に高速
   - 小規模データではCPUの方が速い場合もある
   - データ転送のオーバーヘッドを考慮する必要がある

3. メモリ使用量の観察
   - パラメータ数とメモリ使用量は比例する
   - 大規模モデルには膨大なメモリが必要
   - FP16やInt8量子化でメモリを削減可能
   - 学習時は推論時の2-4倍のメモリが必要

4. 並列計算の効果
   - 独立した計算は並列化しやすい
   - 依存関係のある計算は並列化が困難
   - GPUは並列化可能な計算を大量に処理できる

【重要なポイント】
- AIモデルの学習・推論には大規模な計算リソースが必要
- GPUは行列演算に特化した並列計算装置
- データセンターは数千～数万のGPUを使用
- メモリもCPUと同様に重要なボトルネック
- ハードウェアの理解はAI開発に不可欠

【次回に向けて】
次回は、これらのハードウェアを活用した実際のAIアプリケーション
について学習します。
"""
)

print("\n演習資料の実行が完了しました")
print("\n生成された図:")
if GPU_AVAILABLE:
    print("  - exercise2_cpu_gpu_comparison.png")
print("  - exercise3_memory_usage.png")
print("  - exercise4_parallelization.png")

print("\n" + "=" * 70)
print("お疲れ様でした！")
print("=" * 70)
