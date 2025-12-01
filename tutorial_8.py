"""
深層学習の発展と実用技術 - 演習資料
MDA入門 第7回

本演習では以下の内容を扱います:
1. モデルの層数と性能の関係
2. 正則化手法の効果（L1/L2ノルム）
3. バッチ正規化の効果
4. 時系列データを扱うモデル（LSTM）

必要なライブラリ:
- numpy
- matplotlib
- scikit-learn
- pytorch (or tensorflow)
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_covtype
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

# 日本語フォントの設定（必要に応じて）
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "DejaVu Sans"]

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("ライブラリのインポートが完了しました")
print(f"PyTorch version: {torch.__version__}")


# =====================================
# 演習1: モデルの層数と性能の関係
# =====================================

print("\n" + "=" * 50)
print("演習1: モデルの層数と性能の関係")
print("=" * 50)

# データセットの準備 (Covertype Data)
print("データセットを読み込んでいます...")
print("このデータセットは、カートграфі的変数のみから森林被覆タイプを予測するものです。")
covtype = fetch_covtype()
X = covtype.data
y = covtype.target - 1  # 1-7 -> 0-6

# 学習時間を抑えるためデータをダウンサンプリング
n_samples = 20000
indices = np.random.choice(len(X), n_samples, replace=False)
X = X[indices]
y = y[indices]

# 訓練データとテストデータに分割
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 特徴量の標準化
scaler = StandardScaler()
X_train_np = scaler.fit_transform(X_train_np)
X_test_np = scaler.transform(X_test_np)

# Tensorに変換
X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train_np, dtype=torch.long).to(device)
X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test_np, dtype=torch.long).to(device)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

n_features = X_train.shape[1]
n_classes = len(np.unique(y))

print(f"訓練データ: {X_train.shape}, テストデータ: {X_test.shape}")
print(f"特徴量数: {n_features}, クラス数: {n_classes}")


def create_model(n_layers, input_features, units_per_layer=64):
    """
    指定された層数のモデルを作成 (PyTorch)
    """
    layers = []
    layers.append(nn.Linear(input_features, units_per_layer))
    layers.append(nn.ReLU())

    for _ in range(n_layers - 1):
        layers.append(nn.Linear(units_per_layer, units_per_layer))
        layers.append(nn.ReLU())

    layers.append(nn.Linear(units_per_layer, n_classes))

    return nn.Sequential(*layers)


# 異なる層数でモデルを学習
layer_configs = [1, 3, 5, 7, 10]
results = []

print("\n各層数でのモデル学習を開始...")

for n_layers in layer_configs:
    print(f"\n層数: {n_layers}")

    model = create_model(n_layers, n_features).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    n_epochs = 50

    history = {"accuracy": [], "val_accuracy": []}

    for epoch in tqdm(range(n_epochs), desc=f"Training {n_layers}-layer model"):
        model.train()
        for batch_X, batch_y in train_loader:
            # batch_X = batch_X.view(-1, n_features).to(device) # Already flat
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # 性能評価 (簡易的に訓練データとテストデータを使用)
        model.eval()
        with torch.no_grad():
            # 訓練精度
            train_outputs = model(X_train)
            _, train_preds = torch.max(train_outputs, 1)
            train_acc = (train_preds == y_train).float().mean().item()
            history["accuracy"].append(train_acc)

            # 検証(テスト)精度
            test_outputs = model(X_test)
            _, test_preds = torch.max(test_outputs, 1)
            val_acc = (test_preds == y_test).float().mean().item()
            history["val_accuracy"].append(val_acc)

    test_acc = history["val_accuracy"][-1]
    results.append(
        {
            "n_layers": n_layers,
            "train_acc": history["accuracy"][-1],
            "val_acc": history["val_accuracy"][-1],
            "test_acc": test_acc,
            "history": history,
        }
    )

    print(f"  訓練精度: {history['accuracy'][-1]:.4f}")
    print(f"  検証精度: {history['val_accuracy'][-1]:.4f}")
    print(f"  テスト精度: {test_acc:.4f}")

# 結果の可視化
fig, axes = plt.subplots(1, 1, figsize=(7, 5))

# 層数と精度の関係
ax = axes
layer_nums = [r["n_layers"] for r in results]
train_accs = [r["train_acc"] for r in results]
val_accs = [r["val_acc"] for r in results]
test_accs = [r["test_acc"] for r in results]

ax.plot(layer_nums, train_accs, "o-", label="Train Accuracy", linewidth=2)
ax.plot(layer_nums, val_accs, "s-", label="Validation Accuracy", linewidth=2)
ax.plot(layer_nums, test_accs, "^-", label="Test Accuracy", linewidth=2)
ax.set_xlabel("Number of Layers", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Model Depth vs Accuracy", fontsize=14, fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)

"""# 学習曲線（5層のモデル）
ax = axes[1]
history_5layer = [r for r in results if r["n_layers"] == 5][0]["history"]
ax.plot(history_5layer["accuracy"], label="Train", linewidth=2)
ax.plot(history_5layer["val_accuracy"], label="Validation", linewidth=2)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Learning Curve (5 layers)", fontsize=14, fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)
"""
plt.tight_layout()
plt.savefig("exercise1_depth_comparison.png", dpi=150, bbox_inches="tight")
print("\n図を 'exercise1_depth_comparison.png' として保存しました")

plt.show()


# =====================================
# 演習2: 正則化手法の効果
# =====================================

print("\n" + "=" * 50)
print("演習2: 正則化手法の効果（L1/L2ノルム）")
print("=" * 50)


class RegularizedModel(nn.Module):
    def __init__(self):
        super(RegularizedModel, self).__init__()
        self.fc1 = nn.Linear(n_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, n_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


regularization_configs = [
    ("No Regularization", None, 0),
    ("L1 (lambda=0.001)", "l1", 0.001),
    ("L2 (lambda=0.01)", "l2", 0.01),
]

reg_results = []
l1_feature_importance = None

print("\n各正則化手法でのモデル学習を開始...")

for name, reg_type, reg_lambda in regularization_configs:
    print(f"\n{name}")

    model = RegularizedModel().to(device)
    l2_lambda = reg_lambda if reg_type == "l2" else 0
    optimizer = optim.Adam(model.parameters(), weight_decay=l2_lambda)
    criterion = nn.CrossEntropyLoss()
    n_epochs = 50

    history = {"accuracy": [], "val_accuracy": []}

    for epoch in tqdm(range(n_epochs), desc=f"Training {name}"):
        model.train()
        for batch_X, batch_y in train_loader:
            # batch_X = batch_X.view(-1, 784).to(device)
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # L1正則化
            if reg_type == "l1":
                l1_penalty = sum(p.abs().sum() for p in model.parameters())
                loss += reg_lambda * l1_penalty

            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_outputs = model(X_train)
            _, train_preds = torch.max(train_outputs, 1)
            train_acc = (train_preds == y_train).float().mean().item()
            history["accuracy"].append(train_acc)
            test_outputs = model(X_test)
            _, test_preds = torch.max(test_outputs, 1)
            val_acc = (test_preds == y_test).float().mean().item()
            history["val_accuracy"].append(val_acc)

    test_acc = history["val_accuracy"][-1]
    reg_results.append(
        {
            "name": name,
            "train_acc": history["accuracy"][-1],
            "val_acc": history["val_accuracy"][-1],
            "test_acc": test_acc,
            "history": history,
        }
    )

    print(f"  訓練精度: {history['accuracy'][-1]:.4f}")
    print(f"  検証精度: {history['val_accuracy'][-1]:.4f}")
    print(f"  テスト精度: {test_acc:.4f}")
    overfit_gap = history["accuracy"][-1] - history["val_accuracy"][-1]
    print(f"  過学習度: {overfit_gap:.4f}")

    if reg_type == "l1":
        # 第1層の重みを取得して特徴量の重要度とする
        weights = model.fc1.weight.detach().cpu().numpy()
        l1_feature_importance = np.abs(weights).sum(axis=0)

# 結果の可視化
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

for idx, result in enumerate(reg_results):
    ax = axes[idx]
    history = result["history"]
    ax.plot(history["accuracy"], label="Train", linewidth=2, alpha=0.8)
    ax.plot(history["val_accuracy"], label="Validation", linewidth=2, alpha=0.8)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title(result["name"], fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.0])

ax = axes[3]
names = [r["name"] for r in reg_results]
train_accs = [r["train_acc"] for r in reg_results]
test_accs = [r["test_acc"] for r in reg_results]
x = np.arange(len(names))
width = 0.35
ax.bar(x - width / 2, train_accs, width, label="Train", alpha=0.8)
ax.bar(x + width / 2, test_accs, width, label="Test", alpha=0.8)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_title("Final Accuracy Comparison", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([name.split()[0] for name in names], rotation=45, ha="right")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("exercise2_regularization.png", dpi=150, bbox_inches="tight")
print("\n図を 'exercise2_regularization.png' として保存しました")
plt.show()

# L1正則化による特徴量重要度の可視化
if l1_feature_importance is not None:
    plt.figure(figsize=(12, 9))
    plt.bar(range(len(l1_feature_importance)), l1_feature_importance)
    plt.xlabel("Feature Index", fontsize=12)
    plt.xticks(
        range(len(l1_feature_importance)),
        covtype.feature_names,
        rotation=45,
        ha="right",
        fontsize=8,
    )

    plt.ylabel("Importance (Sum of Absolute Weights)", fontsize=12)
    plt.title(
        "Feature Importance with L1 Regularization", fontsize=14, fontweight="bold"
    )

    # 特徴量の種類の境界線を表示（Covertypeデータセットの構造に基づく）
    # 0-9: 数値データ (10個)
    # 10-13: Wilderness Area (4個, One-hot)
    # 14-53: Soil Type (40個, One-hot)
    plt.axvline(x=9.5, color="r", linestyle="--", alpha=0.5)
    plt.axvline(x=13.5, color="g", linestyle="--", alpha=0.5)

    plt.text(4.5, plt.ylim()[1] * 0.95, "Quantitative", ha="center", color="r")
    plt.text(
        11.5, plt.ylim()[1] * 0.95, "Wilderness", ha="center", color="g"
    )  # , rotation=90)
    plt.text(34, plt.ylim()[1] * 0.95, "Soil Type", ha="center", color="b")

    plt.grid(True, alpha=0.3)
    plt.savefig("exercise2_l1_feature_importance.png", dpi=150, bbox_inches="tight")
    print("図を 'exercise2_l1_feature_importance.png' として保存しました")
    plt.show()

"""
# =====================================
# 演習3: バッチ正規化の効果
# =====================================

print("\n" + "=" * 50)
print("演習3: バッチ正規化の効果")
print("=" * 50)

# 演習3のためにMNISTデータを再ロード
print("MNISTデータセットを読み込んでいます...")
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
mnist_train = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
# downsize
idx = np.random.randint(0, len(mnist_train), size=30000)
mnist_train.data = mnist_train.data[idx]
mnist_train.targets = mnist_train.targets[idx]
mnist_test = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader_mnist = DataLoader(mnist_train, batch_size=256, shuffle=True)
test_loader_mnist = DataLoader(mnist_test, batch_size=1000, shuffle=False)

# グローバル変数の上書き（演習3用）
X_train_mnist = mnist_train.data.view(len(mnist_train), -1).float().to(device)
y_train_mnist = mnist_train.targets.to(device)
X_test_mnist = mnist_test.data.view(len(mnist_test), -1).float().to(device)
y_test_mnist = mnist_test.targets.to(device)

def create_model_with_batchnorm(use_batchnorm=False):
    layers = []
    input_dim = 784
    for units in [128, 64, 32, 16]:
        layers.append(nn.Linear(input_dim, units))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(units))
        layers.append(nn.ReLU())
        input_dim = units
    layers.append(nn.Linear(input_dim, 10))
    return nn.Sequential(*layers)


bn_configs = [
    ("Without Batch Normalization", False),
    ("With Batch Normalization", True),
]
bn_results = []

for name, use_bn in bn_configs:
    print(f"\n{name}")
    model = create_model_with_batchnorm(use_bn).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    history = {"accuracy": [], "val_accuracy": []}

    for epoch in tqdm(range(100), desc=f"Training {name}"):
        model.train()
        for batch_X, batch_y in train_loader_mnist:
            batch_X = batch_X.view(-1, 784).to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_outputs = model(X_train_mnist)
            _, train_preds = torch.max(train_outputs, 1)
            train_acc = (train_preds == y_train_mnist).float().mean().item()
            history["accuracy"].append(train_acc)

            test_outputs = model(X_test_mnist)
            _, test_preds = torch.max(test_outputs, 1)
            val_acc = (test_preds == y_test_mnist).float().mean().item()
            history["val_accuracy"].append(val_acc)

    test_acc = history["val_accuracy"][-1]
    bn_results.append(
        {
            "name": name,
            "train_acc": history["accuracy"][-1],
            "val_acc": history["val_accuracy"][-1],
            "test_acc": test_acc,
            "history": history,
        }
    )
    print(f"  訓練精度: {history['accuracy'][-1]:.4f}")
    print(f"  検証精度: {history['val_accuracy'][-1]:.4f}")
    print(f"  テスト精度: {test_acc:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
for result in bn_results:
    label = "With BN" if "With" in result["name"] else "Without BN"
    ax.plot(result["history"]["val_accuracy"], label=label, linewidth=2)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Validation Accuracy", fontsize=12)
ax.set_title("Effect of Batch Normalization", fontsize=14, fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
names = ["Without BN", "With BN"]
train_accs = [r["train_acc"] for r in bn_results]
val_accs = [r["val_acc"] for r in bn_results]
test_accs = [r["test_acc"] for r in bn_results]
x = np.arange(len(names))
width = 0.25
ax.bar(x - width, train_accs, width, label="Train", alpha=0.8)
ax.bar(x, val_accs, width, label="Validation", alpha=0.8)
ax.bar(x + width, test_accs, width, label="Test", alpha=0.8)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Final Accuracy Comparison", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("exercise3_batchnorm.png", dpi=150, bbox_inches="tight")
print("\n図を 'exercise3_batchnorm.png' として保存しました")
plt.show()
"""
# =====================================
# 演習3: 時系列データを扱うモデル（古典的モデル vs LSTM）
# =====================================

print("\n" + "=" * 50)
print("演習3: 時系列データを扱うモデル（古典的モデル vs LSTM）")
print("=" * 50)

# 実際のデータの準備 (Seaborn Flights Data)
# Flights data: Monthly Airline Passenger Numbers 1949-1960
print("データセットを読み込んでいます...")
flights = sns.load_dataset("flights")
data = flights["passengers"].values.astype(float)

# データの正規化
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data.reshape(-1, 1)).flatten()


def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i : i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


SEQ_LENGTH = 12  # 過去12ヶ月（1年）のデータから翌月を予測
X, y = create_sequences(data_normalized, SEQ_LENGTH)

# 訓練データとテストデータに分割 (時系列なのでシャッフルしない)
train_size = int(len(X) * 0.8)
X_train_np, X_test_np = X[:train_size], X[train_size:]
y_train_np, y_test_np = y[:train_size], y[train_size:]

# Tensorに変換 (LSTM用)
X_train_ts = torch.tensor(X_train_np, dtype=torch.float32).unsqueeze(-1).to(device)
y_train_ts = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(-1).to(device)
X_test_ts = torch.tensor(X_test_np, dtype=torch.float32).unsqueeze(-1).to(device)
y_test_ts = torch.tensor(y_test_np, dtype=torch.float32).unsqueeze(-1).to(device)

ts_train_loader = DataLoader(
    TensorDataset(X_train_ts, y_train_ts), batch_size=32, shuffle=True
)

print(f"全データ数: {len(data)}")
print(f"シーケンス生成後: {len(X)}")
print(f"訓練データ: {X_train_ts.shape}, テストデータ: {X_test_ts.shape}")

# --- 古典的モデル (Linear Regression) ---
# ※ここでは単純な線形回帰を古典的モデルとして使用（自己回帰モデルの一種とみなせる）
print("\n古典的モデル（線形回帰）の学習...")
classical_model = LinearRegression()
classical_model.fit(X_train_np, y_train_np)
y_pred_classical = classical_model.predict(X_test_np)
mae_classical = np.mean(np.abs(y_pred_classical - y_test_np))
print(f"古典的モデル - テストMAE: {mae_classical:.4f}")


# --- LSTMモデル ---
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


print("\nLSTMモデルの学習...")
lstm_model = LSTMModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.01)
n_epochs = 100
lstm_history = []

for epoch in range(n_epochs):
    lstm_model.train()
    optimizer.zero_grad()
    outputs = lstm_model(X_train_ts)
    loss = criterion(outputs, y_train_ts)
    loss.backward()
    optimizer.step()
    lstm_history.append(loss.item())

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}")

lstm_model.eval()
with torch.no_grad():
    y_pred_lstm_ts = lstm_model(X_test_ts)
    y_pred_lstm = y_pred_lstm_ts.cpu().numpy().flatten()
    mae_lstm = np.mean(np.abs(y_pred_lstm - y_test_np))
print(f"LSTMモデル - テストMAE: {mae_lstm:.4f}")

# --- 結果の可視化 ---
# 複数の予測点をサンプルして予測点を点線で結んで可視化
fig, ax = plt.subplots(figsize=(12, 6))

# テストデータ全体の正解値（元のスケールに戻す）
y_test_inv = scaler.inverse_transform(y_test_np.reshape(-1, 1)).flatten()
y_pred_classical_inv = scaler.inverse_transform(
    y_pred_classical.reshape(-1, 1)
).flatten()
y_pred_lstm_inv = scaler.inverse_transform(y_pred_lstm.reshape(-1, 1)).flatten()

# 時間軸 (テストデータの期間)
time_steps = np.arange(len(y_test_inv))

# 正解データのプロット
ax.plot(time_steps, y_test_inv, label="Actual Data", color="black", linewidth=2)

# 古典的モデルの予測
ax.plot(
    time_steps,
    y_pred_classical_inv,
    label=f"Classical (LinearReg) MAE: {mae_classical:.4f}",
    color="blue",
    linestyle="--",
    marker="o",
    markersize=4,
    alpha=0.7,
)

# LSTMモデルの予測
ax.plot(
    time_steps,
    y_pred_lstm_inv,
    label=f"LSTM MAE: {mae_lstm:.4f}",
    color="red",
    linestyle="--",
    marker="x",
    markersize=4,
    alpha=0.7,
)

ax.set_title(
    "Time Series Prediction Comparison: Classical vs LSTM (Flights Data)", fontsize=14
)
ax.set_xlabel("Time Steps (Months in Test Set)", fontsize=12)
ax.set_ylabel("Number of Passengers", fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("exercise4_timeseries_comparison.png", dpi=150, bbox_inches="tight")
print("\n図を 'exercise4_timeseries_comparison.png' として保存しました")

plt.show()

"""
# =====================================
# 演習3 (追加実験): 非線形時系列データ
# LSTMが古典的モデルより優位になるケース
# =====================================


# より複雑なデータに対応するための高性能なLSTMモデルを定義
class EnhancedLSTMModel(nn.Module):
    def __init__(
        self, input_size=1, hidden_size=100, num_layers=2, output_size=1, dropout=0.2
    ):
        super(EnhancedLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm(x)
        # 最後のタイムステップの出力のみを使用
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


print("\n" + "=" * 50)
print("演習3 (追加実験): 非線形時系列データ")
print("=" * 50)

# --- データの生成 ---
# 複数の三角関数を合成した複雑な時系列データを作成
print("複雑な非線形時系列データを生成しています...")
time = np.arange(0, 800, 0.1)
# 周期の異なるsin波と、それらを乗算した項を加えてさらに複雑化
data = (
    np.sin(time / 4) * 0.6
    + np.sin(time / 15) * 1.2
    + np.cos(time / 2) * np.sin(time / 10)  # 乗算項
    + np.random.randn(len(time)) * 0.1
)
data = data.astype(np.float32)

# データの正規化
scaler_synth = StandardScaler()
data_normalized = scaler_synth.fit_transform(data.reshape(-1, 1)).flatten()

# シーケンスの作成
SEQ_LENGTH = 150  # シーケンス長を長くして、長期依存関係をLSTMに学習させる
X, y = create_sequences(data_normalized, SEQ_LENGTH)

# 訓練データとテストデータに分割
train_size = int(len(X) * 0.8)
X_train_np, X_test_np = X[:train_size], X[train_size:]
y_train_np, y_test_np = y[:train_size], y[train_size:]

# Tensorに変換 (LSTM用)
X_train_ts = torch.tensor(X_train_np, dtype=torch.float32).unsqueeze(-1).to(device)
y_train_ts = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(-1).to(device)
X_test_ts = torch.tensor(X_test_np, dtype=torch.float32).unsqueeze(-1).to(device)
y_test_ts = torch.tensor(y_test_np, dtype=torch.float32).unsqueeze(-1).to(device)

# バッチ処理のためのDataLoaderを作成
ts_train_dataset_synth = TensorDataset(X_train_ts, y_train_ts)
ts_train_loader_synth = DataLoader(ts_train_dataset_synth, batch_size=64, shuffle=True)

print(f"全データ数: {len(data)}")
print(f"シーケンス生成後: {len(X)}")
print(f"訓練データ: {X_train_ts.shape}, テストデータ: {X_test_ts.shape}")


# --- 古典的モデル (Linear Regression) ---
print("\n古典的モデル（線形回帰）の学習...")
classical_model_synth = LinearRegression()
classical_model_synth.fit(X_train_np, y_train_np)
y_pred_classical = classical_model_synth.predict(X_test_np)
mae_classical = np.mean(np.abs(y_pred_classical - y_test_np))
print(f"古典的モデル - テストMAE: {mae_classical:.4f}")


# --- LSTMモデル ---
print("\nLSTMモデルの学習...")
# 高性能なLSTMモデルを使用
lstm_model_synth = EnhancedLSTMModel(hidden_size=100, num_layers=2, dropout=0.2).to(
    device
)
criterion = nn.MSELoss()
# 学習率を少し下げて安定させる
optimizer = optim.Adam(lstm_model_synth.parameters(), lr=0.001)
# 学習率スケジューラを追加して、学習を安定させる
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
n_epochs = 400  # エポック数を増やして十分に学習させる
lstm_history = []

for epoch in range(n_epochs):
    lstm_model_synth.train()
    epoch_loss = 0.0
    for batch_X, batch_y in ts_train_loader_synth:
        optimizer.zero_grad()
        outputs = lstm_model_synth(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    scheduler.step()  # スケジューラを更新
    avg_loss = epoch_loss / len(ts_train_loader_synth)
    lstm_history.append(avg_loss)

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {avg_loss:.4f}")

lstm_model_synth.eval()
with torch.no_grad():
    y_pred_lstm_ts = lstm_model_synth(X_test_ts)
    y_pred_lstm = y_pred_lstm_ts.cpu().numpy().flatten()
    mae_lstm = np.mean(np.abs(y_pred_lstm - y_test_np))
print(f"LSTMモデル - テストMAE: {mae_lstm:.4f}")


# --- 結果の可視化 ---
fig, ax = plt.subplots(figsize=(12, 6))

# 元のスケールに戻す
y_test_inv = scaler_synth.inverse_transform(y_test_np.reshape(-1, 1)).flatten()
y_pred_classical_inv = scaler_synth.inverse_transform(
    y_pred_classical.reshape(-1, 1)
).flatten()
y_pred_lstm_inv = scaler_synth.inverse_transform(y_pred_lstm.reshape(-1, 1)).flatten()

time_steps = np.arange(len(y_test_inv))

ax.plot(
    time_steps, y_test_inv, label="Actual Data", color="black", linewidth=2, alpha=0.8
)
ax.plot(
    time_steps,
    y_pred_classical_inv,
    label=f"Classical (LinearReg) MAE: {mae_classical:.4f}",
    color="blue",
    linestyle="--",
    marker="o",
    markersize=3,
    alpha=0.6,
)
ax.plot(
    time_steps,
    y_pred_lstm_inv,
    label=f"LSTM MAE: {mae_lstm:.4f}",
    color="red",
    linestyle="--",
    marker="x",
    markersize=3,
    alpha=0.8,
)

ax.set_title("Prediction on Complex Non-Linear Time Series", fontsize=14)
ax.set_xlabel("Time Steps (in Test Set)", fontsize=12)
ax.set_ylabel("Value", fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    "exercise4_timeseries_nonlinear_comparison.png", dpi=150, bbox_inches="tight"
)
print("\n図を 'exercise4_timeseries_nonlinear_comparison.png' として保存しました")
plt.show()
"""
# =====================================
# 演習のまとめ
# =====================================

print("\n" + "=" * 50)
print("演習のまとめ")
print("=" * 50)

print("""
本演習で学んだこと:

1. モデルの深層化
   - 層数を増やすことで表現力が向上
   - ただし、深すぎると学習が困難になる可能性

2. 正則化手法（L1/L2ノルム）
   - 過学習を防ぐための重要な技術
   - L1: スパースな解、L2: 重みを均等に小さく
   - 適切な正則化パラメータの選択が重要

3. バッチ正規化
   - 学習の安定化と高速化
   - 現代の深層学習では標準的な技術

4. 時系列データとLSTM
   - RNN/LSTMは時系列データに有効
   - 全結合層よりも時間的依存関係を捉えやすい
   - 特に、単純な線形性では捉えきれない複雑なパターンを持つデータで性能を発揮
   - Transformerなどの新しい手法も登場

これらの技術を組み合わせることで、
より高性能で安定したモデルを構築できます。
""")

print("\n演習資料の実行が完了しました")
print("生成された図:")
print("  - exercise1_depth_comparison.png")
print("  - exercise2_regularization.png")
print("  - exercise2_l1_feature_importance.png")
print("  - exercise3_batchnorm.png")
print("  - exercise4_timeseries_comparison.png")
print("  - exercise4_timeseries_nonlinear_comparison.png")
