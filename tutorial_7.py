"""
機械学習モデル構築・評価の演習
- データの前処理（欠損値補完、標準化）
- 複数のモデルの構築と比較
- 交差検証による性能評価
- 学習曲線の可視化と過学習の観察
"""

# ライブラリのインポート
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    KFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")

# 日本語フォントの設定（必要に応じて）
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# グラフのスタイル設定
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

print("=" * 60)
print("機械学習モデル構築・評価の演習")
print("=" * 60)

# =============================================================================
# Part 1: 回帰問題（糖尿病データセット）
# =============================================================================
print("\n" + "=" * 60)
print("Part 1: 回帰問題（糖尿病進行度の予測）")
print("=" * 60)

# データの読み込み
diabetes = load_diabetes()
X_diabetes = diabetes.data
y_diabetes = diabetes.target

# データフレームに変換（可視化のため）
df_diabetes = pd.DataFrame(X_diabetes, columns=diabetes.feature_names)
df_diabetes["target"] = y_diabetes

print("\n[1-1] データの概要")
print(f"データ形状: {df_diabetes.shape}")
print("\n最初の5行:")
print(df_diabetes.head())

print("\n基本統計量:")
print(df_diabetes.describe())

print("\n欠損値の確認:")
print(df_diabetes.isnull().sum())

# 人工的に欠損値を作成（前処理の演習のため）
print("\n[1-2] 欠損値の作成と補完の演習")
np.random.seed(42)
X_diabetes_missing = X_diabetes.copy()
# ランダムに5%のデータを欠損させる
missing_mask = np.random.random(X_diabetes_missing.shape) < 0.05
X_diabetes_missing[missing_mask] = np.nan

print(f"作成した欠損値の数: {np.isnan(X_diabetes_missing).sum()}")

# 欠損値の補完
imputer = SimpleImputer(strategy="mean")  # 平均値で補完
X_diabetes_imputed = imputer.fit_transform(X_diabetes_missing)

print(f"補完後の欠損値の数: {np.isnan(X_diabetes_imputed).sum()}")

# データの分割
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_diabetes_imputed, y_diabetes, test_size=0.2, random_state=42
)

print(f"\n訓練データ: {X_train_reg.shape}")
print(f"テストデータ: {X_test_reg.shape}")

# 標準化
print("\n[1-3] データの標準化")
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

print("標準化前の統計量（訓練データ）:")
print(
    f"  平均: {X_train_reg[:, 0].mean():.4f}, 標準偏差: {X_train_reg[:, 0].std():.4f}"
)
print("標準化後の統計量（訓練データ）:")
print(
    f"  平均: {X_train_reg_scaled[:, 0].mean():.4f}, 標準偏差: {X_train_reg_scaled[:, 0].std():.4f}"
)

# モデルの構築と評価
print("\n[1-4] 複数のモデルの構築と評価")

# 回帰モデルの定義
regression_models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=5),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=42, max_depth=5
    ),
    "SVM": SVR(kernel="rbf", C=1.0, epsilon=0.1),
}

# モデルの訓練と評価
regression_results = {}

for name, model in regression_models.items():
    print(f"\n--- {name} ---")

    # モデルの訓練
    model.fit(X_train_reg_scaled, y_train_reg)

    # 予測
    y_pred_train = model.predict(X_train_reg_scaled)
    y_pred_test = model.predict(X_test_reg_scaled)

    # 評価指標の計算
    train_mse = mean_squared_error(y_train_reg, y_pred_train)
    test_mse = mean_squared_error(y_test_reg, y_pred_test)
    train_r2 = r2_score(y_train_reg, y_pred_train)
    test_r2 = r2_score(y_test_reg, y_pred_test)

    regression_results[name] = {
        "train_mse": train_mse,
        "test_mse": test_mse,
        "train_r2": train_r2,
        "test_r2": test_r2,
    }

    print(f"訓練データ - MSE: {train_mse:.2f}, R²: {train_r2:.4f}")
    print(f"テストデータ - MSE: {test_mse:.2f}, R²: {test_r2:.4f}")

# 結果の可視化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

model_names = list(regression_results.keys())
train_mses = [regression_results[name]["train_mse"] for name in model_names]
test_mses = [regression_results[name]["test_mse"] for name in model_names]
train_r2s = [regression_results[name]["train_r2"] for name in model_names]
test_r2s = [regression_results[name]["test_r2"] for name in model_names]

x_pos = np.arange(len(model_names))
width = 0.35

# MSEの比較
axes[0].bar(x_pos - width / 2, train_mses, width, label="Train", alpha=0.8)
axes[0].bar(x_pos + width / 2, test_mses, width, label="Test", alpha=0.8)
axes[0].set_xlabel("Model")
axes[0].set_ylabel("Mean Squared Error")
axes[0].set_title("Model Comparison: MSE (Lower is Better)")
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(model_names, rotation=45, ha="right")
axes[0].legend()
axes[0].grid(axis="y", alpha=0.3)

# R²スコアの比較
axes[1].bar(x_pos - width / 2, train_r2s, width, label="Train", alpha=0.8)
axes[1].bar(x_pos + width / 2, test_r2s, width, label="Test", alpha=0.8)
axes[1].set_xlabel("Model")
axes[1].set_ylabel("R² Score")
axes[1].set_title("Model Comparison: R² Score (Higher is Better)")
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(model_names, rotation=45, ha="right")
axes[1].legend()
axes[1].axhline(y=0, color="k", linestyle="--", linewidth=0.5)
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("regression_model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# =============================================================================
# Part 2: 交差検証による性能評価
# =============================================================================
print("\n" + "=" * 60)
print("Part 2: 交差検証による性能評価")
print("=" * 60)

print("\n[2-1] k-分割交差検証（k=5）")

# 交差検証の設定
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# 各モデルで交差検証を実行
cv_results = {}

for name, model in regression_models.items():
    print(f"\n--- {name} ---")

    # 交差検証スコアの計算（負のMSEを使用）
    cv_scores_mse = -cross_val_score(
        model, X_train_reg_scaled, y_train_reg, cv=cv, scoring="neg_mean_squared_error"
    )

    # 交差検証スコアの計算（R²）
    cv_scores_r2 = cross_val_score(
        model, X_train_reg_scaled, y_train_reg, cv=cv, scoring="r2"
    )

    cv_results[name] = {"mse_scores": cv_scores_mse, "r2_scores": cv_scores_r2}

    print(f"MSE: {cv_scores_mse.mean():.2f} (+/- {cv_scores_mse.std():.2f})")
    print(f"R²:  {cv_scores_r2.mean():.4f} (+/- {cv_scores_r2.std():.4f})")
    print(f"各分割のMSE: {cv_scores_mse}")

# 交差検証結果の可視化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# MSEのボックスプロット
mse_data = [cv_results[name]["mse_scores"] for name in model_names]
bp1 = axes[0].boxplot(mse_data, labels=model_names, patch_artist=True)
axes[0].set_xlabel("Model")
axes[0].set_ylabel("Mean Squared Error")
axes[0].set_title("Cross-Validation Results: MSE")
axes[0].tick_params(axis="x", rotation=45)
axes[0].grid(axis="y", alpha=0.3)

for patch in bp1["boxes"]:
    patch.set_facecolor("lightblue")

# R²スコアのボックスプロット
r2_data = [cv_results[name]["r2_scores"] for name in model_names]
bp2 = axes[1].boxplot(r2_data, labels=model_names, patch_artist=True)
axes[1].set_xlabel("Model")
axes[1].set_ylabel("R² Score")
axes[1].set_title("Cross-Validation Results: R² Score")
axes[1].tick_params(axis="x", rotation=45)
axes[1].axhline(y=0, color="k", linestyle="--", linewidth=0.5)
axes[1].grid(axis="y", alpha=0.3)

for patch in bp2["boxes"]:
    patch.set_facecolor("lightcoral")

plt.tight_layout()
plt.savefig("cross_validation_results.png", dpi=150, bbox_inches="tight")
plt.show()

# =============================================================================
# Part 3: 学習曲線の可視化と過学習の観察
# =============================================================================
print("\n" + "=" * 60)
print("Part 3: 学習曲線の可視化と過学習の観察")
print("=" * 60)

print("\n[3-1] 学習曲線の作成")


# 学習曲線を描画する関数
def plot_learning_curve(estimator, X, y, title, cv=5):
    """
    学習曲線を描画する関数
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="neg_mean_squared_error",
    )

    # 負のMSEを正のMSEに変換
    train_scores = -train_scores
    test_scores = -test_scores

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )

    plt.xlabel("Training examples")
    plt.ylabel("Mean Squared Error")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(alpha=0.3)

    return plt


# 各モデルの学習曲線を描画
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

for idx, (name, model) in enumerate(regression_models.items()):
    print(f"学習曲線を作成中: {name}")

    train_sizes, train_scores, test_scores = learning_curve(
        model,
        X_train_reg_scaled,
        y_train_reg,
        cv=5,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="neg_mean_squared_error",
    )

    # 負のMSEを正のMSEに変換
    train_scores = -train_scores
    test_scores = -test_scores

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    axes[idx].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[idx].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[idx].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[idx].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Validation score"
    )

    axes[idx].set_xlabel("Training examples")
    axes[idx].set_ylabel("Mean Squared Error")
    axes[idx].set_title(f"Learning Curve: {name}")
    axes[idx].legend(loc="best")
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("learning_curves.png", dpi=150, bbox_inches="tight")
plt.show()

# 過学習の分析
print("\n[3-2] 過学習の観察")
print("\n各モデルの訓練スコアと検証スコアの差（最終時点）:")

for name, model in regression_models.items():
    train_sizes, train_scores, test_scores = learning_curve(
        model,
        X_train_reg_scaled,
        y_train_reg,
        cv=5,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="neg_mean_squared_error",
    )

    train_scores = -train_scores
    test_scores = -test_scores

    final_train_score = np.mean(train_scores[-1])
    final_test_score = np.mean(test_scores[-1])
    gap = final_test_score - final_train_score

    print(f"\n{name}:")
    print(f"  訓練スコア: {final_train_score:.2f}")
    print(f"  検証スコア: {final_test_score:.2f}")
    print(f"  差分: {gap:.2f}")

    if gap > final_train_score * 0.3:
        print("  → 過学習の傾向が見られます")
    elif gap < final_train_score * 0.1:
        print("  → 良好な汎化性能です")
    else:
        print("  → 適度な汎化性能です")

# =============================================================================
# Part 4: 分類問題での演習（乳がんデータセット）
# =============================================================================
print("\n" + "=" * 60)
print("Part 4: 分類問題（乳がん診断の予測）")
print("=" * 60)

# データの読み込み
cancer = load_breast_cancer()
X_cancer = cancer.data
y_cancer = cancer.target

print("\n[4-1] データの概要")
print(f"データ形状: {X_cancer.shape}")
print(f"クラス: {cancer.target_names}")
print(f"クラスの分布: {np.bincount(y_cancer)}")

# データの分割
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_cancer, y_cancer, test_size=0.2, random_state=42, stratify=y_cancer
)

# 標準化
scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)

print(f"\n訓練データ: {X_train_clf_scaled.shape}")
print(f"テストデータ: {X_test_clf_scaled.shape}")

# 分類モデルの定義
classification_models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=10000),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42, max_depth=5
    ),
    "SVM": SVC(kernel="rbf", C=1.0, random_state=42),
}

print("\n[4-2] 分類モデルの訓練と評価")

classification_results = {}

for name, model in classification_models.items():
    print(f"\n--- {name} ---")

    # モデルの訓練
    model.fit(X_train_clf_scaled, y_train_clf)

    # 予測
    y_pred_train = model.predict(X_train_clf_scaled)
    y_pred_test = model.predict(X_test_clf_scaled)

    # 評価指標の計算
    train_acc = accuracy_score(y_train_clf, y_pred_train)
    test_acc = accuracy_score(y_test_clf, y_pred_test)

    classification_results[name] = {"train_acc": train_acc, "test_acc": test_acc}

    print(f"訓練精度: {train_acc:.4f}")
    print(f"テスト精度: {test_acc:.4f}")

    if name == "Random Forest":  # Random Forestの詳細な結果を表示
        print("\n分類レポート:")
        print(
            classification_report(
                y_test_clf, y_pred_test, target_names=cancer.target_names
            )
        )

# 結果の可視化
fig, ax = plt.subplots(figsize=(10, 6))

model_names_clf = list(classification_results.keys())
train_accs = [classification_results[name]["train_acc"] for name in model_names_clf]
test_accs = [classification_results[name]["test_acc"] for name in model_names_clf]

x_pos = np.arange(len(model_names_clf))
width = 0.35

ax.bar(x_pos - width / 2, train_accs, width, label="Train", alpha=0.8)
ax.bar(x_pos + width / 2, test_accs, width, label="Test", alpha=0.8)
ax.set_xlabel("Model")
ax.set_ylabel("Accuracy")
ax.set_title("Classification Model Comparison: Accuracy")
ax.set_xticks(x_pos)
ax.set_xticklabels(model_names_clf, rotation=45, ha="right")
ax.set_ylim([0.8, 1.0])
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("classification_model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# 混同行列の可視化（Random Forestの例）
print("\n[4-3] 混同行列の可視化（Random Forest）")

rf_model = classification_models["Random Forest"]
y_pred_rf = rf_model.predict(X_test_clf_scaled)
cm = confusion_matrix(y_test_clf, y_pred_rf)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=cancer.target_names,
    yticklabels=cancer.target_names,
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix: Random Forest")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()


# print re call precision f1-score
print("\n分類レポート（Random Forest）:")
print(
    classification_report(
        y_test_clf, y_pred_rf, target_names=cancer.target_names
    )
)       

# =============================================================================
# まとめ
# =============================================================================
print("\n" + "=" * 60)
print("演習のまとめ")
print("=" * 60)

print("""
本演習では以下の内容を学習しました:

1. データの前処理
   - 欠損値の補完（SimpleImputer）
   - 標準化（StandardScaler）

2. 複数のモデルの構築と比較
   - 線形回帰、決定木、ランダムフォレスト、SVM
   - 回帰問題と分類問題の両方で実践

3. 交差検証による頑健な性能評価
   - k-分割交差検証（k=5）
   - 複数の評価指標（MSE、R²、Accuracy）

4. 学習曲線の可視化と過学習の観察
   - 訓練データ量と性能の関係
   - 訓練スコアと検証スコアの差分分析

重要なポイント:
- 前処理（欠損値補完、標準化）は性能向上に重要
- 単一のテストセットだけでなく、交差検証で評価することが推奨される
- 学習曲線は過学習の診断に有用
- 訓練スコアと検証スコアの差が大きい場合は過学習の可能性がある
""")

print("\n" + "=" * 60)
print("演習終了")
print("=" * 60)
