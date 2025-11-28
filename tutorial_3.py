"""
第3回演習：統計的思考の基礎
============================

このノートブックには以下の内容が含まれています：

【演習】
1. 記述統計と確率分布（30分）
2. 標本誤差とサンプリング（25分）
3. ベイズ推定の理解（25分）

【応用課題】（時間がある人向け）
A1. Simpson's Paradox（15分）
A2. モンテカルロシミュレーション（15分）
A3. ブートストラップ法（15分）

使い方：
- Google Colabで実行してください
- TODO部分を埋めながら進めてください
- わからない場合はコメントアウトされた正解例を参考に

作成：横浜国立大学 数理・データサイエンス・AI入門
"""

# 必要なライブラリのインポート
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import beta
from sklearn.datasets import load_iris

# 乱数シード設定
np.random.seed(42)

# グラフのスタイル設定
sns.set_style("whitegrid")
sns.set_palette("husl")


# 日本語フォント設定（文字化け対策）
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


"""
print("=" * 70)
print(" 第3回演習：統計的思考の基礎 ")
print("=" * 70)
print("\n全てのライブラリが正常にインポートされました！")
print("準備完了です。演習を始めましょう！\n")

# ============================================================
# 目次の表示
# ============================================================

print("【目次】")
print("\n[演習1] 記述統計と確率分布")
print("  1.1 データセットの読み込みと探索")
print("  1.2 基本統計量の計算")
print("  1.3 ヒストグラムとガウス分布の比較")
print("  1.4 外れ値と統計量")
print("  1.5 相関係数の計算と可視化")
print("  1.6 課題：タイタニックの運賃分布の分析")

print("\n[演習2] 標本誤差とサンプリング")
print("  2.1 母集団の設定")
print("  2.2 標本抽出と標本平均")
print("  2.3 サンプルサイズの影響")
print("  2.4 中心極限定理の確認")
print("  2.5 サンプルの偏り")

print("\n[演習3] ベイズ推定の理解")
print("  3.1 病気検査問題")
print("  3.2 シミュレーションによる確認")
print("  3.3 情報の逐次更新")
print("  3.4 ベイズ推定：連続パラメータ")
print("  3.5 実践例：スパムフィルタ")
print("  3.6 課題：2回目の検査")

print("\n[応用課題]（時間がある人向け）")
print("  A1. Simpson's Paradox")
print("  A2. モンテカルロシミュレーション")
print("  A3. ブートストラップ法")
"""


"""
第3回演習1：記述統計と確率分布
==========================================
目的：
- 実データの統計量を計算する
- ガウス分布との比較を行う
- 記述統計量の選び方を理解する

所要時間：30分
"""


def section_one():
    print("=" * 60)
    print("第3回演習1：記述統計と確率分布")
    print("=" * 60)

    # ============================================================
    # 1.1 データセットの読み込みと探索
    # ============================================================
    print("\n[1.1] データセットの読み込み")
    print("-" * 60)

    # Irisデータセット
    iris = load_iris(as_frame=True)
    df_iris = iris.frame
    print("Irisデータセット（最初の5行）:")
    print(df_iris.head())

    # タイタニックデータセット
    df_titanic = sns.load_dataset("titanic")
    print("\nタイタニックデータセット（最初の5行）:")
    print(df_titanic.head())

    # ============================================================
    # 1.2 基本統計量の計算
    # ============================================================
    print("\n" + "=" * 60)
    print("[1.2] 基本統計量の計算")
    print("=" * 60)

    sepal_length_mean = df_iris["sepal length (cm)"].mean()
    sepal_length_median = df_iris["sepal length (cm)"].median()
    sepal_length_var = df_iris["sepal length (cm)"].var()
    sepal_length_std = df_iris["sepal length (cm)"].std()

    print(f"\nがく片の長さ（sepal length）の平均: {sepal_length_mean:.4f}")
    print(f"がく片の長さの中央値: {sepal_length_median:.4f}")
    print(f"がく片の長さの分散: {sepal_length_var:.4f}")
    print(f"がく片の長さの標準偏差: {sepal_length_std:.4f}")

    print("\n全統計量（describe()メソッド）:")
    print(df_iris["sepal length (cm)"].describe())
    # ← ここにコードを追加

    # ============================================================
    # 1.3 ヒストグラムとガウス分布の比較
    # ============================================================
    print("\n" + "=" * 60)
    print("[1.3] ヒストグラムとガウス分布の比較")
    print("=" * 60)

    # データ取得
    sepal_length = df_iris["sepal length (cm)"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        sepal_length,
        bins=20,
        density=True,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
        label="Actual Data",
    )

    x = np.linspace(sepal_length.min(), sepal_length.max(), 100)
    gaussian = stats.norm.pdf(x, loc=sepal_length_mean, scale=sepal_length_std)

    # ガウス分布をプロット
    ax.plot(x, gaussian, "r-", linewidth=2, label="Gaussian Distribution")

    ax.set_xlabel("Sepal Length (cm)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Sepal Length: Histogram vs Gaussian Distribution", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("exercise1_3_histogram.png", dpi=100, bbox_inches="tight")
    print("グラフを保存しました: exercise1_3_histogram.png")
    plt.show()

    # ============================================================
    # 1.4 外れ値と統計量
    # ============================================================
    print("\n" + "=" * 60)
    print("[1.4] 外れ値と統計量")
    print("=" * 60)

    # タイタニックの年齢データ（欠損値除去）
    ages = df_titanic["age"].dropna()

    print(f"\n乗客数（年齢データあり）: {len(ages)}人")

    age_mean_original = ages.mean()
    age_median_original = ages.median()

    print(f"年齢の平均: {age_mean_original:.2f}歳")
    print(f"年齢の中央値: {age_median_original:.2f}歳")

    # 最高齢者を確認
    max_age = ages.max()
    print(f"\n最高齢: {max_age}歳")

    ages_without_outlier = ages[ages < max_age]

    age_mean_without_outlier = ages_without_outlier.mean()
    age_median_without_outlier = ages_without_outlier.median()

    print("\n【外れ値除去後】")
    print(f"年齢の平均: {age_mean_without_outlier:.2f}歳")
    print(f"年齢の中央値: {age_median_without_outlier:.2f}歳")

    print("\n【変化量】")
    print(f"平均の変化: {age_mean_original - age_mean_without_outlier:.2f}歳")
    print(f"中央値の変化: {age_median_original - age_median_without_outlier:.2f}歳")

    # ============================================================
    # 1.5 相関係数の計算と可視化
    # ============================================================
    print("\n" + "=" * 60)
    print("[1.5] 相関係数の計算と可視化")
    print("=" * 60)

    print("\n相関係数行列:")
    correlation_matrix = df_iris.iloc[:, :4].corr()
    print(correlation_matrix)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
    )

    plt.title("Correlation Matrix of Iris Dataset", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig("exercise1_5_correlation.png", dpi=100, bbox_inches="tight")
    print("\nグラフを保存しました: exercise1_5_correlation.png")
    plt.show()

    pairplot = sns.pairplot(
        df_iris, hue="target", diag_kind="hist", plot_kws={"alpha": 0.6}
    )
    pairplot.savefig("exercise1_5_pairplot.png", dpi=100, bbox_inches="tight")

    print("\nグラフを保存しました: exercise1_5_pairplot.png")

    # ============================================================
    # 1.6 課題：タイタニックの運賃は正規分布に従うか？（演習ではスキップ）
    # ============================================================
    print("\n" + "=" * 60)
    print("[1.6] 課題：タイタニックの運賃分布の分析")
    print("=" * 60)

    # 運賃データ（欠損値と0を除外）
    fares = df_titanic["fare"].dropna()
    fares = fares[fares > 0]  # 運賃0を除外

    print(f"\n運賃データ数: {len(fares)}")

    # TODO 14: 運賃の基本統計量を計算してください
    print("\n【運賃の基本統計量】")
    # ← ここにコードを追加

    # TODO 15: 運賃のヒストグラムとガウス分布を描いてください
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左: 元のデータ
    ax = axes[0]
    # ← ここにコードを追加（ヒストグラムとガウス分布）
    ax.set_xlabel("Fare", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Fare Distribution (Original)", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # TODO 16: 対数変換した運賃のヒストグラムを描いてください
    ax = axes[1]
    # ヒント: np.log()で対数変換
    log_fares = None  # ← ここを修正
    # ← ここにコードを追加（対数変換後のヒストグラムとガウス分布）
    ax.set_xlabel("Log(Fare)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Fare Distribution (Log-transformed)", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("exercise1_6_fare_distribution.png", dpi=100, bbox_inches="tight")
    print("\nグラフを保存しました: exercise1_6_fare_distribution.png")
    plt.show()

    # TODO 17: 以下の質問に答えてください
    print("\n【考察課題】")
    print("Q1. 運賃は正規分布に従いますか？")
    # ← ここに考察を記入

    print("\nQ2. 従わない場合、その理由は何だと考えられますか？")
    # ← ここに考察を記入

    print("\nQ3. 対数変換するとどうなりますか？なぜそうなると思いますか？")
    # ← ここに考察を記入

    print("\n" + "=" * 60)
    print("演習1完了！お疲れ様でした。")
    print("=" * 60)


def section_two():
    """
    第3回演習2：標本誤差とサンプリング
    ==========================================
    目的：
    - 標本誤差を体験的に理解する
    - 中心極限定理を確認する
    - サンプルサイズの重要性を学ぶ

    所要時間：25分
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    # 乱数シード設定（再現性のため）
    np.random.seed(42)

    # グラフのスタイル設定
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    print("=" * 60)
    print("第3回演習2：標本誤差とサンプリング")
    print("=" * 60)

    # ============================================================
    # 2.1 母集団の設定
    # ============================================================
    print("\n[2.1] 母集団の設定")
    print("-" * 60)

    # 母集団：日本人男性の身長（仮想データ）
    # 平均171cm、標準偏差6cm、N=100,000
    population = np.random.normal(loc=171, scale=6, size=100000)

    population_mean = population.mean()
    population_std = population.std()

    print(f"\n母集団サイズ: {len(population):,}人")
    print(f"母平均: {population_mean:.2f} cm")
    print(f"母標準偏差: {population_std:.2f} cm")

    # 母集団のヒストグラム
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        population,
        bins=50,
        density=True,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
        label="Population",
    )
    ax.axvline(
        population_mean,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Population Mean: {population_mean:.2f}",
    )
    ax.set_xlabel("Height (cm)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Population Distribution (Japanese Male Height)", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("exercise2_1_population.png", dpi=100, bbox_inches="tight")
    print("グラフを保存しました: exercise2_1_population.png")
    plt.show()

    # ============================================================
    # 2.2 標本抽出と標本平均
    # ============================================================
    print("\n" + "=" * 60)
    print("[2.2] 標本抽出と標本平均")
    print("=" * 60)

    sample_size = 30
    sample = np.random.choice(population, size=sample_size, replace=False)
    sample_mean = sample.mean()

    print(f"\nサンプルサイズ: {sample_size}")
    print(f"標本平均: {sample_mean:.2f} cm")
    print(f"母平均との差（標本誤差）: {sample_mean - population_mean:.2f} cm")

    n_simulations = 1000
    sample_means = []

    for i in range(n_simulations):
        sample = np.random.choice(population, size=sample_size, replace=False)
        sample_means.append(sample.mean())

    sample_means = np.array(sample_means)

    print(f"\n{n_simulations}回のシミュレーション結果:")
    print(f"標本平均の平均: {sample_means.mean():.2f} cm")
    print(f"標本平均の標準偏差（標準誤差）: {sample_means.std():.2f} cm")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        sample_means,
        bins=30,
        density=False,
        alpha=0.7,
        color="lightcoral",
        edgecolor="black",
        label="Sample Means",
    )
    ax.axvline(
        population_mean,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Population Mean: {population_mean:.2f}",
    )
    ax.axvline(
        sample_means.mean(),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Mean of Sample Means: {sample_means.mean():.2f}",
    )
    ax.set_xlabel("Sample Mean (cm)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(
        f"Distribution of Sample Means (n={sample_size}, {n_simulations} simulations)",
        fontsize=14,
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("exercise2_2_sample_means.png", dpi=100, bbox_inches="tight")
    print("\nグラフを保存しました: exercise2_2_sample_means.png")
    plt.show()

    # ============================================================
    # 2.3 サンプルサイズの影響
    # ============================================================
    print("\n" + "=" * 60)
    print("[2.3] サンプルサイズの影響")
    print("=" * 60)

    sample_sizes = [10, 30, 100, 300]
    results = {}

    for n in sample_sizes:
        # TODO 6: サンプルサイズnで1000回シミュレーション
        sample_means_n = []
        # ← ここにコードを追加
        for i in range(1000):
            sample = np.random.choice(population, size=n, replace=False)
            sample_means_n.append(sample.mean())

        sample_means_n = np.array(sample_means_n)

        # 標準誤差（標本平均の標準偏差）
        standard_error_empirical = sample_means_n.std()

        # TODO 7: 理論値 σ/√n を計算してください
        standard_error_theoretical = population_std / np.sqrt(n)

        results[n] = {
            "empirical": standard_error_empirical,
            "theoretical": standard_error_theoretical,
            "sample_means": sample_means_n,
        }

        print(f"\nn = {n}")
        print(f"  標準誤差（実測値）: {standard_error_empirical:.4f}")
        print(f"  標準誤差（理論値）: {standard_error_theoretical:.4f}")
        print(f"  差: {abs(standard_error_empirical - standard_error_theoretical):.4f}")

    # TODO 8: 4つのサンプルサイズの結果を並べて可視化してください
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, n in enumerate(sample_sizes):
        ax = axes[idx]
        sample_means_n = results[n]["sample_means"]
        se_empirical = results[n]["empirical"]
        se_theoretical = results[n]["theoretical"]

        # ← ここにコードを追加（ヒストグラムを描画）
        ax.hist(sample_means_n, bins=30, alpha=0.7, color="skyblue", edgecolor="black")

        ax.axvline(
            population_mean,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Population Mean",
        )
        ax.set_xlim(left=164, right=178)
        ax.set_xlabel("Sample Mean (cm)", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title(
            f"n = {n}, SE = {se_empirical:.3f} (Theoretical: {se_theoretical:.3f})",
            fontsize=12,
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Effect of Sample Size on Standard Error", fontsize=16, y=1.00)
    plt.tight_layout()
    plt.savefig("exercise2_3_sample_size_effect.png", dpi=100, bbox_inches="tight")
    print("\nグラフを保存しました: exercise2_3_sample_size_effect.png")
    plt.show()

    # TODO 9: 標準誤差とサンプルサイズの関係をプロットしてください
    fig, ax = plt.subplots(figsize=(10, 6))

    sample_sizes_array = np.array(sample_sizes)
    se_empirical_array = [results[n]["empirical"] for n in sample_sizes]
    se_theoretical_array = [results[n]["theoretical"] for n in sample_sizes]

    # ← ここにコードを追加（実測値と理論値をプロット）
    ax.plot(
        sample_sizes_array,
        se_empirical_array,
        "o-",
        linewidth=2,
        markersize=8,
        label="Empirical SE",
    )
    ax.plot(
        sample_sizes_array,
        se_theoretical_array,
        "s--",
        linewidth=2,
        markersize=8,
        label="Theoretical SE (σ/√n)",
    )
    ax.set_xlabel("Sample Size (n)", fontsize=12)
    ax.set_ylabel("Standard Error", fontsize=12)
    ax.set_title("Relationship between Sample Size and Standard Error", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("exercise2_3_se_vs_n.png", dpi=100, bbox_inches="tight")
    print("グラフを保存しました: exercise2_3_se_vs_n.png")
    plt.show()

    print("\n【考察】")
    print("Q1. サンプルサイズを4倍にすると、標準誤差は何倍になりますか？")
    # ← ここに考察を記入

    print("\nQ2. 実際の調査で信頼できるサンプルサイズの目安は？")
    # ← ここに考察を記入

    # ============================================================
    # 2.4 中心極限定理の確認（演習では扱わない）
    # ============================================================
    print("\n" + "=" * 60)
    print("[2.4] 中心極限定理の確認")
    print("=" * 60)

    # 母集団を一様分布に変更
    population_uniform = np.random.uniform(0, 10, size=100000)

    print("\n一様分布の母集団:")
    print(f"母平均: {population_uniform.mean():.2f}")
    print(f"母標準偏差: {population_uniform.std():.2f}")

    # TODO 10: サンプルサイズn=30で標本抽出を1000回繰り返してください
    sample_size_clt = 30
    sample_means_uniform = []

    # ← ここにコードを追加

    sample_means_uniform = np.array(sample_means_uniform)

    # TODO 11: 母集団と標本平均の分布を並べて可視化してください
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左: 母集団（一様分布）
    ax = axes[0]
    # ← ここにコードを追加

    ax.set_xlabel("Value", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Population Distribution (Uniform)", fontsize=14)
    ax.grid(True, alpha=0.3)

    # 右: 標本平均の分布
    ax = axes[1]
    # ← ここにコードを追加

    # ガウス分布を重ねる
    x = np.linspace(sample_means_uniform.min(), sample_means_uniform.max(), 100)
    gaussian = stats.norm.pdf(
        x, loc=sample_means_uniform.mean(), scale=sample_means_uniform.std()
    )
    ax.plot(x, gaussian, "r-", linewidth=2, label="Gaussian Fit")

    ax.set_xlabel("Sample Mean", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Distribution of Sample Means (n={sample_size_clt})", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Central Limit Theorem: Uniform Distribution", fontsize=16, y=1.00)
    plt.tight_layout()
    plt.savefig("exercise2_4_central_limit_theorem.png", dpi=100, bbox_inches="tight")
    print("\nグラフを保存しました: exercise2_4_central_limit_theorem.png")
    plt.show()

    # 参考：正解例（コメントアウト）
    """
    for i in range(1000):
        sample = np.random.choice(population_uniform, size=sample_size_clt, replace=False)
        sample_means_uniform.append(sample.mean())

    axes[0].hist(population_uniform, bins=50, density=True, alpha=0.7,
                color='lightgreen', edgecolor='black', label='Uniform Distribution')

    axes[1].hist(sample_means_uniform, bins=30, density=True, alpha=0.7,
                color='lightcoral', edgecolor='black', label='Sample Means')
    """

    print("\n【考察】中心極限定理の驚き")
    print(
        "母集団が一様分布（矩形）なのに、標本平均の分布はガウス分布（釣鐘型）になる！"
    )
    print("これが中心極限定理の威力です。")

    # ============================================================
    # 2.5 サンプルの偏り
    # ============================================================
    print("\n" + "=" * 60)
    print("[2.5] サンプルの偏り")
    print("=" * 60)

    print("\nシナリオ: 電話調査で高齢者に偏る")
    print("母集団: 20-80歳、一様分布")
    print("偏ったサンプリング: 50歳以上が選ばれる確率が60%高い")

    # 年齢の母集団（20-80歳、一様分布）
    population_age = np.random.uniform(20, 80, size=100000)
    population_age_mean = population_age.mean()

    print(f"\n母集団の平均年齢: {population_age_mean:.2f}歳")

    # TODO 12: ランダムサンプリング（偏りなし）
    n_samples = 1000
    sample_size_bias = 100
    random_sample_means = []

    for i in range(n_samples):
        # ← ここにコードを追加（通常のランダムサンプリング）
        # ランダムサンプリング
        for i in range(n_samples):
            sample = np.random.choice(
                population_age, size=sample_size_bias, replace=False
            )
            random_sample_means.append(sample.mean())
        pass

    random_sample_means = np.array(random_sample_means)

    # TODO 13: 偏ったサンプリング
    biased_sample_means = []

    for i in range(n_samples):
        # 50歳以上のサンプルに重みをつける
        # ヒント: np.where()を使って条件分岐
        # ← ここにコードを追加
        # 偏ったサンプリング
        for i in range(n_samples):
            # 50歳以上に60%高い確率を与える
            weights = np.where(population_age >= 50, 1.6, 1.0)
            weights = weights / weights.sum()
            sample = np.random.choice(
                population_age, size=sample_size_bias, replace=False, p=weights
            )
            biased_sample_means.append(sample.mean())
        pass

    biased_sample_means = np.array(biased_sample_means)

    print(f"\nランダムサンプリングの標本平均: {random_sample_means.mean():.2f}歳")
    print(f"偏ったサンプリングの標本平均: {biased_sample_means.mean():.2f}歳")
    print(
        f"母平均との差（バイアス）: {biased_sample_means.mean() - population_age_mean:.2f}歳"
    )

    # TODO 14: ランダムと偏ったサンプリングを比較して可視化してください
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左: ランダムサンプリング
    ax = axes[0]
    # ← ここにコードを追加（ヒストグラムを描画）
    axes[0].hist(
        random_sample_means, bins=30, alpha=0.7, color="skyblue", edgecolor="black"
    )

    ax.axvline(
        population_age_mean,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Population Mean: {population_age_mean:.2f}",
    )
    ax.axvline(
        random_sample_means.mean(),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Sample Mean: {random_sample_means.mean():.2f}",
    )
    ax.set_xlabel("Sample Mean Age", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Random Sampling (Unbiased)", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 右: 偏ったサンプリング
    ax = axes[1]
    # ← ここにコードを追加（ヒストグラムを描画）
    axes[1].hist(
        biased_sample_means, bins=30, alpha=0.7, color="salmon", edgecolor="black"
    )

    ax.axvline(
        population_age_mean,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Population Mean: {population_age_mean:.2f}",
    )
    ax.axvline(
        biased_sample_means.mean(),
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Sample Mean: {biased_sample_means.mean():.2f}",
    )
    ax.set_xlabel("Sample Mean Age", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Biased Sampling (Older people overrepresented)", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Effect of Sampling Bias", fontsize=16, y=1.00)
    plt.tight_layout()
    plt.savefig("exercise2_5_sampling_bias.png", dpi=100, bbox_inches="tight")
    print("\nグラフを保存しました: exercise2_5_sampling_bias.png")
    plt.show()

    print("\n【考察】")
    print("Q1. サンプルの偏りは標本平均にどのような影響を与えますか？")
    # ← ここに考察を記入

    print("\nQ2. 実際の調査でサンプルの偏りを防ぐにはどうすればよいですか？")
    # ← ここに考察を記入

    print("\n" + "=" * 60)
    print("演習2完了！お疲れ様でした。")
    print("=" * 60)


def section_three():
    """
    第3回演習3：ベイズ推定の理解
    ==========================================
    目的：
    - 条件付き確率を復習する
    - ベイズの定理を直感的に理解する
    - 情報の更新プロセスを体験する

    所要時間：25分
    """
    # ============================================================
    # 3.1 病気検査問題（古典的な例）
    # ============================================================
    print("\n[3.1] 病気検査問題")
    print("-" * 60)

    print("\n【設定】")
    print("- 有病率（事前確率）: 1%")
    print("- 感度（真陽性率）: 99%")
    print("  → 病気の人の99%が陽性と判定される")
    print("- 特異度（真陰性率）: 95%")
    print("  → 健康な人の95%が陰性と判定される")

    # パラメータ
    prior = 0.01  # 有病率（事前確率）
    sensitivity = 0.99  # 感度
    specificity = 0.95  # 特異度

    def bayesian_disease_test(prior, sensitivity, specificity):
        """
        ベイズの定理で事後確率を計算

        P(病気|陽性) = P(陽性|病気) * P(病気) / P(陽性)

        P(陽性) = P(陽性|病気) * P(病気) + P(陽性|健康) * P(健康)
        """
        # TODO: ここを完成させてください
        # ヒント1: P(陽性|病気) = sensitivity
        # ヒント2: P(陽性|健康) = 1 - specificity（偽陽性率）

        p_positive_given_disease = sensitivity
        p_positive_given_healthy = 1 - specificity

        # 全確率の法則でP(陽性)を計算
        p_positive = p_positive_given_disease * prior + p_positive_given_healthy * (
            1 - prior
        )

        # ベイズの定理
        p_disease_given_positive = (p_positive_given_disease * prior) / p_positive

        return p_disease_given_positive

    # 計算実行
    posterior = bayesian_disease_test(prior, sensitivity, specificity)

    print("\n【質問】陽性判定が出たとき、実際に病気である確率は？")
    print("\n【あなたの直感】: _____ %（直感で答えてから下を見てください）")

    print("\n" + "." * 60)
    print(f"\n【正解】: {posterior * 100:.2f}%")
    print("\n衝撃的な結果ですね！なぜこんなに低いのでしょうか？")

    # ============================================================
    # 3.2 シミュレーションによる確認
    # ============================================================
    print("\n" + "=" * 60)
    print("[3.2] シミュレーションによる確認")
    print("=" * 60)

    # TODO 2: 10,000人の仮想集団でシミュレートしてください
    n_people = 10000

    # 病気かどうか（1%が病気）
    has_disease = np.random.rand(n_people) < prior

    # 検査結果をシミュレート
    test_result = np.zeros(n_people, dtype=bool)

    # TODO 3: 感度と特異度に基づいて検査結果を生成してください
    # ヒント: 病気の人は感度の確率で陽性、健康な人は(1-特異度)の確率で陽性
    # ← ここにコードを追加
    for i in range(n_people):
        if has_disease[i]:
            # 病気の人は感度の確率で陽性
            test_result[i] = np.random.rand() < sensitivity
        else:
            # 健康な人は(1-特異度)の確率で陽性（偽陽性）
            test_result[i] = np.random.rand() < (1 - specificity)

    # 陽性者のうち実際に病気の人の割合
    positive_indices = test_result
    n_positive = positive_indices.sum()
    n_true_positive = (has_disease & positive_indices).sum()
    n_false_positive = (~has_disease & positive_indices).sum()

    empirical_posterior = n_true_positive / n_positive if n_positive > 0 else 0

    print(f"\n【シミュレーション結果】（n={n_people}）")
    print(f"陽性判定者数: {n_positive}人")
    print(f"  真陽性（病気で陽性）: {n_true_positive}人")
    print(f"  偽陽性（健康で陽性）: {n_false_positive}人")
    print(f"\n陽性者のうち実際に病気の人の割合: {empirical_posterior * 100:.2f}%")
    print(f"ベイズの定理による計算値: {posterior * 100:.2f}%")
    print(f"差: {abs(empirical_posterior - posterior) * 100:.2f}%")

    # TODO 4: 結果を可視化してください
    fig, ax = plt.subplots(figsize=(10, 8))

    # 4つのカテゴリーを集計
    true_positive = (has_disease & test_result).sum()
    false_positive = (~has_disease & test_result).sum()
    false_negative = (has_disease & ~test_result).sum()
    true_negative = (~has_disease & ~test_result).sum()

    categories = [
        "True Positive\n(Sick & Positive)",
        "False Positive\n(Healthy & Positive)",
        "False Negative\n(Sick & Negative)",
        "True Negative\n(Healthy & Negative)",
    ]
    counts = [true_positive, false_positive, false_negative, true_negative]
    colors = ["#ff6b6b", "#ffa500", "#4ecdc4", "#95e1d3"]

    ax.bar(categories, counts, color=colors, edgecolor="black", linewidth=1.5)

    ax.set_ylabel("Number of People", fontsize=12)
    ax.set_title(f"Disease Test Simulation Results (n={n_people})", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    # 注釈を追加
    for i, (cat, count) in enumerate(zip(categories, counts)):
        ax.text(i, count + 50, f"{count}", ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig("exercise3_2_simulation.png", dpi=100, bbox_inches="tight")
    print("\nグラフを保存しました: exercise3_2_simulation.png")
    plt.show()

    print("\n【なぜこうなるのか？】")
    print("健康な人が圧倒的に多い（9,900人 vs 100人）")
    print("→ 偽陽性（健康なのに陽性）が多数発生")
    print("→ 陽性者の多くは実は健康")

    # ============================================================
    # 3.3 情報の逐次更新
    # ============================================================
    print("\n" + "=" * 60)
    print("[3.3] 情報の逐次更新（ベイズ更新）")
    print("=" * 60)

    print("\n【コイン投げ問題】")
    print("3種類のコインがあります:")
    print("  コインA: 表の出る確率 0.3")
    print("  コインB: 表の出る確率 0.5")
    print("  コインC: 表の出る確率 0.7")
    print("\nどれか1つを選んで投げますが、どのコインかはわかりません。")

    # 3種類のコインの表が出る確率
    coin_probs = np.array([0.3, 0.5, 0.7])
    coin_names = ["Coin A (p=0.3)", "Coin B (p=0.5)", "Coin C (p=0.7)"]

    # 事前確率：均等（各1/3）
    prior_coins = np.array([1 / 3, 1 / 3, 1 / 3])

    print(f"\n事前確率: {prior_coins}")

    # データ：コインを投げた結果（H=表、T=裏）
    data = ["H", "H", "T", "H", "T", "H", "H"]
    print(f"\nデータ: {' → '.join(data)}")

    # TODO 5: データが1つ観測されるごとに事後確率を更新してください
    posteriors_history = [prior_coins.copy()]

    current_posterior = prior_coins.copy()

    for observation in data:
        # ベイズ更新
        # TODO: 尤度（likelihood）を計算
        # ヒント: 表(H)が出たら各コインの確率、裏(T)が出たら(1-確率)
        if observation == "H":
            likelihood = coin_probs
        else:
            likelihood = 1 - coin_probs

        # TODO: 事後確率を計算（ベイズの定理）
        # 事後確率 ∝ 尤度 × 事前確率
        numerator = likelihood * current_posterior
        current_posterior = numerator / numerator.sum()

        posteriors_history.append(current_posterior.copy())

    posteriors_history = np.array(posteriors_history)

    # 結果表示
    print("\n【ベイズ更新の過程】")
    for i, obs in enumerate(["Prior"] + data):
        print(f"{i}. {obs:6s} → ", end="")
        for j, name in enumerate(coin_names):
            print(f"{name}: {posteriors_history[i][j]:.3f}  ", end="")
        print()

    # TODO 6: ベイズ更新のプロセスを可視化してください
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(data) + 1)

    for i, name in enumerate(coin_names):
        ax.plot(
            x,
            posteriors_history[:, i],
            marker="o",
            linewidth=2,
            markersize=8,
            label=name,
        )

    ax.set_xlabel("Number of Observations", fontsize=12)
    ax.set_ylabel("Posterior Probability", fontsize=12)
    ax.set_title("Bayesian Update: Coin Identification", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(["Prior"] + [f"{i + 1}:{obs}" for i, obs in enumerate(data)])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("exercise3_3_bayesian_update.png", dpi=100, bbox_inches="tight")
    print("\nグラフを保存しました: exercise3_3_bayesian_update.png")
    plt.show()

    print("\n【考察】")
    print("最初は全てのコインが同じ確率（1/3）でした。")
    print("データを見るたびに、どのコインかの確信が強まっていきます。")
    print(f"最終的に、{coin_names[current_posterior.argmax()]} の可能性が最も高い。")

    # ============================================================
    # 3.4 ベイズ推定：連続パラメータの場合
    # ============================================================
    print("\n" + "=" * 60)
    print("[3.4] ベイズ推定：コインの表が出る確率θの推定")
    print("=" * 60)

    print("\n【設定】")
    print("コインの表が出る確率θを推定したい（θは0〜1の連続値）")
    print("事前分布: θ ~ Beta(1, 1)（一様分布、何も知らない状態）")
    print("データ: 10回投げて7回表が出た")

    # 事前分布のパラメータ
    alpha_prior = 1
    beta_prior = 1

    # データ
    n_tosses = 10
    n_heads = 7

    print(f"\n10回中{n_heads}回表が出ました。θの値は？")

    # TODO 7: ベイズ更新後の事後分布を計算してください
    # ベータ分布の共役性により、事後分布もベータ分布
    # Beta(α + 表の回数, β + 裏の回数)
    alpha_posterior = alpha_prior + n_heads
    beta_posterior = beta_prior + (n_tosses - n_heads)

    # TODO 8: 事前分布と事後分布を可視化してください
    theta = np.linspace(0, 1, 1000)

    # 事前分布
    prior_dist = beta.pdf(theta, alpha_prior, beta_prior)

    # 事後分布
    posterior_dist = beta.pdf(theta, alpha_posterior, beta_posterior)
    bayesian_estimate = alpha_posterior / (alpha_posterior + beta_posterior)

    # 最尤推定（単純な頻度）
    mle = n_heads / n_tosses

    # ベイズ推定（事後分布の平均）

    fig, ax = plt.subplots(figsize=(12, 6))

    # ← ここにコードを追加（事前分布と事後分布をプロット）
    ax.plot(theta, prior_dist, linewidth=2, label="Prior: Beta(1, 1)")
    ax.plot(theta, posterior_dist, linewidth=2, label="Posterior: Beta(8, 4)")
    ax.fill_between(theta, posterior_dist, alpha=0.3)

    ax.axvline(
        mle,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"MLE (Maximum Likelihood): {mle:.2f}",
    )
    ax.axvline(
        bayesian_estimate,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Bayesian Estimate (Mean): {bayesian_estimate:.2f}",
    )

    ax.set_xlabel("θ (Probability of Heads)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.set_title(
        f"Bayesian Parameter Estimation (n={n_tosses}, heads={n_heads})", fontsize=14
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("exercise3_4_bayesian_estimation.png", dpi=100, bbox_inches="tight")
    print("\nグラフを保存しました: exercise3_4_bayesian_estimation.png")
    plt.show()

    print("\n【推定結果】")
    print(f"最尤推定（MLE）: {mle:.3f}")
    print(f"ベイズ推定: {bayesian_estimate:.3f}")
    print(
        f"\n事後分布の95%信頼区間: [{beta.ppf(0.025, alpha_posterior, beta_posterior):.3f}, "
        f"{beta.ppf(0.975, alpha_posterior, beta_posterior):.3f}]"
    )

    # ============================================================
    # 3.5 実践例：スパムフィルタ（ナイーブベイズ）
    # ============================================================
    print("\n" + "=" * 60)
    print("[3.5] 実践例：スパムフィルタ")
    print("=" * 60)

    print("\n【設定】")
    print("単語の出現パターンからスパム判定")
    print("単語「無料」の出現確率:")
    print("  - スパムメール: 80%")
    print("  - 通常メール: 5%")
    print("スパム率: 20%")

    # パラメータ
    p_spam = 0.20  # スパム率
    p_free_given_spam = 0.80  # スパムで「無料」が出る確率
    p_free_given_ham = 0.05  # 通常メールで「無料」が出る確率

    # TODO 9: 「無料」を含むメールがスパムである確率を計算してください
    def naive_bayes_spam(p_spam, p_word_given_spam, p_word_given_ham):
        """
        単純ベイズでスパム確率を計算
        """
        # TODO: ベイズの定理を適用
        # ← ここにコードを追加
        pass

    p_spam_given_free = naive_bayes_spam(p_spam, p_free_given_spam, p_free_given_ham)

    print(f"\n「無料」を含むメールがスパムである確率: {p_spam_given_free * 100:.2f}%")

    # 参考：正解例（コメントアウト）
    """
    def naive_bayes_spam(p_spam, p_word_given_spam, p_word_given_ham):
        p_ham = 1 - p_spam
        p_word = p_word_given_spam * p_spam + p_word_given_ham * p_ham
        p_spam_given_word = (p_word_given_spam * p_spam) / p_word
        return p_spam_given_word
    """

    # TODO 10: 複数の単語がある場合
    print("\n【発展】複数の単語がある場合")
    print("メール内容: 「無料」「プレゼント」")
    print("  「プレゼント」の出現確率:")
    print("    - スパム: 70%")
    print("    - 通常: 10%")

    p_present_given_spam = 0.70
    p_present_given_ham = 0.10

    # ナイーブベイズの仮定：単語は独立
    # P(スパム|無料,プレゼント) ∝ P(無料|スパム) * P(プレゼント|スパム) * P(スパム)

    # ← ここにコードを追加（複数単語の場合の計算）

    print("\n【考察】")
    print("単語が増えるほど、分類の精度が上がります。")
    print("これが実際のスパムフィルタの基本原理です。")

    # ============================================================
    # 3.6 課題：2回目の検査
    # ============================================================
    print("\n" + "=" * 60)
    print("[3.6] 課題：2回目の検査")
    print("=" * 60)

    print("\n【課題】")
    print("病気検査で1回目が陽性でした（事後確率17%）。")
    print("2回目の検査も陽性だと、確率はどう変わるでしょうか？")

    # TODO 11: 1回目の事後確率を2回目の事前確率として使用
    first_posterior = bayesian_disease_test(prior, sensitivity, specificity)
    print(f"\n1回目の事後確率: {first_posterior * 100:.2f}%")

    # 2回目の検査（1回目の事後確率を事前確率に）
    second_posterior = None  # ← ここを修正
    second_posterior = bayesian_disease_test(first_posterior, sensitivity, specificity)

    print(f"2回目の事後確率: {second_posterior * 100:.2f}%")

    # TODO 12: 3回目、4回目...と繰り返すとどうなるか
    posteriors_sequence = [prior]
    current = prior

    for i in range(10):
        current = bayesian_disease_test(current, sensitivity, specificity)
        posteriors_sequence.append(current)

    # ← ここにコードを追加（繰り返し検査の結果を可視化）

    print("\n" + "=" * 60)
    print("演習3完了！お疲れ様でした。")
    print("=" * 60)


def additional_exercises():
    """
    応用課題1：Simpson's Paradox（シンプソンのパラドックス）
    ==========================================================
    概要：
    全体では逆の傾向が、グループ別に見ると一貫した傾向になる現象

    有用性：
    - 因果推論における交絡因子の理解
    - データの見方による結論の違いを体験
    - 集計のレベル（個別 vs 全体）の重要性

    所要時間：15分
    """
    print("=" * 60)
    print("応用課題1：Simpson's Paradox")
    print("=" * 60)

    # ============================================================
    # 実例：大学入試の男女差別疑惑（Berkeley入試データに基づく）
    # ============================================================

    print("\n【設定】某大学の入試データ")
    print("男性と女性、どちらが合格しやすいか？")

    # データ作成
    # 学科A：難易度低、女性が多く志願
    # 学科B：難易度高、男性が多く志願

    data = {
        "学科": ["A"] * 1000 + ["B"] * 1000,
        "性別": ["女性"] * 800 + ["男性"] * 200 + ["女性"] * 200 + ["男性"] * 800,
        "合格": (
            # 学科A：女性800人中560人合格（70%）
            [1] * 560
            + [0] * 240
            +
            # 学科A：男性200人中140人合格（70%）
            [1] * 140
            + [0] * 60
            +
            # 学科B：女性200人中60人合格（30%）
            [1] * 60
            + [0] * 140
            +
            # 学科B：男性800人中240人合格（30%）
            [1] * 240
            + [0] * 560
        ),
    }

    df = pd.DataFrame(data)

    print(f"\n総志願者数: {len(df)}人")
    print(f"  女性: {(df['性別'] == '女性').sum()}人")
    print(f"  男性: {(df['性別'] == '男性').sum()}人")

    # ============================================================
    # 全体での合格率
    # ============================================================

    print("\n" + "=" * 60)
    print("【全体での合格率】")
    print("=" * 60)

    overall_stats = df.groupby("性別")["合格"].agg(["sum", "count", "mean"])
    overall_stats.columns = ["合格者数", "志願者数", "合格率"]
    overall_stats["合格率(%)"] = overall_stats["合格率"] * 100

    print("\n" + str(overall_stats))

    print(f"\n→ 女性の合格率: {overall_stats.loc['女性', '合格率(%)']:.1f}%")
    print(f"→ 男性の合格率: {overall_stats.loc['男性', '合格率(%)']:.1f}%")
    print("\n結論：女性の方が合格率が高い！")

    # ============================================================
    # 学科別での合格率
    # ============================================================

    print("\n" + "=" * 60)
    print("【学科別での合格率】")
    print("=" * 60)

    dept_stats = (
        df.groupby(["学科", "性別"])["合格"].agg(["sum", "count", "mean"]).reset_index()
    )
    dept_stats.columns = ["学科", "性別", "合格者数", "志願者数", "合格率"]
    dept_stats["合格率(%)"] = dept_stats["合格率"] * 100

    print("\n" + str(dept_stats))

    for dept in ["A", "B"]:
        dept_data = dept_stats[dept_stats["学科"] == dept]
        female_rate = dept_data[dept_data["性別"] == "女性"]["合格率(%)"].values[0]
        male_rate = dept_data[dept_data["性別"] == "男性"]["合格率(%)"].values[0]
        print(f"\n学科{dept}:")
        print(f"  女性: {female_rate:.1f}%")
        print(f"  男性: {male_rate:.1f}%")
        print("  → 両方とも同じ合格率！")

    print("\n結論：各学科では男女で合格率に差がない！")
    print("\n" + "!" * 60)
    print("これがSimpson's Paradox（シンプソンのパラドックス）です")
    print("!" * 60)

    # ============================================================
    # 可視化
    # ============================================================

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 左：全体の合格率
    ax = axes[0]
    x = np.arange(2)
    overall_rates = [
        overall_stats.loc["女性", "合格率(%)"] / 100,
        overall_stats.loc["男性", "合格率(%)"] / 100,
    ]
    colors = ["#ff6b6b", "#4ecdc4"]
    bars = ax.bar(x, overall_rates, color=colors, edgecolor="black", linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(["女性", "男性"], fontsize=12)
    ax.set_ylabel("合格率", fontsize=12)
    ax.set_title("全体での合格率", fontsize=14, fontweight="bold")
    ax.set_ylim([0, 0.8])
    ax.grid(True, alpha=0.3, axis="y")

    for i, (bar, rate) in enumerate(zip(bars, overall_rates)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 0.02,
            f"{rate * 100:.1f}%",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

    # 中央：学科A
    ax = axes[1]
    dept_a_data = dept_stats[dept_stats["学科"] == "A"]
    female_a = dept_a_data[dept_a_data["性別"] == "女性"]["合格率(%)"].values[0] / 100
    male_a = dept_a_data[dept_a_data["性別"] == "男性"]["合格率(%)"].values[0] / 100
    bars = ax.bar(x, [female_a, male_a], color=colors, edgecolor="black", linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(["女性", "男性"], fontsize=12)
    ax.set_ylabel("合格率", fontsize=12)
    ax.set_title("学科Aでの合格率", fontsize=14, fontweight="bold")
    ax.set_ylim([0, 0.8])
    ax.grid(True, alpha=0.3, axis="y")

    for i, (bar, rate) in enumerate(zip(bars, [female_a, male_a])):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 0.02,
            f"{rate * 100:.1f}%",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

    # 右：学科B
    ax = axes[2]
    dept_b_data = dept_stats[dept_stats["学科"] == "B"]
    female_b = dept_b_data[dept_b_data["性別"] == "女性"]["合格率(%)"].values[0] / 100
    male_b = dept_b_data[dept_b_data["性別"] == "男性"]["合格率(%)"].values[0] / 100
    bars = ax.bar(x, [female_b, male_b], color=colors, edgecolor="black", linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(["女性", "男性"], fontsize=12)
    ax.set_ylabel("合格率", fontsize=12)
    ax.set_title("学科Bでの合格率", fontsize=14, fontweight="bold")
    ax.set_ylim([0, 0.8])
    ax.grid(True, alpha=0.3, axis="y")

    for i, (bar, rate) in enumerate(zip(bars, [female_b, male_b])):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 0.02,
            f"{rate * 100:.1f}%",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

    plt.suptitle(
        "Simpson's Paradox in University Admission",
        fontsize=16,
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()
    plt.savefig("advanced1_simpsons_paradox.png", dpi=100, bbox_inches="tight")
    print("\nグラフを保存しました: advanced1_simpsons_paradox.png")
    plt.show()

    # ============================================================
    # なぜこうなるのか？
    # ============================================================

    print("\n" + "=" * 60)
    print("【なぜこのようなパラドックスが起きるのか？】")
    print("=" * 60)

    print("\n原因：志願者の分布の偏り")
    print("\n学科Aの志願者:")
    print(
        f"  女性: {(df[(df['学科'] == 'A') & (df['性別'] == '女性')]).shape[0]}人 (80%)"
    )
    print(
        f"  男性: {(df[(df['学科'] == 'A') & (df['性別'] == '男性')]).shape[0]}人 (20%)"
    )
    print("  → 合格しやすい学科に女性が集中")

    print("\n学科Bの志願者:")
    print(
        f"  女性: {(df[(df['学科'] == 'B') & (df['性別'] == '女性')]).shape[0]}人 (20%)"
    )
    print(
        f"  男性: {(df[(df['学科'] == 'B') & (df['性別'] == '男性')]).shape[0]}人 (80%)"
    )
    print("  → 合格しにくい学科に男性が集中")

    print("\n結果:")
    print("- 各学科では男女の合格率は同じ（公平）")
    print("- しかし全体で見ると女性の方が合格率が高い")
    print("- これは志願する学科の選択（交絡因子）の影響")

    # ============================================================
    # 実世界での重要性
    # ============================================================

    print("\n" + "=" * 60)
    print("【実世界での重要性】")
    print("=" * 60)

    print("\n1. 医療研究:")
    print("   - 新薬の効果を評価する際、年齢や重症度で層別化が必要")
    print("   - 全体では効果なしでも、特定の患者群では効果あり（またはその逆）")

    print("\n2. ビジネス分析:")
    print("   - Webサイトの改善効果を評価する際、デバイスや時間帯で層別化")
    print("   - 全体の売上は減少でも、主要顧客層では増加している可能性")

    print("\n3. 社会科学:")
    print("   - 教育政策の効果、差別の有無の判定など")
    print("   - 集計レベルによって結論が逆転する危険性")

    print("\n" + "=" * 60)
    print("応用課題1完了！")
    print("=" * 60)

    assert False

    """
    応用課題2：モンテカルロシミュレーション
    ==========================================
    概要：
    ランダムサンプリングを繰り返して数値計算や確率推定を行う手法

    有用性：
    - 解析的に解けない問題を数値的に解決
    - 確率や期待値の推定
    - リスク分析、意思決定支援

    所要時間：15分
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from matplotlib.patches import Circle

    np.random.seed(42)
    sns.set_style("whitegrid")

    print("=" * 60)
    print("応用課題2：モンテカルロシミュレーション")
    print("=" * 60)

    # ============================================================
    # 例1：円周率πの推定
    # ============================================================

    print("\n[例1] 円周率πの推定")
    print("-" * 60)

    print("\n【方法】")
    print("1. 1辺が2の正方形の中に半径1の円を描く")
    print("2. 正方形内にランダムに点を打つ")
    print("3. 円内に入る点の割合 ≈ π/4")
    print("   （円の面積 π / 正方形の面積 4）")

    def estimate_pi(n_samples):
        """
        モンテカルロ法で円周率を推定
        """
        # ランダムに点を生成（-1〜1の範囲）
        x = np.random.uniform(-1, 1, n_samples)
        y = np.random.uniform(-1, 1, n_samples)

        # 原点からの距離
        distance = np.sqrt(x**2 + y**2)

        # 円内の点の数
        inside_circle = (distance <= 1).sum()

        # πの推定値
        pi_estimate = 4 * inside_circle / n_samples

        return pi_estimate, x, y, distance <= 1

    # 様々なサンプル数で実験
    sample_sizes = [100, 1000, 10000, 100000]
    results = []

    print("\n【実験結果】")
    for n in sample_sizes:
        pi_est, _, _, _ = estimate_pi(n)
        error = abs(pi_est - np.pi)
        results.append((n, pi_est, error))
        print(f"n={n:6d}: π ≈ {pi_est:.6f}, 誤差: {error:.6f}")

    print(f"\n真の値: π = {np.pi:.6f}")

    # 可視化：サンプル数と精度の関係
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    for idx, n in enumerate([100, 1000, 10000, 100000]):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        # n個のサンプルで推定
        pi_est, x, y, inside = estimate_pi(n)

        # プロット
        ax.scatter(x[inside], y[inside], c="red", s=1, alpha=0.5, label="Inside circle")
        ax.scatter(
            x[~inside], y[~inside], c="blue", s=1, alpha=0.5, label="Outside circle"
        )

        # 円を描画
        circle = Circle((0, 0), 1, fill=False, edgecolor="black", linewidth=2)
        ax.add_patch(circle)

        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.set_aspect("equal")
        ax.set_title(
            f"n={n}: π ≈ {pi_est:.6f} (error: {abs(pi_est - np.pi):.6f})", fontsize=12
        )
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=9)

    plt.suptitle(
        "Estimating π using Monte Carlo Method", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig("advanced2_monte_carlo_pi.png", dpi=100, bbox_inches="tight")
    print("\nグラフを保存しました: advanced2_monte_carlo_pi.png")
    plt.show()

    # ============================================================
    # 例2：積分の計算
    # ============================================================

    print("\n" + "=" * 60)
    print("[例2] 積分の計算")
    print("-" * 60)

    print("\n【問題】")
    print("∫[0,1] x² dx = ?")
    print("解析解: 1/3 ≈ 0.333...")

    print("\n【モンテカルロ法】")
    print("1. [0,1]区間でランダムにx座標を生成")
    print("2. 各点でf(x)=x²を計算")
    print("3. 平均値 × 区間の長さ ≈ 積分値")

    def monte_carlo_integral(func, a, b, n_samples):
        """
        モンテカルロ法で積分を計算
        """
        x = np.random.uniform(a, b, n_samples)
        y = func(x)
        integral_estimate = (b - a) * np.mean(y)
        return integral_estimate, x, y

    # 関数定義
    def f(x):
        return x**2

    # 真の値
    true_value = 1 / 3

    # 様々なサンプル数で実験
    print("\n【実験結果】")
    for n in [100, 1000, 10000, 100000]:
        estimate, _, _ = monte_carlo_integral(f, 0, 1, n)
        error = abs(estimate - true_value)
        print(f"n={n:6d}: ∫x²dx ≈ {estimate:.6f}, 誤差: {error:.6f}")

    print(f"\n真の値: {true_value:.6f}")

    # 可視化
    n_vis = 1000
    estimate, x_samples, y_samples = monte_carlo_integral(f, 0, 1, n_vis)

    fig, ax = plt.subplots(figsize=(10, 8))

    # 関数のプロット
    x_curve = np.linspace(0, 1, 1000)
    y_curve = f(x_curve)
    ax.plot(x_curve, y_curve, "b-", linewidth=2, label="f(x) = x²")
    ax.fill_between(
        x_curve, y_curve, alpha=0.3, color="lightblue", label="Area (Integral)"
    )

    # サンプル点
    ax.scatter(
        x_samples,
        y_samples,
        c="red",
        s=10,
        alpha=0.5,
        label=f"Random samples (n={n_vis})",
    )

    # 平均値の線
    mean_y = np.mean(y_samples)
    ax.axhline(
        mean_y, color="green", linestyle="--", linewidth=2, label=f"Mean: {mean_y:.3f}"
    )

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("f(x)", fontsize=12)
    ax.set_title(
        f"Monte Carlo Integration: ∫x²dx ≈ {estimate:.6f} (True: {true_value:.6f})",
        fontsize=14,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.1])

    plt.tight_layout()
    plt.savefig("advanced2_monte_carlo_integration.png", dpi=100, bbox_inches="tight")
    print("\nグラフを保存しました: advanced2_monte_carlo_integration.png")
    plt.show()

    # ============================================================
    # 例3：リスク分析（投資ポートフォリオ）
    # ============================================================

    print("\n" + "=" * 60)
    print("[例3] リスク分析：投資ポートフォリオのシミュレーション")
    print("-" * 60)

    print("\n【設定】")
    print("初期投資: 100万円")
    print("年間リターン: 平均7%、標準偏差15%（正規分布と仮定）")
    print("投資期間: 10年")
    print("質問: 10年後の資産はどう分布する？")

    # パラメータ
    initial_investment = 1000000  # 100万円
    mean_return = 0.07  # 7%
    std_return = 0.15  # 15%
    years = 10
    n_simulations = 10000

    # シミュレーション
    final_values = []

    for _ in range(n_simulations):
        value = initial_investment
        for year in range(years):
            annual_return = np.random.normal(mean_return, std_return)
            value *= 1 + annual_return
        final_values.append(value)

    final_values = np.array(final_values)

    # 統計量
    mean_final = final_values.mean()
    median_final = np.median(final_values)
    percentile_5 = np.percentile(final_values, 5)
    percentile_95 = np.percentile(final_values, 95)

    print(f"\n【シミュレーション結果】（{n_simulations}回）")
    print(f"平均値: {mean_final / 10000:.1f}万円")
    print(f"中央値: {median_final / 10000:.1f}万円")
    print(f"5パーセンタイル: {percentile_5 / 10000:.1f}万円")
    print(f"95パーセンタイル: {percentile_95 / 10000:.1f}万円")
    print(
        f"\n元本割れの確率: {(final_values < initial_investment).sum() / n_simulations * 100:.1f}%"
    )
    print(
        f"2倍以上になる確率: {(final_values >= 2 * initial_investment).sum() / n_simulations * 100:.1f}%"
    )

    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左：ヒストグラム
    ax = axes[0]
    ax.hist(
        final_values / 10000,
        bins=50,
        density=True,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
    )
    ax.axvline(
        initial_investment / 10000,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Initial: {initial_investment / 10000:.0f}万円",
    )
    ax.axvline(
        mean_final / 10000,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_final / 10000:.1f}万円",
    )
    ax.axvline(
        median_final / 10000,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_final / 10000:.1f}万円",
    )
    ax.axvline(
        percentile_5 / 10000,
        color="purple",
        linestyle=":",
        linewidth=2,
        label=f"5%ile: {percentile_5 / 10000:.1f}万円",
    )
    ax.axvline(
        percentile_95 / 10000,
        color="purple",
        linestyle=":",
        linewidth=2,
        label=f"95%ile: {percentile_95 / 10000:.1f}万円",
    )
    ax.set_xlabel("Final Value (万円)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Distribution of Final Portfolio Value", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 右：いくつかのシミュレーションパスを可視化
    ax = axes[1]

    # 10個のシミュレーションパスを描画
    np.random.seed(42)
    for i in range(10):
        value = initial_investment
        values = [value]
        for year in range(years):
            annual_return = np.random.normal(mean_return, std_return)
            value *= 1 + annual_return
            values.append(value)
        ax.plot(range(years + 1), np.array(values) / 10000, alpha=0.5, linewidth=1)

    # 期待値のパス
    expected_path = [
        initial_investment * (1 + mean_return) ** year for year in range(years + 1)
    ]
    ax.plot(
        range(years + 1),
        np.array(expected_path) / 10000,
        "r-",
        linewidth=3,
        label="Expected Path",
    )

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Portfolio Value (万円)", fontsize=12)
    ax.set_title("Sample Simulation Paths", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Portfolio Simulation ({n_simulations} simulations, {years} years)",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("advanced2_monte_carlo_portfolio.png", dpi=100, bbox_inches="tight")
    print("\nグラフを保存しました: advanced2_monte_carlo_portfolio.png")
    plt.show()

    # ============================================================
    # モンテカルロ法の利点と注意点
    # ============================================================

    print("\n" + "=" * 60)
    print("【モンテカルロ法の利点】")
    print("=" * 60)
    print("1. 複雑な問題を簡単なコードで解ける")
    print("2. 高次元の問題にも適用可能")
    print("3. 確率分布を仮定できれば不確実性を定量化できる")
    print("4. 直感的でわかりやすい")

    print("\n【注意点】")
    print("1. サンプル数が少ないと精度が低い")
    print("2. 計算コストがかかる（大量のサンプルが必要）")
    print("3. 乱数の質に依存")
    print("4. 確率分布の仮定が適切でないと誤った結論")

    print("\n" + "=" * 60)
    print("応用課題2完了！")
    print("=" * 60)

    assert False

    """
    応用課題3：ブートストラップ法
    ==============================
    概要：
    標本から復元抽出を繰り返して統計量の分布を推定する手法

    有用性：
    - 理論分布がわからない場合の信頼区間推定
    - 小サンプルでの推測統計
    - 機械学習モデルの性能評価

    所要時間：15分
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    np.random.seed(42)
    sns.set_style("whitegrid")

    print("=" * 60)
    print("応用課題3：ブートストラップ法")
    print("=" * 60)

    # ============================================================
    # 基本概念
    # ============================================================

    print("\n【ブートストラップ法とは？】")
    print("-" * 60)
    print("手元にあるデータから復元抽出を繰り返し、")
    print("統計量の分布を推定する方法")
    print("\n例：平均値の信頼区間を知りたいが、")
    print("    母集団の分布がわからない場合に有効")

    # ============================================================
    # 例1：平均値の信頼区間推定
    # ============================================================

    print("\n" + "=" * 60)
    print("[例1] 平均値の信頼区間推定")
    print("=" * 60)

    # 元のサンプルデータ（未知の分布から得られたと仮定）
    # 実際には指数分布から生成（非正規）
    original_sample = np.random.exponential(scale=10, size=30)

    print(f"\n元のサンプルサイズ: {len(original_sample)}")
    print(f"元のサンプルの平均: {original_sample.mean():.2f}")
    print(f"元のサンプルの中央値: {np.median(original_sample):.2f}")

    # ブートストラップ
    n_bootstrap = 10000
    bootstrap_means = []

    for i in range(n_bootstrap):
        # 復元抽出
        bootstrap_sample = np.random.choice(
            original_sample, size=len(original_sample), replace=True
        )
        bootstrap_means.append(bootstrap_sample.mean())

    bootstrap_means = np.array(bootstrap_means)

    # 信頼区間（パーセンタイル法）
    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)

    print(f"\n【ブートストラップ結果】（{n_bootstrap}回）")
    print(f"平均の推定値: {bootstrap_means.mean():.2f}")
    print(f"標準誤差: {bootstrap_means.std():.2f}")
    print(f"95%信頼区間: [{ci_lower:.2f}, {ci_upper:.2f}]")

    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左：元のサンプルのヒストグラム
    ax = axes[0]
    ax.hist(
        original_sample,
        bins=15,
        density=True,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
        label="Original Sample",
    )
    ax.axvline(
        original_sample.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {original_sample.mean():.2f}",
    )
    ax.set_xlabel("Value", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Original Sample (n={len(original_sample)})", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 右：ブートストラップ平均の分布
    ax = axes[1]
    ax.hist(
        bootstrap_means,
        bins=50,
        density=True,
        alpha=0.7,
        color="lightcoral",
        edgecolor="black",
        label="Bootstrap Means",
    )
    ax.axvline(
        bootstrap_means.mean(),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Mean of Means: {bootstrap_means.mean():.2f}",
    )
    ax.axvline(
        ci_lower,
        color="purple",
        linestyle=":",
        linewidth=2,
        label=f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]",
    )
    ax.axvline(ci_upper, color="purple", linestyle=":", linewidth=2)
    ax.fill_betweenx(
        [0, ax.get_ylim()[1]], ci_lower, ci_upper, alpha=0.2, color="purple"
    )
    ax.set_xlabel("Bootstrap Mean", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Distribution of Bootstrap Means (B={n_bootstrap})", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Bootstrap Method for Mean Estimation", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig("advanced3_bootstrap_mean.png", dpi=100, bbox_inches="tight")
    print("\nグラフを保存しました: advanced3_bootstrap_mean.png")
    plt.show()

    # ============================================================
    # 例2：中央値の信頼区間（通常の方法では難しい）
    # ============================================================

    print("\n" + "=" * 60)
    print("[例2] 中央値の信頼区間推定")
    print("=" * 60)

    print("\n中央値の信頼区間は理論的に求めにくいが、")
    print("ブートストラップ法なら簡単に推定できる！")

    # ブートストラップで中央値の分布を推定
    bootstrap_medians = []

    for i in range(n_bootstrap):
        bootstrap_sample = np.random.choice(
            original_sample, size=len(original_sample), replace=True
        )
        bootstrap_medians.append(np.median(bootstrap_sample))

    bootstrap_medians = np.array(bootstrap_medians)

    # 信頼区間
    ci_lower_median = np.percentile(bootstrap_medians, 2.5)
    ci_upper_median = np.percentile(bootstrap_medians, 97.5)

    print("\n【ブートストラップ結果】")
    print(f"中央値の推定値: {bootstrap_medians.mean():.2f}")
    print(f"95%信頼区間: [{ci_lower_median:.2f}, {ci_upper_median:.2f}]")

    # 可視化
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        bootstrap_medians,
        bins=50,
        density=True,
        alpha=0.7,
        color="lightgreen",
        edgecolor="black",
        label="Bootstrap Medians",
    )
    ax.axvline(
        bootstrap_medians.mean(),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Mean of Medians: {bootstrap_medians.mean():.2f}",
    )
    ax.axvline(
        ci_lower_median,
        color="purple",
        linestyle=":",
        linewidth=2,
        label=f"95% CI: [{ci_lower_median:.2f}, {ci_upper_median:.2f}]",
    )
    ax.axvline(ci_upper_median, color="purple", linestyle=":", linewidth=2)
    ax.fill_betweenx(
        [0, ax.get_ylim()[1]],
        ci_lower_median,
        ci_upper_median,
        alpha=0.2,
        color="purple",
    )
    ax.set_xlabel("Bootstrap Median", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Distribution of Bootstrap Medians (B={n_bootstrap})", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("advanced3_bootstrap_median.png", dpi=100, bbox_inches="tight")
    print("\nグラフを保存しました: advanced3_bootstrap_median.png")
    plt.show()

    # ============================================================
    # 例3：複雑な統計量（四分位範囲）の信頼区間
    # ============================================================

    print("\n" + "=" * 60)
    print("[例3] 四分位範囲（IQR）の信頼区間")
    print("=" * 60)

    # 四分位範囲: Q3 - Q1
    original_iqr = np.percentile(original_sample, 75) - np.percentile(
        original_sample, 25
    )

    print(f"\n元のサンプルのIQR: {original_iqr:.2f}")

    # ブートストラップでIQRの分布を推定
    bootstrap_iqrs = []

    for i in range(n_bootstrap):
        bootstrap_sample = np.random.choice(
            original_sample, size=len(original_sample), replace=True
        )
        q75 = np.percentile(bootstrap_sample, 75)
        q25 = np.percentile(bootstrap_sample, 25)
        bootstrap_iqrs.append(q75 - q25)

    bootstrap_iqrs = np.array(bootstrap_iqrs)

    # 信頼区間
    ci_lower_iqr = np.percentile(bootstrap_iqrs, 2.5)
    ci_upper_iqr = np.percentile(bootstrap_iqrs, 97.5)

    print("\n【ブートストラップ結果】")
    print(f"IQRの推定値: {bootstrap_iqrs.mean():.2f}")
    print(f"95%信頼区間: [{ci_lower_iqr:.2f}, {ci_upper_iqr:.2f}]")

    # 可視化
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        bootstrap_iqrs,
        bins=50,
        density=True,
        alpha=0.7,
        color="lightsalmon",
        edgecolor="black",
        label="Bootstrap IQRs",
    )
    ax.axvline(
        bootstrap_iqrs.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean of IQRs: {bootstrap_iqrs.mean():.2f}",
    )
    ax.axvline(
        ci_lower_iqr,
        color="purple",
        linestyle=":",
        linewidth=2,
        label=f"95% CI: [{ci_lower_iqr:.2f}, {ci_upper_iqr:.2f}]",
    )
    ax.axvline(ci_upper_iqr, color="purple", linestyle=":", linewidth=2)
    ax.fill_betweenx(
        [0, ax.get_ylim()[1]], ci_lower_iqr, ci_upper_iqr, alpha=0.2, color="purple"
    )
    ax.set_xlabel("Bootstrap IQR", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Distribution of Bootstrap IQRs (B={n_bootstrap})", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("advanced3_bootstrap_iqr.png", dpi=100, bbox_inches="tight")
    print("\nグラフを保存しました: advanced3_bootstrap_iqr.png")
    plt.show()

    # ============================================================
    # 例4：2つのグループの平均値の差の検定
    # ============================================================

    print("\n" + "=" * 60)
    print("[例4] 2グループ間の平均値の差の検定")
    print("=" * 60)

    print("\n【設定】")
    print("新しい教育プログラムの効果を検証")
    print("グループA（従来法）とグループB（新プログラム）の成績を比較")

    # データ生成（実際にはわずかに差がある）
    group_a = np.random.normal(70, 10, size=25)  # 平均70
    group_b = np.random.normal(75, 10, size=25)  # 平均75

    observed_diff = group_b.mean() - group_a.mean()

    print(f"\nグループA（n={len(group_a)}）: 平均 {group_a.mean():.2f}")
    print(f"グループB（n={len(group_b)}）: 平均 {group_b.mean():.2f}")
    print(f"観測された差: {observed_diff:.2f}")

    # ブートストラップで差の分布を推定
    bootstrap_diffs = []

    for i in range(n_bootstrap):
        # 各グループから復元抽出
        bootstrap_a = np.random.choice(group_a, size=len(group_a), replace=True)
        bootstrap_b = np.random.choice(group_b, size=len(group_b), replace=True)

        diff = bootstrap_b.mean() - bootstrap_a.mean()
        bootstrap_diffs.append(diff)

    bootstrap_diffs = np.array(bootstrap_diffs)

    # 信頼区間
    ci_lower_diff = np.percentile(bootstrap_diffs, 2.5)
    ci_upper_diff = np.percentile(bootstrap_diffs, 97.5)

    print("\n【ブートストラップ結果】")
    print(f"差の推定値: {bootstrap_diffs.mean():.2f}")
    print(f"95%信頼区間: [{ci_lower_diff:.2f}, {ci_upper_diff:.2f}]")

    # 統計的有意性の判定
    if ci_lower_diff > 0:
        print("\n結論: グループBの方が有意に高い（95%信頼水準）")
    elif ci_upper_diff < 0:
        print("\n結論: グループAの方が有意に高い（95%信頼水準）")
    else:
        print("\n結論: 有意差なし（95%信頼水準）")

    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左：元のデータの箱ひげ図
    ax = axes[0]
    data_to_plot = [group_a, group_b]
    bp = ax.boxplot(
        data_to_plot,
        labels=["Group A\n(Traditional)", "Group B\n(New Program)"],
        patch_artist=True,
        widths=0.6,
    )
    bp["boxes"][0].set_facecolor("lightblue")
    bp["boxes"][1].set_facecolor("lightcoral")

    # 平均値を追加
    means = [group_a.mean(), group_b.mean()]
    ax.scatter([1, 2], means, color="red", s=100, zorder=3, label="Mean", marker="D")

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Original Data", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # 右：差の分布
    ax = axes[1]
    ax.hist(
        bootstrap_diffs,
        bins=50,
        density=True,
        alpha=0.7,
        color="lightgreen",
        edgecolor="black",
        label="Bootstrap Differences",
    )
    ax.axvline(0, color="black", linestyle="-", linewidth=2, label="No difference")
    ax.axvline(
        bootstrap_diffs.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean Diff: {bootstrap_diffs.mean():.2f}",
    )
    ax.axvline(
        ci_lower_diff,
        color="purple",
        linestyle=":",
        linewidth=2,
        label=f"95% CI: [{ci_lower_diff:.2f}, {ci_upper_diff:.2f}]",
    )
    ax.axvline(ci_upper_diff, color="purple", linestyle=":", linewidth=2)
    ax.fill_betweenx(
        [0, ax.get_ylim()[1]], ci_lower_diff, ci_upper_diff, alpha=0.2, color="purple"
    )
    ax.set_xlabel("Difference in Means (B - A)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"Bootstrap Distribution of Differences (B={n_bootstrap})", fontsize=14
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Bootstrap Hypothesis Testing: Two-Group Comparison",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("advanced3_bootstrap_two_groups.png", dpi=100, bbox_inches="tight")
    print("\nグラフを保存しました: advanced3_bootstrap_two_groups.png")
    plt.show()

    # ============================================================
    # 例5：機械学習モデルの性能評価
    # ============================================================

    print("\n" + "=" * 60)
    print("[例5] 機械学習モデルの性能評価")
    print("=" * 60)

    print("\n【設定】")
    print("分類モデルの精度を評価")
    print("テストデータでの精度の信頼区間を推定")

    # シミュレーションデータ
    # 実際の正解ラベル
    y_true = np.array([1] * 60 + [0] * 40)  # 100サンプル

    # モデルの予測（80%の精度）
    np.random.seed(42)
    y_pred = y_true.copy()
    # ランダムに20個を間違える
    wrong_indices = np.random.choice(100, size=20, replace=False)
    y_pred[wrong_indices] = 1 - y_pred[wrong_indices]

    # 元の精度
    original_accuracy = (y_true == y_pred).mean()

    print(f"\nテストデータサイズ: {len(y_true)}")
    print(f"観測された精度: {original_accuracy:.3f}")

    # ブートストラップで精度の分布を推定
    bootstrap_accuracies = []

    for i in range(n_bootstrap):
        # 復元抽出（予測と正解のペアを保持）
        indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
        bootstrap_y_true = y_true[indices]
        bootstrap_y_pred = y_pred[indices]

        accuracy = (bootstrap_y_true == bootstrap_y_pred).mean()
        bootstrap_accuracies.append(accuracy)

    bootstrap_accuracies = np.array(bootstrap_accuracies)

    # 信頼区間
    ci_lower_acc = np.percentile(bootstrap_accuracies, 2.5)
    ci_upper_acc = np.percentile(bootstrap_accuracies, 97.5)

    print("\n【ブートストラップ結果】")
    print(f"精度の推定値: {bootstrap_accuracies.mean():.3f}")
    print(f"標準誤差: {bootstrap_accuracies.std():.3f}")
    print(f"95%信頼区間: [{ci_lower_acc:.3f}, {ci_upper_acc:.3f}]")

    # 可視化
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        bootstrap_accuracies,
        bins=50,
        density=True,
        alpha=0.7,
        color="lightblue",
        edgecolor="black",
        label="Bootstrap Accuracies",
    )
    ax.axvline(
        bootstrap_accuracies.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean Accuracy: {bootstrap_accuracies.mean():.3f}",
    )
    ax.axvline(
        ci_lower_acc,
        color="purple",
        linestyle=":",
        linewidth=2,
        label=f"95% CI: [{ci_lower_acc:.3f}, {ci_upper_acc:.3f}]",
    )
    ax.axvline(ci_upper_acc, color="purple", linestyle=":", linewidth=2)
    ax.fill_betweenx(
        [0, ax.get_ylim()[1]], ci_lower_acc, ci_upper_acc, alpha=0.2, color="purple"
    )
    ax.set_xlabel("Accuracy", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"Bootstrap Distribution of Model Accuracy (B={n_bootstrap})", fontsize=14
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("advanced3_bootstrap_ml_accuracy.png", dpi=100, bbox_inches="tight")
    print("\nグラフを保存しました: advanced3_bootstrap_ml_accuracy.png")
    plt.show()

    # ============================================================
    # ブートストラップ法の利点と注意点
    # ============================================================

    print("\n" + "=" * 60)
    print("【ブートストラップ法の利点】")
    print("=" * 60)
    print("1. 母集団の分布を仮定しなくてよい（ノンパラメトリック）")
    print("2. 複雑な統計量の信頼区間も推定可能")
    print("3. 小サンプルでも適用可能")
    print("4. 実装が簡単")
    print("5. 直感的でわかりやすい")

    print("\n【注意点】")
    print("1. 元のサンプルが母集団を代表していることが前提")
    print("2. サンプルサイズが極端に小さいと信頼性が低い")
    print("3. 計算コストがかかる（B回の再サンプリング）")
    print("4. 偏ったサンプルからは偏った結果")

    print("\n【推奨されるブートストラップ回数】")
    print("- 標準誤差の推定: B = 1,000〜2,000")
    print("- 信頼区間の推定: B = 5,000〜10,000")
    print("- 仮説検定: B = 10,000〜20,000")

    print("\n【実世界での応用】")
    print("1. 医学研究: 小サンプルでの治療効果の推定")
    print("2. 経済学: 経済指標の信頼区間推定")
    print("3. 機械学習: モデル性能の不確実性評価")
    print("4. A/Bテスト: コンバージョン率の差の検定")
    print("5. 品質管理: 不良率の信頼区間推定")

    # ============================================================
    # ボーナス：ブートストラップ回数の影響
    # ============================================================

    print("\n" + "=" * 60)
    print("[ボーナス] ブートストラップ回数の影響")
    print("=" * 60)

    # 様々なB値で信頼区間の安定性を確認
    b_values = [100, 500, 1000, 5000, 10000]
    ci_results = []

    for b in b_values:
        bootstrap_means_b = []
        for i in range(b):
            bootstrap_sample = np.random.choice(
                original_sample, size=len(original_sample), replace=True
            )
            bootstrap_means_b.append(bootstrap_sample.mean())

        bootstrap_means_b = np.array(bootstrap_means_b)
        ci_lower_b = np.percentile(bootstrap_means_b, 2.5)
        ci_upper_b = np.percentile(bootstrap_means_b, 97.5)
        ci_width = ci_upper_b - ci_lower_b

        ci_results.append(
            {
                "B": b,
                "CI_lower": ci_lower_b,
                "CI_upper": ci_upper_b,
                "CI_width": ci_width,
            }
        )

        print(
            f"B={b:5d}: CI=[{ci_lower_b:.2f}, {ci_upper_b:.2f}], width={ci_width:.2f}"
        )

    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左：信頼区間の変化
    ax = axes[0]
    b_array = [r["B"] for r in ci_results]
    ci_lowers = [r["CI_lower"] for r in ci_results]
    ci_uppers = [r["CI_upper"] for r in ci_results]

    ax.plot(b_array, ci_lowers, "o-", linewidth=2, markersize=8, label="Lower bound")
    ax.plot(b_array, ci_uppers, "s-", linewidth=2, markersize=8, label="Upper bound")
    ax.set_xlabel("Number of Bootstrap Samples (B)", fontsize=12)
    ax.set_ylabel("Confidence Interval Boundary", fontsize=12)
    ax.set_title("Stability of Confidence Interval with B", fontsize=14)
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 右：信頼区間の幅
    ax = axes[1]
    ci_widths = [r["CI_width"] for r in ci_results]

    ax.plot(b_array, ci_widths, "o-", linewidth=2, markersize=8, color="green")
    ax.set_xlabel("Number of Bootstrap Samples (B)", fontsize=12)
    ax.set_ylabel("Width of Confidence Interval", fontsize=12)
    ax.set_title("CI Width Stabilization", fontsize=14)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Effect of Bootstrap Sample Size on Confidence Interval",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("advanced3_bootstrap_b_effect.png", dpi=100, bbox_inches="tight")
    print("\nグラフを保存しました: advanced3_bootstrap_b_effect.png")
    plt.show()

    print("\n観察: B=1,000以上で信頼区間が安定")

    print("\n" + "=" * 60)
    print("応用課題3完了！")
    print("=" * 60)

    print("\n" + "=" * 70)
    print(" 全ての演習・応用課題が完了しました！お疲れ様でした！ ")
    print("=" * 70)


if __name__ == "__main__":
    section_three()
