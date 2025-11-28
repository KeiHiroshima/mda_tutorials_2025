import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 日本語フォント設定（環境に応じて調整が必要な場合があります）
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 画像サイズ，フォントサイズの設定
plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams.update(
    {
        "font.size": 16,
        "axes.titlesize": 24,
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "figure.titlesize": 24,
    }
)

# データの読み込み
penguins = sns.load_dataset("penguins")
# 欠損値を含む行を削除
penguins_clean = penguins.dropna()

print("データセットの概要")
print(penguins_clean.head())
print("\nデータの基本統計量")
print(penguins_clean.describe())

# 1. 棒グラフ (Bar Chart)
fig, ax = plt.subplots()
ax.bar(
    ["Adelie", "Gentoo", "Chinstrap"],
    penguins_clean["species"].value_counts(),
    color=["#1f77b4", "#ff7f0e", "#2ca02c"],
)
ax.set_xlabel("Species")
ax.set_ylabel("Count")
ax.set_ylim(60, penguins_clean["species"].value_counts().max() + 10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("penguins_bar_ylim.pdf")
plt.close()


# 2. 円グラフ (Pie Chart)
# 種別ごとのオス・メス比を可視化
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax = ax.flatten()
for i, species in enumerate(penguins_clean["species"].unique()):
    species_data = penguins_clean[penguins_clean["species"] == species]
    sex_counts = species_data["sex"].value_counts()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    ax[i].pie(
        sex_counts.values,
        labels=sex_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
    )
    ax[i].set_title(f"{species}")

plt.tight_layout()
plt.savefig("penguins_pie.pdf")
plt.close()


# 3. 積み上げ棒グラフ (Stacked Bar Chart)
fig, ax = plt.subplots()
# 種別と性別でグループ化してカウント
species_sex = penguins_clean.groupby(["species", "sex"]).size().unstack()
species_sex = species_sex.reindex(["Adelie", "Gentoo", "Chinstrap"])

species_sex.plot(kind="bar", stacked=True, ax=ax)
ax.set_xlabel("Species")
ax.set_ylabel("Count")

ax.legend(title="Sex")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.tight_layout()
plt.savefig("penguins_stacked_bar.pdf")
plt.close()


# 4. ヒストグラム (Histogram)
num_bin = 20  # ビンの数
fig, ax = plt.subplots(figsize=(11, 6))
# 種別の体重の分布を可視化
sns.histplot(
    data=penguins_clean,
    x="body_mass_g",
    hue="species",
    bins=num_bin,
    kde=False,
    ax=ax,
    legend=True,
)
ax.set_xlabel("Body Mass (g)")
ax.set_ylabel("Frequency")
ax.grid(axis="y", alpha=0.3)
sns.move_legend(ax, "upper right", bbox_to_anchor=(1.3, 1))
plt.tight_layout()
plt.savefig(f"penguins_histogram_{num_bin}bins.pdf")
plt.close()


# 5. 箱ひげ図 (Box Plot)
# 種別ごとの体重の分布を比較
fig, ax = plt.subplots()
sns.boxplot(
    data=penguins_clean,
    x="species",
    y="body_mass_g",
    ax=ax,
    palette=["#1f77b4", "#ff7f0e", "#2ca02c"],
)
ax.set_xlabel("Species")
ax.set_ylabel("Body Mass (g)")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("penguins_boxplot.pdf")
plt.close()


# ===================================================================
# 6. 散布図 (Scatter Plot)
# ===================================================================
# くちばしの長さと深さの関係を可視化
fig, ax = plt.subplots(figsize=(12, 6))
for species in penguins_clean["species"].unique():
    species_data = penguins_clean[penguins_clean["species"] == species]
    ax.scatter(
        species_data["bill_length_mm"],
        species_data["bill_depth_mm"],
        label=species,
        alpha=0.6,
        s=50,
    )
ax.set_xlabel("Bill Length (mm)")
ax.set_ylabel("Bill Depth (mm)")
ax.legend(title="Species", loc="upper right", bbox_to_anchor=(1.28, 1))
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("penguins_scatter.pdf")
plt.close()

"""
# ===================================================================
# 7. 折れ線グラフ (Line Chart)
# ===================================================================
# 年次ごとの観測個体数の推移を可視化
fig, ax = plt.subplots()
yearly_counts = penguins_clean.groupby(["year", "species"]).size().unstack(fill_value=0)
for species in yearly_counts.columns:
    ax.plot(
        yearly_counts.index,
        yearly_counts[species],
        marker="o",
        label=species,
        linewidth=2,
    )
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Temporal Trend of Penguin Observations", fontsize=14, fontweight="bold")
ax.legend(title="Species")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("penguins_line.pdf")
plt.close()

breakpoint()"""

# 8. ビニング手法の比較：等幅ビン vs 等度数ビン
# 体重データを使用してビニング手法の違いを可視化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes = axes.flatten()
body_mass = penguins_clean["body_mass_g"]

# (1) 等幅ビン：10分割（ビン境界の可視化）
bin_edges_equal_width = np.linspace(body_mass.min(), body_mass.max(), 11)
axes[0].hist(
    body_mass,
    bins=bin_edges_equal_width,
    color="lightcoral",
    edgecolor="black",
    alpha=0.7,
)
for edge in bin_edges_equal_width:
    axes[0].axvline(edge, color="red", linestyle="--", linewidth=1, alpha=0.5)
axes[0].set_xlabel("Body Mass (g)")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Equal-Width Binning")
axes[0].grid(axis="y", alpha=0.3)

# (2) 等度数ビン：10分割（ビン境界の可視化）
# quantileを使用して各ビンに同程度のデータ数が入るようにする
bin_edges_equal_freq = np.percentile(body_mass, np.linspace(0, 100, 11))
axes[1].hist(
    body_mass,
    bins=bin_edges_equal_freq,
    color="lightgreen",
    edgecolor="black",
    alpha=0.7,
)
for edge in bin_edges_equal_freq:
    axes[1].axvline(edge, color="green", linestyle="--", linewidth=1, alpha=0.5)
axes[1].set_xlabel("Body Mass (g)")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Equal-Frequency Binning")
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("penguins_binning_comparison.pdf")
plt.close()


# 9. ビニング手法の詳細比較
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 等幅ビンと等度数ビンのビン幅を比較
bin_widths_equal_width = np.diff(bin_edges_equal_width)
bin_widths_equal_freq = np.diff(bin_edges_equal_freq)

# 等幅ビンのビン幅
axes[0].bar(
    range(len(bin_widths_equal_width)),
    bin_widths_equal_width,
    color="lightcoral",
    edgecolor="black",
    alpha=0.7,
)
axes[0].set_xlabel("Bin Index")
axes[0].set_ylabel("Bin Width (g)")
axes[0].set_title("Equal-Width Binning")
axes[0].grid(axis="y", alpha=0.3)

# 等度数ビンのビン幅
axes[1].bar(
    range(len(bin_widths_equal_freq)),
    bin_widths_equal_freq,
    color="lightgreen",
    edgecolor="black",
    alpha=0.7,
)
axes[1].set_xlabel("Bin Index")
axes[1].set_ylabel("Bin Width (g)")
axes[1].set_title("Equal-Frequency Binning")
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("penguins_binning_widths.pdf")
plt.close()


# ===================================================================
# ビニング手法の解説
# ===================================================================
print("\n--- ビニング手法の違い ---")
print("等幅ビン (Equal-Width Binning):")
print("  - データの最小値から最大値までを等間隔に分割")
print("  - 各ビンの幅が一定")
print("  - データが偏っている場合、一部のビンに多くのデータが集中する可能性がある")
print("\n等度数ビン (Equal-Frequency Binning):")
print("  - 各ビンに含まれるデータ数がほぼ同じになるように分割")
print("  - ビンの幅は不均一")
print("  - データの分布に適応的で、外れ値の影響を受けにくい")

fig, ax = plt.subplots(figsize=(7.5, 6))

# 種別と島でクロス集計
species_island = pd.crosstab(penguins_clean["species"], penguins_clean["island"])

# ヒートマップの作成
sns.heatmap(
    species_island,
    annot=True,  # セル内に個体数を表示
    fmt="d",  # 整数で表示
    linewidths=0.5,
    cbar_kws={"label": "Count"},
    cmap="Blues",
    ax=ax,
)

ax.set_xlabel("Island")
ax.set_ylabel("Species")
plt.tight_layout()
plt.savefig("heatmap_species_island.pdf")
plt.close()
