import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

INPUT_PATH  = "data/processed/mutual_funds_clean.csv"
OUTPUT_PATH = "data/processed/mutual_funds_scored.csv"
PLOTS_DIR   = "outputs/ml_plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

TOP_N = 50   # ← FundLens targets Top 50

PALETTE    = ["#0D3349", "#1A6B8A", "#27C4A0", "#F5A623", "#E84855"]
BG_COLOR   = "#F8F9FB"
GRID_COLOR = "#E2E8F0"
plt.rcParams.update({
    "figure.facecolor": BG_COLOR, "axes.facecolor": BG_COLOR,
    "axes.grid": True, "grid.color": GRID_COLOR, "grid.linewidth": 0.8,
    "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
})

# ─────────────────────────────────────────────
print("=" * 60)
print("  STEP 1 — LOADING CLEAN DATA")
print("=" * 60)
df = pd.read_csv(INPUT_PATH)
print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

# ─────────────────────────────────────────────
print("=" * 60)
print("  STEP 2 — FEATURE SELECTION FOR CLUSTERING")
print("=" * 60)

CANDIDATE_FEATURES = [
    "composite_return", "sharpe", "std_dev", "beta",
    "alpha", "expense_ratio", "rari", "net_return",
    "returns_1yr", "returns_3yr", "returns_5yr"
]
CLUSTER_FEATURES = [f for f in CANDIDATE_FEATURES if f in df.columns]
print(f"  Using features: {CLUSTER_FEATURES}\n")

X = df[CLUSTER_FEATURES].copy()
X.fillna(X.median(), inplace=True)
for col in X.columns:
    lo, hi = X[col].quantile(0.01), X[col].quantile(0.99)
    X[col] = X[col].clip(lo, hi)

# ─────────────────────────────────────────────
print("=" * 60)
print("  STEP 3 — FINDING OPTIMAL NUMBER OF CLUSTERS")
print("=" * 60)

scaler_std = StandardScaler()
X_scaled = scaler_std.fit_transform(X)

K_RANGE = range(2, 10)
inertias, silhouettes = [], []

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))
    print(f"  k={k}  Inertia={km.inertia_:,.0f}  Silhouette={silhouette_score(X_scaled, labels):.4f}")

best_k = list(K_RANGE)[np.argmax(silhouettes)]
print(f"\n  ✔  Best k = {best_k} (highest silhouette score)\n")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(K_RANGE, inertias, "o-", color=PALETTE[1], linewidth=2.5, markersize=8)
ax1.axvline(best_k, color=PALETTE[4], linestyle="--", linewidth=1.8, label=f"Best k={best_k}")
ax1.set_title("Elbow Method — Inertia", fontsize=13, fontweight="bold")
ax1.set_xlabel("Number of Clusters (k)"); ax1.set_ylabel("Inertia"); ax1.legend()
ax2.plot(K_RANGE, silhouettes, "s-", color=PALETTE[2], linewidth=2.5, markersize=8)
ax2.axvline(best_k, color=PALETTE[4], linestyle="--", linewidth=1.8, label=f"Best k={best_k}")
ax2.set_title("Silhouette Score by k", fontsize=13, fontweight="bold")
ax2.set_xlabel("Number of Clusters (k)"); ax2.set_ylabel("Silhouette Score"); ax2.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/01_elbow_silhouette.png", dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
plt.close()
print(f"  Saved → {PLOTS_DIR}/01_elbow_silhouette.png")

# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  STEP 4 — FITTING KMEANS (k={best_k})")
print("=" * 60)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(n_clusters=best_k, random_state=42, n_init=15))
])
pipeline.fit(X)
df["cluster"] = pipeline.named_steps["kmeans"].labels_

# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 5 — CLUSTER PROFILING & RISK LABELLING")
print("=" * 60)

profile_cols = [c for c in ["composite_return", "std_dev", "sharpe", "expense_ratio",
                              "fund_size_cr", "alpha", "beta"] if c in df.columns]
cluster_profile = df.groupby("cluster")[profile_cols].median().round(3)
print(cluster_profile.to_string())

cluster_profile["risk_score_raw"] = (
    cluster_profile.get("std_dev", 0) - cluster_profile.get("composite_return", 0)
)
sorted_clusters = cluster_profile["risk_score_raw"].sort_values().index.tolist()

TIER_NAMES = {
    0: "⭐ High Return / Low Risk",
    1: "📈 High Return / Moderate Risk",
    2: "⚖️  Moderate Return / Moderate Risk",
    3: "🔻 Low Return / High Risk",
    4: "⚠️  Very High Risk"
}
cluster_label_map = {}
for rank, cid in enumerate(sorted_clusters):
    tier = min(rank, len(TIER_NAMES) - 1)
    cluster_label_map[cid] = TIER_NAMES[tier]

df["risk_cluster_label"] = df["cluster"].map(cluster_label_map)
print("\n  Cluster → Risk Label mapping:")
for cid, label in cluster_label_map.items():
    n = (df["cluster"] == cid).sum()
    print(f"    Cluster {cid}: {label}  ({n:,} funds)")

# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 6 — INVESTMENT ATTRACTIVENESS SCORE (MinMaxScaler)")
print("=" * 60)
# 
# Scoring logic (resume-ready description):
#   Each metric is independently MinMax-normalised to [0, 1].
#   Higher 3-year return → preferred (weight: 30%)
#   Higher composite return → preferred (weight: 20%)
#   Lower expense ratio → preferred (weight: 20%) — score on inverted value
#   Higher Sharpe ratio → preferred (weight: 15%)
#   Higher Alpha → preferred (weight: 10%)
#   Higher RARI (return per unit risk) → preferred (weight: 5%)
#
# All weights sum to 1.00. Final score is scaled to 0–100.

mms = MinMaxScaler()

SCORE_FEATURES = {
    "returns_3yr":      {"weight": 0.30, "invert": False},
    "composite_return": {"weight": 0.20, "invert": False},
    "expense_ratio":    {"weight": 0.20, "invert": True},   # lower is better
    "sharpe":           {"weight": 0.15, "invert": False},
    "alpha":            {"weight": 0.10, "invert": False},
    "rari":             {"weight": 0.05, "invert": False},
}

# Fall back gracefully if a column is missing
SCORE_FEATURES = {k: v for k, v in SCORE_FEATURES.items() if k in df.columns}

# Re-normalise weights to 1.0 after dropping missing cols
total_w = sum(v["weight"] for v in SCORE_FEATURES.values())
for k in SCORE_FEATURES:
    SCORE_FEATURES[k]["weight"] /= total_w

score = pd.Series(np.zeros(len(df)), index=df.index)

print("  Score components:")
for feat, cfg in SCORE_FEATURES.items():
    col = df[feat].copy().fillna(df[feat].median())
    lo, hi = col.quantile(0.01), col.quantile(0.99)
    col_clipped = col.clip(lo, hi).values.reshape(-1, 1)
    normalised = mms.fit_transform(col_clipped).flatten()
    if cfg["invert"]:
        normalised = 1 - normalised
    contribution = pd.Series(normalised, index=df.index) * cfg["weight"]
    score += contribution
    direction = "↓ lower is better" if cfg["invert"] else "↑ higher is better"
    print(f"    {feat:<25} weight={cfg['weight']*100:.0f}%   {direction}")

df["investment_score"] = (score * 100).round(2)

# Additional quality gate: 1-year return must be positive (consistency signal)
if "returns_1yr" in df.columns:
    df["positive_1yr"] = df["returns_1yr"] > 0
    penalty_mask = df["returns_1yr"] <= 0
    df.loc[penalty_mask, "investment_score"] *= 0.80   # 20% penalty for negative 1yr return
    df["investment_score"] = df["investment_score"].round(2)
    penalised = penalty_mask.sum()
    print(f"\n  ⚠  Applied 20% penalty to {penalised:,} funds with negative 1yr returns")

print(f"\n  Score distribution:")
print(f"    Mean   : {df['investment_score'].mean():.1f}")
print(f"    Median : {df['investment_score'].median():.1f}")
print(f"    Min    : {df['investment_score'].min():.1f}")
print(f"    Max    : {df['investment_score'].max():.1f}")

# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  STEP 7 — TOP {TOP_N} FUNDS")
print("=" * 60)

top_n_cols = [c for c in [
    "scheme_name", "amc_name", "category", "sub_category", "fund_type",
    "risk_level", "risk_cluster_label", "rating", "fund_manager",
    "returns_1yr", "returns_3yr", "returns_5yr", "composite_return",
    "net_return", "expense_ratio", "sharpe", "alpha", "beta", "std_dev",
    "rari", "fund_size_cr", "fund_size_category", "min_sip", "min_lumpsum",
    "investment_score", "return_consistency", "cluster"
] if c in df.columns]

top_n = df.nlargest(TOP_N, "investment_score")[top_n_cols].reset_index(drop=True)
top_n.index += 1
top_n.insert(0, "rank", top_n.index)

print(f"  Top 10 preview:")
preview_cols = [c for c in ["scheme_name", "category", "risk_cluster_label",
                              "returns_3yr", "expense_ratio", "investment_score"] if c in top_n.columns]
print(top_n[preview_cols].head(10).to_string())
print(f"\n  ... full Top {TOP_N} saved to output CSV\n")

# ─────────────────────────────────────────────
# PLOTS

def save_fig(name):
    path = f"{PLOTS_DIR}/{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"  Saved → {path}")


# Plot 1 — PCA cluster scatter
X_scaled_arr = pipeline.named_steps["scaler"].transform(X)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled_arr)
var_exp = pca.explained_variance_ratio_

fig, ax = plt.subplots(figsize=(10, 7))
for i, (cid, label) in enumerate(cluster_label_map.items()):
    mask = df["cluster"] == cid
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=PALETTE[i % len(PALETTE)], label=label,
               alpha=0.55, s=25, edgecolors="none")
ax.set_title("Fund Clusters — PCA 2D Projection", fontsize=14, fontweight="bold")
ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}% variance)")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
plt.tight_layout()
save_fig("02_pca_cluster_plot")

# Plot 2 — Score distribution with Top-50 threshold
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(df["investment_score"], bins=50, color=PALETTE[2], alpha=0.85, edgecolor="white")
ax.axvline(df["investment_score"].mean(), color=PALETTE[4], linewidth=2,
           linestyle="--", label=f"Mean: {df['investment_score'].mean():.1f}")
thresh = top_n["investment_score"].min()
ax.axvline(thresh, color=PALETTE[3], linewidth=2,
           linestyle="-.", label=f"Top {TOP_N} threshold: {thresh:.1f}")
ax.set_title(f"Investment Attractiveness Score — Top {TOP_N} Threshold", fontsize=14, fontweight="bold")
ax.set_xlabel("Score (0–100)"); ax.set_ylabel("Number of Funds")
ax.legend()
plt.tight_layout()
save_fig("03_investment_score_distribution")

# Plot 3 — Top 20 horizontal bar
if "scheme_name" in df.columns:
    top20 = df.nlargest(20, "investment_score")[["scheme_name", "investment_score", "cluster"]].copy()
    top20["short_name"] = top20["scheme_name"].str[:48]
    fig, ax = plt.subplots(figsize=(13, 8))
    colors = [PALETTE[int(c) % len(PALETTE)] for c in top20["cluster"]]
    bars = ax.barh(top20["short_name"][::-1], top20["investment_score"][::-1],
                   color=colors[::-1], edgecolor="white", height=0.7)
    for bar in bars:
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.1f}", va="center", fontsize=9, fontweight="bold")
    ax.set_title(f"Top 20 Funds by Investment Attractiveness Score (FundLens)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Investment Score (0–100)")
    ax.set_xlim(0, top20["investment_score"].max() + 8)
    plt.tight_layout()
    save_fig("04_top20_funds_bar")

# Plot 4 — Cluster radar
radar_features = [c for c in ["composite_return", "sharpe", "expense_ratio", "alpha", "rari"] if c in df.columns]
if len(radar_features) >= 3:
    cluster_means = df.groupby("cluster")[radar_features].mean()
    for col in radar_features:
        mn, mx = cluster_means[col].min(), cluster_means[col].max()
        cluster_means[col] = (cluster_means[col] - mn) / (mx - mn + 1e-9)

    N = len(radar_features)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    ax.set_facecolor(BG_COLOR)
    for i, (cid, row) in enumerate(cluster_means.iterrows()):
        values = row.tolist() + row.tolist()[:1]
        ax.plot(angles, values, "o-", linewidth=2, color=PALETTE[i % len(PALETTE)],
                label=cluster_label_map.get(cid, f"Cluster {cid}"))
        ax.fill(angles, values, alpha=0.1, color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f.replace("_", "\n") for f in radar_features], fontsize=10)
    ax.set_title("Cluster Profile Radar", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)
    plt.tight_layout()
    save_fig("05_cluster_radar")

# Plot 5 — Return vs Risk scatter coloured by Top-50 status
if "std_dev" in df.columns and "composite_return" in df.columns:
    fig, ax = plt.subplots(figsize=(10, 7))
    is_top = df["investment_score"] >= thresh
    ax.scatter(df.loc[~is_top, "std_dev"], df.loc[~is_top, "composite_return"],
               color=PALETTE[1], alpha=0.3, s=20, label="Other funds", edgecolors="none")
    ax.scatter(df.loc[is_top, "std_dev"], df.loc[is_top, "composite_return"],
               color=PALETTE[3], alpha=0.85, s=60, label=f"Top {TOP_N} funds ★",
               edgecolors="white", linewidths=0.5, zorder=5)
    ax.set_title(f"Return vs Risk — Top {TOP_N} Highlighted", fontsize=14, fontweight="bold")
    ax.set_xlabel("Standard Deviation (Risk)")
    ax.set_ylabel("Composite Return (%)")
    ax.legend(fontsize=10)
    plt.tight_layout()
    save_fig("06_return_vs_risk_top50")

# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 8 — SAVING SCORED DATASET")
print("=" * 60)

# Add full-dataset rank columns for Power BI
df["rank_by_score"]  = df["investment_score"].rank(ascending=False, method="min").astype(int)
if "composite_return" in df.columns:
    df["rank_by_return"] = df["composite_return"].rank(ascending=False, method="min").astype(int)
if "sharpe" in df.columns:
    df["rank_by_sharpe"] = df["sharpe"].rank(ascending=False, method="min").astype(int)
df[f"is_top{TOP_N}"] = df["rank_by_score"] <= TOP_N

df.to_csv(OUTPUT_PATH, index=False)
print(f"  Saved → {OUTPUT_PATH}")
print(f"  Columns added: cluster, risk_cluster_label, investment_score,")
print(f"                 rank_by_score, rank_by_return, rank_by_sharpe, is_top{TOP_N}")
print(f"  Total rows: {len(df):,}")
print("\n" + "=" * 60)
print("  ✅  Script 2 complete. Run powerbi_data_prep.py next.")
print("=" * 60)