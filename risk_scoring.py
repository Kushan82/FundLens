import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

INPUT_PATH  = "data/processed/mutual_funds_clean.csv"
OUTPUT_PATH = "data/processed/mutual_funds_scored.csv"
PLOTS_DIR   = "outputs/ml_plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

PALETTE   = ["#0D3349", "#1A6B8A", "#27C4A0", "#F5A623", "#E84855"]
BG_COLOR  = "#F8F9FB"
GRID_COLOR = "#E2E8F0"
plt.rcParams.update({
    "figure.facecolor": BG_COLOR, "axes.facecolor": BG_COLOR,
    "axes.grid": True, "grid.color": GRID_COLOR, "grid.linewidth": 0.8,
    "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
})

print("=" * 60)
print("  STEP 1 — LOADING CLEAN DATA")
print("=" * 60)
df = pd.read_csv(INPUT_PATH)
print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

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

print("=" * 60)
print("  STEP 3 — FINDING OPTIMAL NUMBER OF CLUSTERS")
print("=" * 60)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

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

print("\n" + "=" * 60)
print(f"  STEP 4 — FITTING KMEANS (k={best_k})")
print("=" * 60)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(n_clusters=best_k, random_state=42, n_init=15))
])
pipeline.fit(X)
df["cluster"] = pipeline.named_steps["kmeans"].labels_

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

print("\n" + "=" * 60)
print("  STEP 6 — INVESTMENT ATTRACTIVENESS SCORE")
print("=" * 60)

def minmax_norm(series):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mn) / (mx - mn)

score = pd.Series(np.zeros(len(df)), index=df.index)
weights = {}

if "composite_return" in df.columns:
    score += minmax_norm(df["composite_return"]) * 0.30
    weights["composite_return"] = 0.30
if "sharpe" in df.columns:
    score += minmax_norm(df["sharpe"]) * 0.25
    weights["sharpe"] = 0.25
if "expense_ratio" in df.columns:
    score += minmax_norm(-df["expense_ratio"]) * 0.15  
    weights["expense_ratio (inverted)"] = 0.15
if "alpha" in df.columns:
    score += minmax_norm(df["alpha"]) * 0.15
    weights["alpha"] = 0.15
if "rari" in df.columns:
    score += minmax_norm(df["rari"]) * 0.10
    weights["rari"] = 0.10
if "return_consistency" in df.columns:
    score += minmax_norm(df["return_consistency"]) * 0.05
    weights["return_consistency"] = 0.05

df["investment_score"] = (score * 100).round(2)

print("  Score weights used:")
for k, v in weights.items():
    print(f"    {k:<35} {v*100:.0f}%")

print(f"\n  Score stats:")
print(f"    Mean  : {df['investment_score'].mean():.1f}")
print(f"    Median: {df['investment_score'].median():.1f}")
print(f"    Min   : {df['investment_score'].min():.1f}")
print(f"    Max   : {df['investment_score'].max():.1f}")

print("\n" + "=" * 60)
print("  STEP 7 — TOP 30 FUNDS")
print("=" * 60)

top30_cols = [c for c in ["scheme_name", "amc_name", "category", "risk_cluster_label",
                            "composite_return", "sharpe", "expense_ratio",
                            "investment_score", "fund_size_cr"] if c in df.columns]
top30 = df.nlargest(30, "investment_score")[top30_cols].reset_index(drop=True)
top30.index += 1
print(top30[["scheme_name", "risk_cluster_label", "composite_return", "investment_score"]
            [:len(top30_cols)]].head(10).to_string())
print("  ... (top 30 saved in output CSV)\n")


def save_fig(name):
    path = f"{PLOTS_DIR}/{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"  Saved → {path}")


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

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(df["investment_score"], bins=50, color=PALETTE[2], alpha=0.85, edgecolor="white")
ax.axvline(df["investment_score"].mean(), color=PALETTE[4], linewidth=2,
           linestyle="--", label=f"Mean: {df['investment_score'].mean():.1f}")
thresh = top30["investment_score"].min()
ax.axvline(thresh, color=PALETTE[3], linewidth=2,
           linestyle="-.", label=f"Top 30 threshold: {thresh:.1f}")
ax.set_title("Investment Attractiveness Score Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Score (0–100)"); ax.set_ylabel("Number of Funds")
ax.legend()
plt.tight_layout()
save_fig("03_investment_score_distribution")

if "scheme_name" in df.columns:
    top20 = df.nlargest(20, "investment_score")[["scheme_name", "investment_score", "cluster"]].copy()
    top20["short_name"] = top20["scheme_name"].str[:45]
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = [PALETTE[int(c) % len(PALETTE)] for c in top20["cluster"]]
    bars = ax.barh(top20["short_name"][::-1], top20["investment_score"][::-1],
                   color=colors[::-1], edgecolor="white", height=0.7)
    for bar in bars:
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.1f}", va="center", fontsize=9, fontweight="bold")
    ax.set_title("Top 20 Funds by Investment Attractiveness Score", fontsize=14, fontweight="bold")
    ax.set_xlabel("Investment Score (0–100)")
    ax.set_xlim(0, top20["investment_score"].max() + 8)
    plt.tight_layout()
    save_fig("04_top20_funds_bar")

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

print("\n" + "=" * 60)
print("  STEP 8 — SAVING SCORED DATASET")
print("=" * 60)

df.to_csv(OUTPUT_PATH, index=False)
print(f"  Saved → {OUTPUT_PATH}")
print(f"  Columns added: cluster, risk_cluster_label, investment_score")
print(f"  Total rows: {len(df):,}")
print("\n" + "=" * 60)
print("  ✅  Script 2 complete. Run 03_powerbi_data_prep.py next.")
print("=" * 60)