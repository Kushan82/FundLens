import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

RAW_DATA_PATH   = "data/raw/mutual_funds_india.csv"   
CLEAN_DATA_PATH = "data/processed/mutual_funds_clean.csv"
PLOTS_DIR       = "outputs/eda_plots"

os.makedirs("data/processed", exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

PALETTE   = ["#0D3349", "#1A6B8A", "#27C4A0", "#F5A623", "#E84855"]
BG_COLOR  = "#F8F9FB"
GRID_COLOR = "#E2E8F0"

plt.rcParams.update({
    "figure.facecolor":  BG_COLOR,
    "axes.facecolor":    BG_COLOR,
    "axes.grid":         True,
    "grid.color":        GRID_COLOR,
    "grid.linewidth":    0.8,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

print("=" * 60)
print("  STEP 1 — LOADING DATA")
print("=" * 60)

df = pd.read_csv(RAW_DATA_PATH)
print(f"  Rows    : {df.shape[0]:,}")
print(f"  Columns : {df.shape[1]}")
print(f"\n  Columns : {list(df.columns)}\n")
print("=" * 60)
print("  STEP 2 — STANDARDISING COLUMNS")
print("=" * 60)

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(r"[\s/\-]+", "_", regex=True)
    .str.replace(r"[^a-z0-9_]", "", regex=True)
)
print(f"  Cleaned columns: {list(df.columns)}\n")

rename_map = {
    "scheme_name":           "scheme_name",
    "amc_name":              "amc_name",
    "category":              "category",
    "sub_category":          "sub_category",
    "fund_type":             "fund_type",
    "rating":                "rating",
    "risk":                  "risk_level",
    "min_sip":               "min_sip",
    "min_lumpsum":           "min_lumpsum",
    "expense_ratio":         "expense_ratio",
    "fund_size_cr":          "fund_size_cr",
    "fund_manager":          "fund_manager",
    "returns_1yr":           "returns_1yr",
    "returns_3yr":           "returns_3yr",
    "returns_5yr":           "returns_5yr",
    "sortino":               "sortino",
    "alpha":                 "alpha",
    "sd":                    "std_dev",
    "beta":                  "beta",
    "sharpe":                "sharpe",
}

actual_rename = {k: v for k, v in rename_map.items() if k in df.columns}
df.rename(columns=actual_rename, inplace=True)
print(f"  Renamed {len(actual_rename)} columns.\n")


print("=" * 60)
print("  STEP 3 — MISSING VALUE TREATMENT")
print("=" * 60)

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
miss_df = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
miss_df = miss_df[miss_df["missing_count"] > 0].sort_values("missing_pct", ascending=False)
print(miss_df.to_string())

return_cols = [c for c in ["returns_1yr", "returns_3yr", "returns_5yr"] if c in df.columns]
df.dropna(subset=return_cols, how="all", inplace=True)

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
for col in cat_cols:
    mode_val = df[col].mode()
    df[col].fillna(mode_val[0] if len(mode_val) > 0 else "Unknown", inplace=True)

print(f"\n  Remaining nulls: {df.isnull().sum().sum()}")
print(f"  Rows after cleaning: {len(df):,}\n")

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

if "rating" in df.columns:
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0).clip(0, 5)

print("=" * 60)
print("  STEP 4 — FEATURE ENGINEERING")
print("=" * 60)

available_returns = [c for c in ["returns_1yr", "returns_3yr", "returns_5yr"] if c in df.columns]
weights = {"returns_1yr": 0.20, "returns_3yr": 0.35, "returns_5yr": 0.45}
w_sum = sum(weights[c] for c in available_returns)
df["composite_return"] = sum(
    df[c] * (weights[c] / w_sum) for c in available_returns
)
print("  ✔  composite_return          (weighted blend of 1/3/5yr returns)")

# 6b. Sharpe Ratio (if not already present)
if "sharpe" not in df.columns and "std_dev" in df.columns and len(available_returns) > 0:
    RISK_FREE = 6.5   # approx Indian 10-yr Gsec yield %
    df["sharpe"] = (df["composite_return"] - RISK_FREE) / df["std_dev"].replace(0, np.nan)
    df["sharpe"].fillna(0, inplace=True)
    print("  ✔  sharpe                    (computed from composite_return & std_dev)")
else:
    print("  ✔  sharpe                    (already in dataset)")

# 6c. Risk-Adjusted Return Index (RARI)
if "std_dev" in df.columns:
    df["rari"] = df["composite_return"] / df["std_dev"].replace(0, np.nan)
    df["rari"].fillna(0, inplace=True)
    print("  ✔  rari                      (return per unit of risk)")

# 6d. Expense-Adjusted Return
if "expense_ratio" in df.columns:
    df["net_return"] = df["composite_return"] - df["expense_ratio"]
    print("  ✔  net_return                (composite_return − expense_ratio)")

# 6e. Fund Size Category
if "fund_size_cr" in df.columns:
    df["fund_size_category"] = pd.cut(
        df["fund_size_cr"],
        bins=[0, 500, 5_000, 20_000, float("inf")],
        labels=["Small (<500 Cr)", "Mid (500–5K Cr)", "Large (5K–20K Cr)", "Giant (>20K Cr)"]
    )
    print("  ✔  fund_size_category        (binned AUM)")

if len(available_returns) > 1:
    df["return_consistency"] = df[available_returns].std(axis=1)
    df["return_consistency"] = df["return_consistency"].max() - df["return_consistency"]  
    print("  ✔  return_consistency        (higher = more stable across time horizons)")

print()

print("=" * 60)
print("  STEP 5 — EDA VISUALISATIONS")
print("=" * 60)


def save_fig(name):
    path = f"{PLOTS_DIR}/{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"  Saved → {path}")


fig, axes = plt.subplots(1, len(available_returns), figsize=(5 * len(available_returns), 5))
if len(available_returns) == 1:
    axes = [axes]
for ax, col, color in zip(axes, available_returns, PALETTE):
    ax.hist(df[col].clip(-10, 50), bins=40, color=color, alpha=0.85, edgecolor="white")
    ax.axvline(df[col].mean(), color="#E84855", linewidth=1.8, linestyle="--", label=f"Mean: {df[col].mean():.1f}%")
    ax.set_title(col.replace("_", " ").title(), fontsize=13, fontweight="bold")
    ax.set_xlabel("Return (%)", fontsize=10)
    ax.set_ylabel("Number of Funds", fontsize=10)
    ax.legend(fontsize=9)
fig.suptitle("Return Distributions Across Time Horizons", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig("01_return_distributions")

if "risk_level" in df.columns:
    risk_counts = df["risk_level"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(risk_counts.index, risk_counts.values, color=PALETTE[:len(risk_counts)], edgecolor="white")
    for bar in bars:
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f"{int(bar.get_width()):,}", va="center", fontsize=10, fontweight="bold")
    ax.set_title("Fund Distribution by Risk Level", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Funds")
    plt.tight_layout()
    save_fig("02_risk_level_distribution")

if "amc_name" in df.columns:
    top_amcs = df["amc_name"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_amcs.index[::-1], top_amcs.values[::-1], color=PALETTE[1], edgecolor="white")
    ax.set_title("Top 15 AMCs by Number of Schemes", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Schemes")
    plt.tight_layout()
    save_fig("03_top_amc_by_fund_count")

if "expense_ratio" in df.columns:
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        df["expense_ratio"], df["composite_return"],
        alpha=0.45, c=PALETTE[2], edgecolors="white", linewidths=0.4, s=30
    )
    z = np.polyfit(df["expense_ratio"].clip(0, 5), df["composite_return"], 1)
    p = np.poly1d(z)
    xline = np.linspace(df["expense_ratio"].clip(0,5).min(), df["expense_ratio"].clip(0,5).max(), 200)
    ax.plot(xline, p(xline), color=PALETTE[3], linewidth=2, linestyle="--", label="Trend")
    ax.set_title("Expense Ratio vs Composite Return", fontsize=14, fontweight="bold")
    ax.set_xlabel("Expense Ratio (%)")
    ax.set_ylabel("Composite Return (%)")
    ax.legend()
    plt.tight_layout()
    save_fig("04_expense_ratio_vs_return")

if "sharpe" in df.columns and "category" in df.columns:
    top_cats = df["category"].value_counts().head(6).index
    fig, ax = plt.subplots(figsize=(10, 6))
    data_by_cat = [df[df["category"] == c]["sharpe"].clip(-2, 4).dropna().values for c in top_cats]
    bp = ax.boxplot(data_by_cat, patch_artist=True, notch=False, vert=True,
                    medianprops={"color": "#E84855", "linewidth": 2})
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticklabels(top_cats, rotation=20, ha="right", fontsize=9)
    ax.set_title("Sharpe Ratio Distribution by Fund Category", fontsize=14, fontweight="bold")
    ax.set_ylabel("Sharpe Ratio")
    plt.tight_layout()
    save_fig("05_sharpe_by_category")

corr_cols = [c for c in ["returns_1yr", "returns_3yr", "returns_5yr",
                          "sharpe", "expense_ratio", "std_dev", "beta",
                          "alpha", "rari", "composite_return"] if c in df.columns]
if len(corr_cols) >= 4:
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax,
                annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig("06_correlation_heatmap")

if "fund_size_category" in df.columns:
    size_counts = df["fund_size_category"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        size_counts, labels=size_counts.index, autopct="%1.1f%%",
        colors=PALETTE[:len(size_counts)], startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 2}
    )
    for t in autotexts:
        t.set_fontsize(10)
        t.set_fontweight("bold")
    ax.set_title("Fund Distribution by AUM Size", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig("07_fund_size_distribution")

print("\n" + "=" * 60)
print("  STEP 6 — SAVING CLEANED DATASET")
print("=" * 60)

df.to_csv(CLEAN_DATA_PATH, index=False)
print(f"  Saved → {CLEAN_DATA_PATH}")
print(f"  Final dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\n  New features added: composite_return, sharpe, rari,")
print(f"  net_return, fund_size_category, return_consistency\n")
print("=" * 60)
print("  ✅  Script 1 complete. Run 02_ml_risk_scoring.py next.")
print("=" * 60)