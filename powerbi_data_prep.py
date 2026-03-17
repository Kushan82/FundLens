import os
import numpy as np
import pandas as pd

INPUT_PATH = "data/processed/mutual_funds_scored.csv"
PBI_DIR    = "data/powerbi"
os.makedirs(PBI_DIR, exist_ok=True)

TOP_N = 50  
print("=" * 60)
print("  FUNDLENS — POWER BI DATA PREP")
print("=" * 60)

df = pd.read_csv(INPUT_PATH)
print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

top_col = f"is_top{TOP_N}"
if top_col not in df.columns:
    thresh = df["investment_score"].nlargest(TOP_N).min()
    df[top_col] = df["investment_score"] >= thresh
    print(f"  ⚠  '{top_col}' not found — derived from investment_score (threshold={thresh:.1f})\n")

FACT_COLS = [c for c in [
    "scheme_name", "amc_name", "category", "sub_category", "fund_type",
    "risk_level", "risk_cluster_label", "cluster",
    "rating", "fund_manager",
    "returns_1yr", "returns_3yr", "returns_5yr", "composite_return",
    "net_return", "expense_ratio",
    "sharpe", "alpha", "beta", "std_dev", "rari",
    "fund_size_cr", "fund_size_category",
    "min_sip", "min_lumpsum",
    "investment_score", "return_consistency",
    "rank_by_score", "rank_by_return", "rank_by_sharpe",
    top_col,
] if c in df.columns]

fact_funds = df[FACT_COLS].copy()
fact_funds.insert(0, "fund_id", range(1, len(fact_funds) + 1))

if "rank_by_score" not in fact_funds.columns:
    fact_funds["rank_by_score"] = fact_funds["investment_score"].rank(ascending=False, method="min").astype(int)
if "composite_return" in fact_funds.columns and "rank_by_return" not in fact_funds.columns:
    fact_funds["rank_by_return"] = fact_funds["composite_return"].rank(ascending=False, method="min").astype(int)
if "sharpe" in fact_funds.columns and "rank_by_sharpe" not in fact_funds.columns:
    fact_funds["rank_by_sharpe"] = fact_funds["sharpe"].rank(ascending=False, method="min").astype(int)

fact_funds.to_csv(f"{PBI_DIR}/fact_funds.csv", index=False)
print(f"  ✔  fact_funds.csv            → {len(fact_funds):,} rows, {len(fact_funds.columns)} cols")

if "category" in df.columns:
    agg_cols = {
        "fund_count":        ("scheme_name", "count"),
        "avg_1yr_return":    ("returns_1yr", "mean"),
        "avg_3yr_return":    ("returns_3yr", "mean"),
        "avg_5yr_return":    ("returns_5yr", "mean"),
        "avg_expense_ratio": ("expense_ratio", "mean"),
        "avg_sharpe":        ("sharpe", "mean"),
        "avg_score":         ("investment_score", "mean"),
    }
    if top_col in df.columns:
        agg_cols[f"top{TOP_N}_count"] = (top_col, "sum")

    dim_category = (
        df.groupby("category")
        .agg(**agg_cols)
        .round(3)
        .reset_index()
    )
    dim_category["category_id"] = range(1, len(dim_category) + 1)
    dim_category.to_csv(f"{PBI_DIR}/dim_category.csv", index=False)
    print(f"  ✔  dim_category.csv          → {len(dim_category)} categories")

if "amc_name" in df.columns:
    amc_agg_cols = {
        "fund_count":        ("scheme_name", "count"),
        "avg_composite_ret": ("composite_return", "mean"),
        "avg_expense_ratio": ("expense_ratio", "mean"),
        "avg_score":         ("investment_score", "mean"),
        "total_aum_cr":      ("fund_size_cr", "sum"),
    }
    if top_col in df.columns:
        amc_agg_cols[f"top{TOP_N}_count"] = (top_col, "sum")

    dim_amc = (
        df.groupby("amc_name")
        .agg(**amc_agg_cols)
        .round(3)
        .reset_index()
    )
    dim_amc["amc_id"] = range(1, len(dim_amc) + 1)
    dim_amc.to_csv(f"{PBI_DIR}/dim_amc.csv", index=False)
    print(f"  ✔  dim_amc.csv               → {len(dim_amc)} AMCs")

top_n_df = fact_funds[fact_funds[top_col] == True].copy()
top_n_df = top_n_df.sort_values("investment_score", ascending=False).reset_index(drop=True)
top_n_df.index += 1
top_n_df.insert(0, f"top{TOP_N}_rank", top_n_df.index)
top_n_df.to_csv(f"{PBI_DIR}/agg_top{TOP_N}.csv", index=False)
print(f"  ✔  agg_top{TOP_N}.csv             → {len(top_n_df)} top-ranked funds")

if "category" in df.columns:
    return_cols = [c for c in ["returns_1yr", "returns_3yr", "returns_5yr"] if c in df.columns]
    cat_ret = df.groupby("category")[return_cols].mean().round(3).reset_index()
    cat_ret_long = cat_ret.melt(id_vars="category", var_name="horizon", value_name="avg_return")
    cat_ret_long["horizon"] = (
        cat_ret_long["horizon"]
        .str.replace("returns_", "")
        .str.replace("yr", " Year")
    )
    cat_ret_long.to_csv(f"{PBI_DIR}/agg_category_returns.csv", index=False)
    print(f"  ✔  agg_category_returns.csv  → {len(cat_ret_long)} rows (long format)")

if "risk_cluster_label" in df.columns:
    cluster_agg = (
        df.groupby("risk_cluster_label")
        .agg(
            fund_count        = ("scheme_name", "count"),
            avg_return        = ("composite_return", "mean"),
            avg_std_dev       = ("std_dev", "mean"),
            avg_sharpe        = ("sharpe", "mean"),
            avg_expense_ratio = ("expense_ratio", "mean"),
            avg_score         = ("investment_score", "mean"),
        )
        .round(3)
        .reset_index()
    )
    cluster_agg.to_csv(f"{PBI_DIR}/agg_risk_cluster.csv", index=False)
    print(f"  ✔  agg_risk_cluster.csv      → {len(cluster_agg)} clusters")

if "amc_name" in df.columns:
    amc_perf = (
        df.groupby("amc_name")
        .agg(
            fund_count   = ("scheme_name", "count"),
            avg_return   = ("composite_return", "mean"),
            avg_score    = ("investment_score", "mean"),
            total_aum_cr = ("fund_size_cr", "sum"),
        )
        .round(3)
        .reset_index()
        .nlargest(20, "total_aum_cr")
        .reset_index(drop=True)
    )
    amc_perf.to_csv(f"{PBI_DIR}/agg_amc_performance.csv", index=False)
    print(f"  ✔  agg_amc_performance.csv   → {len(amc_perf)} top AMCs by AUM")

if "min_sip" in df.columns and "min_lumpsum" in df.columns and "category" in df.columns:
    sip_lumpsum = (
        df.groupby("category")
        .agg(
            avg_min_sip      = ("min_sip", "mean"),
            avg_min_lumpsum  = ("min_lumpsum", "mean"),
            median_min_sip   = ("min_sip", "median"),
            fund_count       = ("scheme_name", "count"),
        )
        .round(2)
        .reset_index()
    )
    sip_lumpsum.to_csv(f"{PBI_DIR}/agg_sip_lumpsum.csv", index=False)
    print(f"  ✔  agg_sip_lumpsum.csv       → {len(sip_lumpsum)} rows")

if "fund_type" in df.columns and "fund_size_cr" in df.columns:
    fund_type_vol = (
        df.groupby("fund_type")
        .agg(
            fund_count       = ("scheme_name", "count"),
            total_aum_cr     = ("fund_size_cr", "sum"),
            avg_return       = ("composite_return", "mean"),
            avg_expense_ratio= ("expense_ratio", "mean"),
        )
        .round(3)
        .reset_index()
        .sort_values("total_aum_cr", ascending=False)
        .reset_index(drop=True)
    )
    fund_type_vol.to_csv(f"{PBI_DIR}/agg_fund_type_volume.csv", index=False)
    print(f"  ✔  agg_fund_type_volume.csv  → {len(fund_type_vol)} fund types")

print("\n" + "=" * 60)
print("  ALL POWER BI FILES EXPORTED — FundLens")
print("=" * 60)
print(f"  Output directory: {os.path.abspath(PBI_DIR)}/\n")
for f in sorted(os.listdir(PBI_DIR)):
    if f.endswith(".csv"):
        rows = len(pd.read_csv(f"{PBI_DIR}/{f}"))
        print(f"    📄  {f:<40} {rows:>6,} rows")

print()
print("  Next steps in Power BI Desktop:")
print("  1. Home → Get Data → Text/CSV → import all CSVs from data/powerbi/")
print("  2. Model view: link fact_funds.amc_name    → dim_amc.amc_name")
print("  3. Model view: link fact_funds.category    → dim_category.category")
print("  4. Build dashboard — see POWERBI_GUIDE.md for DAX measures")
print()
print("=" * 60)
print("  ✅  Script 3 complete. Run excel_export.py for the Excel workbook.")
print("=" * 60)