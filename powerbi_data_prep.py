import os
import numpy as np
import pandas as pd

INPUT_PATH = "data/processed/mutual_funds_scored.csv"
PBI_DIR    = "data/powerbi"
os.makedirs(PBI_DIR, exist_ok=True)

print("=" * 60)
print("  POWER BI DATA PREP")
print("=" * 60)

df = pd.read_csv(INPUT_PATH)
print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

FACT_COLS = [c for c in [
    "scheme_name", "amc_name", "category", "sub_category", "fund_type",
    "risk_level", "risk_cluster_label", "cluster",
    "rating", "fund_manager",
    "returns_1yr", "returns_3yr", "returns_5yr", "composite_return",
    "net_return", "expense_ratio",
    "sharpe", "alpha", "beta", "std_dev", "rari",
    "fund_size_cr", "fund_size_category",
    "min_sip", "min_lumpsum",
    "investment_score", "return_consistency"
] if c in df.columns]

fact_funds = df[FACT_COLS].copy()

fact_funds.insert(0, "fund_id", range(1, len(fact_funds) + 1))

fact_funds["rank_by_score"]  = fact_funds["investment_score"].rank(ascending=False, method="min").astype(int)
if "composite_return" in fact_funds.columns:
    fact_funds["rank_by_return"] = fact_funds["composite_return"].rank(ascending=False, method="min").astype(int)
if "sharpe" in fact_funds.columns:
    fact_funds["rank_by_sharpe"] = fact_funds["sharpe"].rank(ascending=False, method="min").astype(int)

fact_funds["is_top30"] = fact_funds["rank_by_score"] <= 30

fact_funds.to_csv(f"{PBI_DIR}/fact_funds.csv", index=False)
print(f"  ✔  fact_funds.csv            → {len(fact_funds):,} rows, {len(fact_funds.columns)} cols")

if "category" in df.columns:
    dim_category = (
        df.groupby("category")
        .agg(
            fund_count        = ("scheme_name", "count"),
            avg_1yr_return    = ("returns_1yr", "mean"),
            avg_3yr_return    = ("returns_3yr", "mean"),
            avg_5yr_return    = ("returns_5yr", "mean"),
            avg_expense_ratio = ("expense_ratio", "mean"),
            avg_sharpe        = ("sharpe", "mean"),
            avg_score         = ("investment_score", "mean"),
        )
        .round(3)
        .reset_index()
    )
    dim_category["category_id"] = range(1, len(dim_category) + 1)
    dim_category.to_csv(f"{PBI_DIR}/dim_category.csv", index=False)
    print(f"  ✔  dim_category.csv          → {len(dim_category)} categories")

if "amc_name" in df.columns:
    dim_amc = (
        df.groupby("amc_name")
        .agg(
            fund_count        = ("scheme_name", "count"),
            avg_composite_ret = ("composite_return", "mean"),
            avg_expense_ratio = ("expense_ratio", "mean"),
            avg_score         = ("investment_score", "mean"),
            total_aum_cr      = ("fund_size_cr", "sum"),
            top30_count       = ("is_top30" if "is_top30" in fact_funds.columns else "investment_score",
                                  lambda x: (fact_funds.loc[x.index, "is_top30"].sum()
                                             if "is_top30" in fact_funds.columns else 0))
        )
        .round(3)
        .reset_index()
    )
    dim_amc["amc_id"] = range(1, len(dim_amc) + 1)
    dim_amc.to_csv(f"{PBI_DIR}/dim_amc.csv", index=False)
    print(f"  ✔  dim_amc.csv               → {len(dim_amc)} AMCs")

top30 = fact_funds[fact_funds["is_top30"] == True].copy() if "is_top30" in fact_funds.columns \
        else fact_funds.nlargest(30, "investment_score").copy()

top30 = top30.sort_values("investment_score", ascending=False).reset_index(drop=True)
top30.index += 1
top30.insert(0, "top30_rank", top30.index)
top30.to_csv(f"{PBI_DIR}/agg_top30.csv", index=False)
print(f"  ✔  agg_top30.csv             → {len(top30)} top-ranked funds")

if "category" in df.columns:
    return_cols = [c for c in ["returns_1yr", "returns_3yr", "returns_5yr"] if c in df.columns]
    cat_ret = df.groupby("category")[return_cols].mean().round(3).reset_index()
    cat_ret_long = cat_ret.melt(id_vars="category", var_name="horizon", value_name="avg_return")
    cat_ret_long["horizon"] = cat_ret_long["horizon"].str.replace("returns_", "").str.replace("yr", " Year")
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
            fund_count    = ("scheme_name", "count"),
            avg_return    = ("composite_return", "mean"),
            avg_score     = ("investment_score", "mean"),
            total_aum_cr  = ("fund_size_cr", "sum"),
        )
        .round(3)
        .reset_index()
        .nlargest(20, "total_aum_cr")
        .reset_index(drop=True)
    )
    amc_perf.to_csv(f"{PBI_DIR}/agg_amc_performance.csv", index=False)
    print(f"  ✔  agg_amc_performance.csv   → {len(amc_perf)} top AMCs")

print("\n" + "=" * 60)
print("  ALL POWER BI FILES EXPORTED")
print("=" * 60)
print(f"  Output directory: {os.path.abspath(PBI_DIR)}/")
print()
for f in sorted(os.listdir(PBI_DIR)):
    rows = len(pd.read_csv(f"{PBI_DIR}/{f}"))
    print(f"    📄  {f:<35} {rows:>6,} rows")

print()
print("  Next steps in Power BI Desktop:")
print("  1. Home → Get Data → Text/CSV → import all 7 CSVs")
print("  2. Model view → link fact_funds.amc_name  → dim_amc.amc_name")
print("  3. Model view → link fact_funds.category  → dim_category.category")
print("  4. Build your dashboard pages (see POWERBI_GUIDE.md)")
print()
print("=" * 60)
print("  ✅  Script 3 complete. Open Power BI Desktop to build dashboard.")
print("=" * 60)