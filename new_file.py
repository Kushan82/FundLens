import pandas as pd



#print(df.columns)
#print(df.head())

#category_aum = (
 #   df.groupby("category", as_index=False)["fund_size_cr"]
  #  .sum()
   # .rename(columns={"fund_size_cr": "total_aum_cr"}))

#category_aum["category"] = category_aum["category"].str.strip()

#category_aum.to_csv("category_aum.csv", index=False)



df = pd.read_csv("category_aum.csv")

print(df.head())
print("\nColumns:", df.columns.tolist())
print("\nRow count:", len(df))
print("\nUnique categories:", df["category"].nunique())
print("\nAUM summary:\n", df["total_aum_cr"].describe())