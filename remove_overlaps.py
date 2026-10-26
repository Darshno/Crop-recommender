import pandas as pd

file = "independent_data.csv"

df = pd.read_csv(file)

df["label"] = df["label"].str.strip().str.lower()

df.to_csv(file, index=False)

print("Updated crop labels:")
print(sorted(df["label"].unique()))