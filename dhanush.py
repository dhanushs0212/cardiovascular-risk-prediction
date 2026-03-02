import pandas as pd

df = pd.read_csv("heart.csv")

print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nTarget distribution:\n", df["target"].value_counts())
