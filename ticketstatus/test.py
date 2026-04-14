import pandas as pd

df = pd.read_csv("railway_dataset.csv")

print(df["Confirmation Status"].value_counts())