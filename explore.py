import pandas as pd

df = pd.read_csv('Fifa_23_Players_Data.csv', low_memory=False)
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print(df.info())