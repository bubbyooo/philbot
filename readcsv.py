from sklearn.model_selection import train_test_split
import pandas as pd

"""
This file gives an overview of the dataset and training information.
"""

df = pd.read_csv("data/raw/Dr_Dragon_moves_dataset.csv")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

train_df = train_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

print("colums:")
print(df.columns)
print()


print("rows:", len(df))
print("total unique moves:", df["move_uci"].nunique())
print()

print("train rows:", len(train_df))
print("test rows:", len(test_df))
print("train unique games:", train_df["game_id"].nunique())
print("test unique games:", test_df["game_id"].nunique())
print("train unique moves:", train_df["move_uci"].nunique())
print()

most_common = train_df["move_uci"].value_counts().idxmax()
baseline_acc = (test_df["move_uci"] == most_common).mean()

print("Most common move baseline accuracy:", baseline_acc)