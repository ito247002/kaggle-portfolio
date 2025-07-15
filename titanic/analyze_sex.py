
import pandas as pd

# train.csvを読み込み
train_df = pd.read_csv("C:/Users/kesti/kaggle/dataset/titanic/train.csv")

# 性別ごとの生存率を計算して表示
print("--- Survival Rate by Sex ---")
print(train_df[["Sex", "Survived"]].groupby(["Sex"], as_index=False).mean().sort_values(by="Survived", ascending=False))
