import pandas as pd
df_train = pd.read_csv('study/dataset/train.csv')
df_train.info()
print(df_train.isnull().sum())
print(df_train[["Sex", "Survived"]].groupby(["Sex"], as_index=False).mean().sort_values(by="Survived", ascending=False))
print(df_train[["Pclass","Survived"]].groupby(["Pclass"], as_index=False).mean().sort_values(by="Survived",ascending=False))