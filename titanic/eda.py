

import pandas as pd

# 

train_df = pd.read_csv("C:/Users/kesti/kaggle/dataset/titanic/train.csv")
test_df = pd.read_csv("C:/Users/kesti/kaggle/dataset/titanic/test.csv")

# train.csv
print("--- train.csv info ---")
train_df.info()
print("\n" + "="*30 + "\n")

# train.csv
print("--- train.csv describe ---")
print(train_df.describe())
print("\n" + "="*30 + "\n")

# test.csv
print("--- test.csv info ---")
test_df.info()
print("\n" + "="*30 + "\n")

# test.csv
print("--- test.csv describe ---")
print(test_df.describe())
print("\n" + "="*30 + "\n")

# 
print("--- Missing Values in train.csv ---")
print(train_df.isnull().sum())
print("\n" + "="*30 + "\n")

print("--- Missing Values in test.csv ---")
print(test_df.isnull().sum())

