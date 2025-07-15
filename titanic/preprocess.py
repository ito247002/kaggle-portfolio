

import pandas as pd
import numpy as np

# --- データ読み込み ---
def load_data():
    train_df = pd.read_csv("C:/Users/kesti/kaggle/dataset/titanic/train.csv")
    test_df = pd.read_csv("C:/Users/kesti/kaggle/dataset/titanic/test.csv")
    return train_df, test_df

# --- 前処理関数 ---
def preprocess(df, is_train=True):
    # 1. 欠損値の補完
    # Age: 中央値で補完
    df['Age'].fillna(df['Age'].median(), inplace=True)
    # Embarked: 最頻値で補完 (学習データにのみ適用)
    if is_train:
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    # Fare: 中央値で補完 (テストデータにのみ適用)
    else:
        df['Fare'].fillna(df['Fare'].median(), inplace=True)

    # 2. カテゴリ変数の数値化
    # Sex: male=0, female=1
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}).astype(int)
    # Embarked: S=0, C=1, Q=2
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    # 3. 新しい特徴量の作成 (FamilySize)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # 4. 不要な列の削除
    # Cabinは欠損が多すぎるため削除
    # Name, Ticketも今回は使用しないため削除
    # PassengerIdはテストデータの提出に必要なので、ここでは削除しない
    df.drop(['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis=1, inplace=True)

    return df

# --- メイン処理 ---
if __name__ == "__main__":
    # データの読み込み
    train_df, test_df = load_data()
    
    # テストデータのPassengerIdを保持
    test_passenger_id = test_df['PassengerId']

    # 前処理の実行
    print("Preprocessing training data...")
    train_processed = preprocess(train_df.copy(), is_train=True)
    
    print("Preprocessing testing data...")
    test_processed = preprocess(test_df.copy(), is_train=False)

    # 処理済みデータの保存
    train_processed.to_csv("C:/Users/kesti/kaggle/dataset/titanic/train_processed.csv", index=False)
    test_processed.to_csv("C:/Users/kesti/kaggle/dataset/titanic/test_processed.csv", index=False)

    print("\nPreprocessing complete.")
    print("Processed files saved to:")
    print("- C:/Users/kesti/kaggle/dataset/titanic/train_processed.csv")
    print("- C:/Users/kesti/kaggle/dataset/titanic/test_processed.csv")

    print("\n--- Processed Training Data Head ---")
    print(train_processed.head())

