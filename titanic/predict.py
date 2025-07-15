

import pandas as pd
from sklearn.linear_model import LogisticRegression

# --- 1. データの読み込み ---
print("Loading preprocessed data...")
train_data = pd.read_csv("C:/Users/kesti/kaggle/dataset/titanic/train_processed.csv")
test_data = pd.read_csv("C:/Users/kesti/kaggle/dataset/titanic/test_processed.csv")

# --- 2. 学習データ全体でモデルを再学習 ---
print("Retraining the model on the full training dataset...")

# 特徴量 (X) とターゲット (y) を定義
X_train = train_data.drop(["Survived", "PassengerId"], axis=1)
y_train = train_data["Survived"]

# テストデータの特徴量を定義
# PassengerIdは提出に必要なので保持しておく
X_test = test_data.drop("PassengerId", axis=1)

# ロジスティック回帰モデルを学習
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Model training complete.")

# --- 3. テストデータで予測を実行 ---
print("Predicting on the test data...")
test_predictions = model.predict(X_test)

# --- 4. 提出ファイルの作成 ---
print("Creating submission file...")

# PassengerIdと予測結果をデータフレームにまとめる
submission = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": test_predictions
})

# CSVファイルとして保存
submission_path = "C:/Users/kesti/kaggle/submission.csv"
submission.to_csv(submission_path, index=False)

print("\n--- Submission File Creation Complete ---")
print(f"Submission file saved to: {submission_path}")
print("\n--- Submission File Head ---")
print(submission.head())

