

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. データの読み込み
print("Loading preprocessed data...")
data = pd.read_csv("C:/Users/kesti/kaggle/dataset/titanic/train_processed.csv")

# 2. 特徴量 (X) とターゲット (y) の分離
# PassengerIdは予測に不要なため除外
X = data.drop(["Survived", "PassengerId"], axis=1)
y = data["Survived"]

# 3. 学習データと検証データに分割
# データを80%の学習用と20%の検証用に分割
# random_stateを固定することで、毎回同じ分割結果になり、再現性を確保
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")

# 4. モデルの選択と学習
print("\nTraining a Logistic Regression model...")
model = LogisticRegression(max_iter=1000) # max_iterを増やして収束を確実に
model.fit(X_train, y_train)

# 5. モデルの評価
print("Evaluating the model...")
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

print("\n--- Model Evaluation ---")
print(f"Accuracy on validation set: {accuracy:.4f}")

