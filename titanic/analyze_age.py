import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# train.csvを読み込み
train_df = pd.read_csv("C:/Users/kesti/kaggle/dataset/titanic/train.csv")

# --- 年齢と生存率の分析 ---
print("--- Analysis of Age and Survival ---")

# 1. 年齢の欠損値を中央値で補完
# 注意: これは仮の対応です。より良い方法はステップ4で検討します。
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)

# 2. 年齢をカテゴリに分割
bins = [0, 12, 18, 60, 100] # 0-12(子供), 13-18(少年), 19-60(大人), 61-100(高齢者)
labels = ['Child', 'Teenager', 'Adult', 'Senior']
train_df['AgeGroup'] = pd.cut(train_df['Age'], bins=bins, labels=labels, right=False)

# 3. 年代グループごとの生存率を計算
print("\n--- Survival Rate by Age Group ---")
age_group_survival = train_df[["AgeGroup", "Survived"]].groupby(["AgeGroup"], as_index=False).mean()
print(age_group_survival)

# 4. 可視化
# seabornのスタイルを設定
sns.set(style="whitegrid")

# 年齢の分布と生存の関係を可視化するグラフを作成
plt.figure(figsize=(12, 5))

# 全体の年齢分布（ヒストグラム）
plt.subplot(1, 2, 1)
sns.histplot(data=train_df, x='Age', hue='Survived', multiple='stack', kde=True)
plt.title('Age Distribution by Survival')
plt.xlabel('Age')
plt.ylabel('Count')

# 年代グループごとの生存率（棒グラフ）
plt.subplot(1, 2, 2)
sns.barplot(x='AgeGroup', y='Survived', data=age_group_survival)
plt.title('Survival Rate by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Survival Rate')

# グラフを保存して表示
plt.tight_layout()
plt.savefig('C:/Users/kesti/kaggle/age_survival_analysis.png')

print("\nGraph saved to: C:/Users/kesti/kaggle/age_survival_analysis.png")