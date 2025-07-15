import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv(r"C:\Users\k\kaggle-portfolio\study\dataset\train.csv")
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())

bins = [0,12,18,60,100]
labels = ['Child','Teenager', 'Adult','Senior']
train_df['AgeGroup'] = pd.cut(train_df['Age'], bins=bins, labels=labels, right=False)

# 年代グループごとの生存率を計算
print("\n--- Survival Rate by Age Group ---")
age_group_survival = train_df[["AgeGroup", "Survived"]].groupby(
    ["AgeGroup"], as_index=False, observed=False
).mean()
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
plt.savefig(r'C:\Users\k\kaggle-portfolio\study\age_survival_analysis.png')

print(r"Graph saved to: C:\Users\k\kaggle-portfolio\study\age_survival_analysis.png")