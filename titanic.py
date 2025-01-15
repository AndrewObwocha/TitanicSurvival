# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19"
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import matplotlib.pyplot as plt
import seaborn as sns
import miceforest as mf

# %%
df = pd.read_csv('/kaggle/input/titanic/train.csv')
df.head()

# %%
kernel = mf.KernelDataSet(
    df,
    save_all_iterations=True,
    random_state=1991
)

# %%
df.isnull().sum()

# %%
df = df.dropna(subset='Embarked')
df.isnull().sum()

# %%
df['Age'].fillna(df['Age'].mean(), inplace=True)
df.isnull().sum()

# %%
df = df.drop('Cabin', axis=1)
df.isnull().sum()

# %%
df[df.duplicated()]

# %%
df.columns

# %%
df['Name'].head(50)

# %%
df = df.drop('PassengerId', axis=1)
df.columns

# %%
# df['FamMembers'] = df['SibSp'] + df['Parch']
df = df.drop(['SibSp', 'Parch'], axis=1)
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
numerical_df = df[numerical_cols]
correlation_matrix = numerical_df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation of features')
plt.show()

# %%
high_correlations = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.3:
            feature1 = correlation_matrix.columns[i]
            feature2 = correlation_matrix.columns[j]
            correlation = correlation_matrix.iloc[i, j]
            high_correlations.append({
                'Feature 1': feature1,
                'Feature 2': feature2,
                'Correlation': correlation
            })
high_corr_df = pd.DataFrame(high_correlations).sort_values(
        by='Correlation', 
        key=abs,
        ascending=False
    )
high_corr_df.head()

# %%
import re

def extract_title(text,
                 titles=None,
                 case_sensitive=False):
    if titles is None:
        titles = ['Mr', 'Mrs', 'Ms', 'Miss', 'Dr', 'Prof', 'Rev', 'Sir',
                'Master', 'Dame', 'Lady', 'Lord', 'Captain', 'Don', 'Mme',
                 'Major', 'Jonkheer', 'Countess', 'Capt', 'Col', 'Mlle']
    
    patterns = []
    for title in titles:
        base_pattern = f'{title}\.?'
        base_pattern += r'\s*'
        patterns.append(base_pattern)
    
    full_pattern = '|'.join(patterns)
    full_pattern = r'\b(' + full_pattern + r')\b'
    
    flag = 0 if case_sensitive else re.IGNORECASE
    regex = re.compile(full_pattern, flag)

    match = regex.search(text)
    if match:
        title = match.group(1)
        title = title.strip()
        title = title.title()
        if not title.endswith('.'):
            title += '.'
        return title
    return None

def create_title_feature(df, 
                         old_feature='Name', 
                         new_feature='Title'):

    df[new_feature] = df[old_feature].apply(lambda x: extract_title(str(x)))
    return df

df = create_title_feature(df)

df.columns
   

# %%
df.isnull().sum()

# %%
df.head()

# %%
df.describe()

# %%
df = df.drop('Name', axis=1)
df.columns

# %%
df.head()

# %%
df['Ticket'].nunique()

# %%
df = df.drop('Ticket', axis=1)

# %%
df.describe()

# %%
df['Sex'] = df['Sex'].map({'male' : 1, 'female' : 0})
df['Embarked'] = df['Embarked'].map({'S' : 2, 'C' : 1, 'Q' : 0})
for i in range(3):
    df = df.drop(df['Fare'].idxmax())
df.describe()

# %%

from category_encoders import BinaryEncoder

encoder = BinaryEncoder(cols=['Title'])
df = encoder.fit_transform(df)

df.head()
df.describe()

# %%
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.histplot(df['Age'], bins=10, kde=True)
plt.title('Boxplot of Age')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', data=df)
plt.title('Countplot for Survived')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.countplot(x='Pclass', data=df)
plt.title('Countplot for Social Class')
plt.xlabel('Social Class')
plt.ylabel('Count')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.countplot(x='Sex', data=df)
plt.title('Countplot for Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()

# %%
'''
plt.figure(figsize=(8, 6))
sns.countplot(x='SibSp', data=df)
plt.title('Countplot for Siblings/Spouse')
plt.xlabel('Number of Siblings/Spouse')
plt.ylabel('Count')
plt.show()
'''

# %%
'''
plt.figure(figsize=(8, 6))
sns.countplot(x='Parch', data=df)
plt.title('Countplot for Parents/Children')
plt.xlabel('Number of Parents/Children')
plt.ylabel('Count')
plt.show()
'''

# %%
plt.figure(figsize=(8, 6))
sns.histplot(df['Fare'], bins=15, kde=True)
plt.title('Histogram for Fare')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.countplot(x='Embarked', data=df)
plt.title('Countplot for Embarked Location')
plt.xlabel('Embarked Location')
plt.ylabel('Count')
plt.show()

# %%
df[df['Fare'] > 65.5]

# %%
# sns.pairplot(df)
# plt.show()

# %%
correlation_matrix = df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation of features')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', hue='Pclass', data=df)
plt.title('Distribution of Pclass within each Survived category')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

# %%
plt.figure(figsize=(10, 6))
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Distribution of Sex within each Survived category')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

# %%
plt.figure(figsize=(10, 6))
sns.boxplot(x='Survived', y='Fare', data=df)
plt.title('Distribution of Fare for each Survived category')
plt.xlabel('Survived')
plt.ylabel('Fare Paid')
plt.show()

# %%
plt.figure(figsize=(10, 6))
sns.countplot(x='Survived', hue='Embarked', data=df)
plt.title('Distribution of Embarked Location for each Survived category')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.boxplot(x='Survived', y='Age', data=df)
plt.title('Distribution of Age for each Survived category')
plt.xlabel('Survived')
plt.ylabel('Age')
plt.show()

# %%
'''
plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', hue='SibSp', data=df)
plt.title('Distribution of Sibling/Spouse for each Survived category')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()
'''

# %%
'''
plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', hue='Parch', data=df)
plt.title('Distribution of Parents/Children for each Survived category')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()
'''

# %%
df = df.drop('Age', axis=1)
# df = df.drop('SibSp', axis=1)
# df = df.drop('Parch', axis=1)
df = df.drop('Fare', axis=1)

df.describe()

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

print(classification_report(y_test, y_pred))

