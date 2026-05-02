# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import pickle

# data collection
df = pd.read_csv('insurance.csv')

# basic checks
print(df.head())
print(df.shape)
print(df.info())

# statistical summary
print(df.describe())

# EDA
print(df.isnull().sum())
print(df.duplicated().sum())

# plots (same as yours)
sns.set()

plt.figure(figsize=(6,3))
sns.histplot(df['age'], kde=True)
plt.title('Distribution of age')
plt.show()

plt.figure(figsize=(6,3))
sns.countplot(x=df['sex'])
plt.title('Distribution of sex')
plt.show()

plt.figure(figsize=(6,3))
sns.histplot(df['bmi'], kde=True)
plt.title('Distribution of bmi')
plt.show()

plt.figure(figsize=(6,3))
sns.histplot(df['children'], kde=True)
plt.title('Distribution of children')
plt.show()

plt.figure(figsize=(6,3))
sns.countplot(x=df['smoker'])
plt.title('Distribution of smoker')
plt.show()

plt.figure(figsize=(6,3))
sns.countplot(x=df['region'])
plt.title('Distribution of region')
plt.show()

plt.figure(figsize=(6,3))
sns.histplot(df['charges'], kde=True)
plt.title('Distribution of charges')
plt.show()

# ==========================
# 🔥 IMPROVEMENT 1: encoding
# ==========================
df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

# ==========================
# 🔥 IMPROVEMENT 2: feature engineering
# smoker_yes column becomes available after encoding
# ==========================
if 'smoker_yes' in df.columns:
    df['smoker_bmi'] = df['smoker_yes'] * df['bmi']

# split features and target
X = df.drop(columns='charges')
y = df['charges']

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

# ==========================
# 🔥 IMPROVEMENT 3: better model
# ==========================
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ==========================
# evaluation
# ==========================
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print("Train R2:", metrics.r2_score(y_train, train_pred))
print("Test R2:", metrics.r2_score(y_test, test_pred))

print("MAE:", metrics.mean_absolute_error(y_test, test_pred))

# ==========================
# test prediction
# ==========================
input_data = (31, 25.74, 0, 1, 0)  
# NOTE: order depends on columns after get_dummies

input_array = np.asarray(input_data).reshape(1, -1)

prediction = model.predict(input_array)
print("Insurance cost:", prediction[0])

# ==========================
# save model
# ==========================
with open("insurance_model.pkl", "wb") as file:
    pickle.dump(model, file)