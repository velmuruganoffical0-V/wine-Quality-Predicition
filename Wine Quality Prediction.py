import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("WineQT.csv") 

print("\n Data Loaded")
print(df.head())

if 'Id' in df.columns:
    df.drop('Id', axis=1, inplace=True)

print("\n Checking for nulls:")
print(df.isnull().sum())

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title(" Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

X = df.drop("quality", axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)

sgd = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd.fit(X_train_scaled, y_train)
sgd_pred = sgd.predict(X_test_scaled)

svc = SVC()
svc.fit(X_train_scaled, y_train)
svc_pred = svc.predict(X_test_scaled)

print("\n Random Forest Classifier:")
print(classification_report(y_test, rf_pred))

print("\n Stochastic Gradient Descent Classifier:")
print(classification_report(y_test, sgd_pred))

print("\n Support Vector Classifier:")
print(classification_report(y_test, svc_pred))

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, cmap='Blues', fmt='d')
plt.title(" Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()



