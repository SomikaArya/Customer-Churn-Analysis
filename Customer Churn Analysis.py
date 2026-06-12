# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:39:16 2026

@author: Somika Arya
"""


# ============================
# Customer Churn Analysis
# ============================

# Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Load the Dataset
# ----------------------------
dataset = pd.read_csv(
    r"F:\SOMIKA ARYA\PROJECTS\Customer Churn Analysis\Telco-Customer-Churn.csv"
)

# Display First 5 Rows
print("First 5 Rows of Dataset:\n")
print(dataset.head())

# ----------------------------
# Check Missing Values
# ----------------------------
print("\nMissing Values:\n")
print(dataset.isnull().sum())

# ----------------------------
# Summary Statistics
# ----------------------------
print("\nSummary Statistics:\n")
print(dataset.describe())

# ----------------------------
# Churn Distribution
# ----------------------------
print("\nChurn Count:\n")
print(dataset["Churn"].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(
    x="Churn",
    hue="Churn",
    data=dataset,
    palette="coolwarm",
    legend=False
)
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.show()

# ============================
# Data Preprocessing
# ============================

# Convert TotalCharges to Numeric
dataset["TotalCharges"] = pd.to_numeric(
    dataset["TotalCharges"],
    errors="coerce"
)

# Fill Missing Values with Median
dataset["TotalCharges"] = dataset["TotalCharges"].fillna(
    dataset["TotalCharges"].median()
)

# ----------------------------
# Label Encoding
# ----------------------------
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

categorical_columns = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "Churn"
]

for column in categorical_columns:
    dataset[column] = labelencoder.fit_transform(dataset[column])

# ============================
# Feature Selection
# ============================

X = dataset.drop(["customerID", "Churn"], axis=1)
y = dataset["Churn"]

# ----------------------------
# Train-Test Split
# ----------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ============================
# Feature Scaling
# ============================

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================
# Model Training
# ============================

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

# ============================
# Prediction
# ============================

y_pred = model.predict(X_test)

# ============================
# Model Evaluation
# ============================

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy:.2f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ----------------------------
# Confusion Matrix
# ----------------------------
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Churn", "Churn"]
)

disp.plot(cmap="coolwarm")
plt.title("Confusion Matrix")
plt.show()

# ============================
# Feature Importance
# ============================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:\n")
print(feature_importance)

# ----------------------------
# Plot Feature Importance
# ----------------------------
plt.figure(figsize=(10, 6))

sns.barplot(
    data=feature_importance,
    x="Importance",
    y="Feature"
)

plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()






