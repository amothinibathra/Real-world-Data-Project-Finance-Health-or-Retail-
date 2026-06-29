# ==========================================================
# REAL-WORLD DATA SCIENCE PROJECT
# Retail Sales Analysis and Sales Prediction
# ==========================================================

# ---------------------------
# Import Libraries
# ---------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ---------------------------
# Load Dataset
# ---------------------------

df = pd.read_csv("Superstore.csv", encoding="latin1")

print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nDataset Information")
print(df.info())

# ---------------------------
# Data Cleaning
# ---------------------------

print("\nMissing Values")
print(df.isnull().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)

# Convert Order Date into datetime
df["Order Date"] = pd.to_datetime(df["Order Date"])

print("\nDuplicates Removed Successfully")

# ---------------------------
# Feature Engineering
# ---------------------------

df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.month

# ---------------------------
# Exploratory Data Analysis
# ---------------------------

print("\nStatistical Summary")
print(df.describe())

# ---------------------------
# Total Sales
# ---------------------------

print("\nTotal Sales = ", round(df["Sales"].sum(), 2))

print("Total Profit = ", round(df["Profit"].sum(), 2))

# ==========================================================
# Visualization 1
# Sales by Category
# ==========================================================

category_sales = df.groupby("Category")["Sales"].sum()

plt.figure(figsize=(8,5))
category_sales.plot(kind="bar", color=["skyblue","orange","green"])

plt.title("Sales by Category")
plt.xlabel("Category")
plt.ylabel("Sales")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# ==========================================================
# Visualization 2
# Sales by Region
# ==========================================================

plt.figure(figsize=(8,5))

sns.barplot(
    x="Region",
    y="Sales",
    data=df,
    estimator=np.sum
)

plt.title("Sales by Region")
plt.tight_layout()
plt.show()

# ==========================================================
# Visualization 3
# Monthly Sales Trend
# ==========================================================

monthly_sales = df.groupby("Month")["Sales"].sum()

plt.figure(figsize=(10,5))

plt.plot(
    monthly_sales.index,
    monthly_sales.values,
    marker="o",
    linewidth=2
)

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.grid(True)
plt.tight_layout()
plt.show()

# ==========================================================
# Visualization 4
# Profit Distribution
# ==========================================================

plt.figure(figsize=(8,5))

sns.histplot(df["Profit"], bins=30, kde=True)

plt.title("Profit Distribution")
plt.tight_layout()
plt.show()

# ==========================================================
# Visualization 5
# Correlation Heatmap
# ==========================================================

plt.figure(figsize=(6,5))

corr = df[["Sales","Profit","Discount","Quantity"]].corr()

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# ==========================================================
# Visualization 6
# Boxplot
# ==========================================================

plt.figure(figsize=(8,5))

sns.boxplot(
    x="Category",
    y="Profit",
    data=df
)

plt.title("Profit by Category")
plt.tight_layout()
plt.show()

# ==========================================================
# Prepare Machine Learning Data
# ==========================================================

features = df[[
    "Quantity",
    "Discount",
    "Year",
    "Month"
]]

target = df["Sales"]

# ---------------------------
# Train Test Split
# ---------------------------

X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.20,
    random_state=42
)

# ---------------------------
# Build Model
# ---------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------------------
# Prediction
# ---------------------------

predictions = model.predict(X_test)

# ---------------------------
# Evaluation
# ---------------------------

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(mean_squared_error(y_test, predictions))

r2 = r2_score(y_test, predictions)

print("\n========== MODEL PERFORMANCE ==========")

print("MAE :", round(mae,2))
print("RMSE:", round(rmse,2))
print("R2 Score:", round(r2,4))

# ==========================================================
# Actual vs Predicted
# ==========================================================

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
})

print("\nSample Predictions")
print(results.head(10))

# ==========================================================
# Feature Importance
# ==========================================================

importance = pd.Series(
    model.feature_importances_,
    index=features.columns
)

importance = importance.sort_values()

plt.figure(figsize=(7,4))

importance.plot(
    kind="barh",
    color="purple"
)

plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# ==========================================================
# Scatter Plot
# ==========================================================

plt.figure(figsize=(7,5))

plt.scatter(
    y_test,
    predictions,
    alpha=0.6
)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")

plt.tight_layout()
plt.show()

# ==========================================================
# Save Prediction Results
# ==========================================================

results.to_csv("Sales_Predictions.csv", index=False)

print("\nPrediction file saved as 'Sales_Predictions.csv'")

# ==========================================================
# Business Insights
# ==========================================================

print("\n================ BUSINESS INSIGHTS ================")

print("1. Technology category generally contributes high sales.")
print("2. Discounts have a negative impact on profit.")
print("3. Quantity sold strongly influences sales.")
print("4. Regional sales vary across the business.")
print("5. Monthly trends reveal seasonal demand patterns.")
print("6. The Random Forest model can be used to estimate future sales.")

print("\nProject Completed Successfully!")
