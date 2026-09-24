import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------
# Part A: Profiling & Missing Data
# ---------------------------------------------------------
print("--- TASK 1: DATA PROFILING ---")
csv_path = "titanic.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    df = sns.load_dataset("titanic")
    df.to_csv(csv_path, index=False)

print(f"Dataset Shape: {df.shape}")
missing_pct = (df.isnull().sum() / len(df)) * 100
affected = missing_pct[missing_pct > 0]
print("\nMissing Percentages:\n", affected)

# Task 3: IQR Outliers and Skewness
def report_iqr(series, name):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    outliers = series[(series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))]
    print(f"{name} Outlier Count: {len(outliers)}")

report_iqr(df["age"].dropna(), "Age")
report_iqr(df["fare"].dropna(), "Fare")

fare_mean = df["fare"].mean()
fare_median = df["fare"].median()
fare_mode = df["fare"].mode()[0]
print(f"Fare Mean: {fare_mean:.2f}, Median: {fare_median:.2f}, Mode: {fare_mode:.2f}")

# Task 4: Bivariate Survival Rates
print("\nSurvival by Sex:\n", df[df["survived"] == 1]["sex"].value_counts() / df["sex"].value_counts())
print("\nSurvival by Pclass:\n", df[df["survived"] == 1]["pclass"].value_counts() / df["pclass"].value_counts())

corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr_matrix = df[corr_cols].corr()
print("\nCorrelation Matrix:\n", corr_matrix)

# ---------------------------------------------------------
# Part B: Modeling Pipeline
# ---------------------------------------------------------
print("\n--- MODELING PIPELINE ---")
features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
target = "survived"

data = df[features + [target]].dropna(subset=[target])
X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

num_cols = ["pclass", "age", "sibsp", "parch", "fare"]
cat_cols = ["sex", "embarked"]

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

# Classifier training & evaluation
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, oob_score=True)
}

results = []
for name, clf in models.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1": f1_score(y_test, preds),
        "AUC": roc_auc_score(y_test, probs)
    })

print("\nModel Comparison Table:")
print(pd.DataFrame(results).to_string(index=False))

# Hyperparameter Tuning
rf_pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(random_state=42, oob_score=True))
])

param_grid = {
    "clf__n_estimators": [50, 100],
    "clf__max_depth": [5, 10, None],
    "clf__max_features": ["sqrt", "log2"]
}

grid = GridSearchCV(rf_pipeline, param_grid, cv=3, scoring="f1")
grid.fit(X_train, y_train)
best_rf = grid.best_estimator_
print(f"\nBest RF Params: {grid.best_params_}")
print(f"Best RF OOB Score: {best_rf.named_steps['clf'].oob_score_:.4f}")

# Regression Side-Task: Predict Fare
reg_features = ["pclass", "sex", "age", "sibsp", "parch", "embarked"]
y_reg = df["fare"].dropna()
X_reg = df.loc[y_reg.index, reg_features]

X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg_num = ["pclass", "age", "sibsp", "parch"]
reg_cat = ["sex", "embarked"]

reg_prep = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), reg_num),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore"))]), reg_cat)
])

reg_pipe = Pipeline([("prep", reg_prep), ("model", LinearRegression())])
reg_pipe.fit(X_r_train, y_r_train)
reg_preds = reg_pipe.predict(X_r_test)

mae = mean_absolute_error(y_r_test, reg_preds)
rmse = np.sqrt(mean_squared_error(y_r_test, reg_preds))
r2 = r2_score(y_r_test, reg_preds)
n, p = len(y_r_test), X_r_train.shape[1]
adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

print(f"\nRegression (Fare) -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}, Adj R2: {adj_r2:.4f}")

# Save full pipeline
joblib.dump(best_rf, "best_pipeline.joblib")
print("Saved pipeline to best_pipeline.joblib")
