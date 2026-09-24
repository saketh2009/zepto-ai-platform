# zepto data & AI platform capstone

This repository contains an end-to-end data and machine learning platform covering web scraping, relational data modeling, exploratory analysis, predictive modeling pipelines, and model evaluation.

## Repository Structure
- /data_pipeline: Scraping, cleaning, currency normalization, and SQLite database storage.
- `/analytics`: Exploratory data analysis, statistical tests, predictive modeling pipelines, and model evaluation.
- `requirements.txt`: Unified dependencies for all modules.

---

## Module 1: Data Pipeline Decisions
- **Data Source:** Scraped from `books.toscrape.com` (100 records across 5 catalog pages).
- **Currency Conversion:** Computed using the project constant `1 GBP = 105.50 INR`.
- **Relational Schema:** Implemented a two-table schema with primary/foreign keys (`categories` and `books`).
- **Equivalence Verification:** The output of the multi-table SQL JOIN matches `pd.merge` output identically.

---

## Module 2: Analytics & Modeling Decisions
- **Missing Value Handling:**
  - `embarked` / `embark_town` (< 5% missing): Imputed with the most frequent value.
  - `age` (~19.8% missing): Imputed using median to minimize outlier sensitivity.
  - `deck` (~77.2% missing): Excluded from modeling because missingness exceeds the 30% reliability threshold.
- **Skewness:** `fare` is strongly right-skewed (`mean > median > mode`).
- **Stratified Split:** Used stratified sampling on `survived` to preserve target class balance in train/test sets.
- **Leakage Prevention:** All transformations (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`) are fit strictly on training splits inside scikit-learn `Pipeline` and `ColumnTransformer` objects.
- **Imbalance Comparison:** Evaluated baseline, balanced class weights, and SMOTE oversampling.
- **Hyperparameter Tuning:** Tuned `RandomForestClassifier` using `GridSearchCV` with OOB score enabled.
- **Regression Side-Task:** Modeled `fare` using linear regression; residuals indicate heteroscedasticity due to high fare variance.
-
