# Loan System AI Prediction

## Executive Summary
This repository contains the pipeline and methodology used to predict the binary `Target` variable for a financial dataset. The pipeline transitions from Exploratory Data Analysis (EDA) in R, through algorithmic preprocessing and hyperparameter tuning in Python, culminating in the generation of predictions using an optimized XGBoost Classifier.

## Project Structure

```text
.
├── Data/
│   ├── Assignment_Data.csv           # Raw original data
│   ├── cleaned_assignment_data.csv   # Intermediate dataset after R-based cleaning
│   ├── Final_Predictions.csv         # Final Deliverable (Target predictions)
│   └── final_model_data/             # Processed, split DataFrames for ML
├── R_Scripts/                        # EDA & Cleaning
│   ├── eda_analysis.Rmd              # Initial visualizations and correlation tests
│   └── pre_process_financial.Rmd     # Data-cleaning logic (coercion, imputation)
├── Python_Scripts/                   # Machine Learning
│   ├── model_creation.ipynb          # Feature Engineering, One-Hot Encoding, Data Splitting
│   └── model_training.ipynb          # Baseline Training, Hyperparameter tuning, Feature Importance
├── Original Docs/                    # Original project documentation
└── Final_Report.md                   # Comprehensive project report
```

## Methodology

### 1. Exploratory Data Analysis (EDA)
- **Target Imbalance:** Identified a 30/70 class imbalance (Class 0: 30%, Class 1: 70%), prompting the use of F1-Score and ROC-AUC over accuracy.
- **Algorithms:** Committed to Tree-based algorithms (like XGBoost) due to weak linear relationships.
- **Multicollinearity & Zero-Variance:** Removed highly redundant variables (>0.85 correlation) and zero-variance columns to prevent model instability.

### 2. Feature Engineering & Preprocessing
- **Missingness Flags:** Created boolean flags to capture the predictive signal of the absence of data before median imputation.
- **Date Engineering:** Engineered numerical differences from raw `datetime` objects (e.g., `days_late`, `days_to_arrive`).
- **Combined Encoding:** Performed One-Hot Encoding across the entire dataset simultaneously prior to the Train/Test split to prevent dimensional mismatch errors during final prediction.

### 3. Model Selection & Hyperparameter Tuning
- **Baseline:** XGBoost outperformed RandomForest, utilizing `scale_pos_weight` to mathematically force the model to penalize errors on the minority class, drastically improving minority recall.
- **Tuning:** Utilized `RandomizedSearchCV` with 5-Fold Cross Validation testing 50 architectural combinations. The model was optimized specifically for **ROC-AUC**, improving the score from 0.6900 to 0.7176.
- **Feature Importance:** The engineered feature `days_to_arrive` emerged as the most important driver of the target.

## Setup & Execution
1. Review the data preprocessing steps and EDA in the `R_Scripts/` directory.
2. Run `Python_Scripts/model_creation.ipynb` to process the data, perform feature engineering, and apply encoding.
3. Run `Python_Scripts/model_training.ipynb` to train the XGBoost model, perform hyperparameter tuning, and generate the final predictions in `Data/Final_Predictions.csv`.

For a detailed breakdown of the analysis, decisions, and findings, please refer to the [`Final_Report.md`](Final_Report.md).
