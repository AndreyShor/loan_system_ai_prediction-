# Financial Analysis: Final Project Report

## Executive Summary
This report details the methodology and decisions made to fulfill the assignment objective: predicting the binary `Target` variable for rows 2058 to 2572. The pipeline transitions from Exploratory Data Analysis (EDA) in R, through algorithmic preprocessing and hyperparameter tuning in Python, culminating in the generation of `Final_Predictions.csv` using an optimized XGBoost Classifier.

---

## Phase 1: Exploratory Data Analysis (EDA)
**Objective:** Understand the data structure, identify data-quality hazards, and formulate a modeling strategy.

**Key Findings & Decisions:**
1. **Target Imbalance:** The dataset exhibited a 30/70 class imbalance (Class 0: 30%, Class 1: 70%). **Decision:** Accuracy was discarded as an evaluation metric in favor of F1-Score and ROC-AUC.
2. **Weak Linear Relationships:** Correlation analysis revealed that no single variable possessed a linear correlation with the Target higher than `0.152`. **Decision:** Linear models (e.g., Logistic Regression) were abandoned. We committed strictly to **Tree-based algorithms** capable of mapping non-linear interactions.
3. **Multicollinearity:** We identified multiple redundant variables boasting >0.85 correlations with each other. **Decision:** Features like `Variable_11`, `Variable_26`, and `Variable_39` were marked for deletion to prevent model instability.
4. **Zero-Variance:** Several columns (`Variable_28`, `Variable_3`, etc.) contained exactly 1 unique value across the entire dataset. **Decision:** These were dropped as they hold zero mathematical predictive power.

---

## Phase 2: Feature Engineering & Preprocessing
**Objective:** Clean the dataset and format it exclusively into numerical matrices for Machine Learning ingestion.

**Key Preprocessing Steps:**
1. **Missingness Flags (MNAR):** Missing values in critical numerical columns were not random. We engineered boolean flags (e.g., `is_missing_Variable_21`) to capture the predictive signal of the *absence* of data, before using Median Imputation to fill the gaps.
2. **Dropping Identifiers:** `ID`, `country_id`, `application_id`, `product_id`, and `customer_id` were aggressively dropped from the training set. **Justification:** If left in, tree algorithms would simply memorize the Customer IDs rather than learning underlying financial patterns, causing massive overfitting.
3. **Date Engineering:** Machine Learning models cannot ingest raw `datetime` objects. We engineered numerical differences: `days_late = paid_date - due_date` and `days_to_arrive = arrived_date - paid_date`.
4. **The "Combined Encoding" Strategy:** We delayed the Train/Test data split until the very end. We One-Hot Encoded the categorical variables (`Variable_5`, `Variable_12`, etc.) across the *entire* dataset simultaneously. **Justification:** This guarantees that the final unlabelled Prediction Set has the exact same columns as the Training Set, preventing fatal dimensional mismatch errors during final prediction.

---

## Phase 3: Model Selection & Hyperparameter Tuning
**Objective:** Establish baselines, combat class imbalance, and squeeze out maximum predictive power.

**Baseline Evaluation:**
We tested two algorithms: `RandomForestClassifier` vs `XGBClassifier`. 
* To combat the 30/70 imbalance, we injected a `scale_pos_weight` directly into XGBoost mathematically forcing it to penalize errors on the Minority class.
* **Result:** XGBoost proved vastly superior. It successfully traded a tiny bit of majority accuracy to double its Minority (Class 0) Recall from 24% to **54%**.

**Hyperparameter Tuning:**
We utilized `RandomizedSearchCV` with 5-Fold Cross Validation testing 50 different architectural combinations. 
* **Justification:** Randomized Search tests dozens of combinations (like Tree Depth and Learning Rate) in a fraction of the time of an exhaustive Grid Search, while Cross Validation ensures the model isn't just getting "lucky" on a specific data split.
* The model was specifically refit and tuned to mathematically optimize for **ROC-AUC** instead of F1. This pushed the final score up from a baseline 0.6900 to **0.7176**.

**Feature Importance & Drivers of the Target:**
Post-tuning, we utilized XGBoost's mathematical weight extraction to verify exactly which variables the model uses to make its decisions. The top 5 driving features for our final ROC-AUC optimized model are:
1. `days_to_arrive` (Engineered Feature: `arrived_date` - `paid_date`)
2. `Variable_7`
3. `Variable_20`
4. `Variable_15`
5. `Variable_9`

**Conclusion:** Our engineered feature `days_to_arrive` emerged as the absolute #1 most important variable in the entire dataset. This mathematically proves that the time difference between payment and arrival is the strongest indicator for determining the Target. If we had not engineered those dates into numerical differences, the model would have lost its most powerful predictor.

---

## Phase 4: Final Inference
The unlabelled rows (2058 to 2572) were fed through the finalized XGBoost model. The resulting binary predictions were restitched to their original `ID` columns and exported.

## Future Work: Potential Next Paths
While the current XGBoost model utilizes algorithmic class weighting to combat the 30/70 data imbalance, future iterations of this pipeline could explore **SMOTE (Synthetic Minority Over-sampling Technique)**.
* **Why SMOTE?** SMOTE mathematically generates "fake" synthetic data points for the minority class (Class 0) based on the proximity of real data points. By artificially inflating the minority class to achieve a perfect 50/50 balance *before* training, algorithms like XGBoost can sometimes learn much sharper, more robust decision boundaries, which could further improve the ROC-AUC score.

---

## Project Directory & File Explanations

### 1. `Data/` Folder
* `Assignment_Data.csv`: The raw original data provided in the prompt.
* `cleaned_assignment_data.csv`: The intermediate dataset exported after R-based cleaning (handling zero-variance and missingness).
* `Final_Predictions.csv`: **The Final Deliverable.** Contains the 515 requested Target predictions.
* `final_model_data/`: Contains the fully processed, split DataFrames ready for ML ingestion (`X_train.csv`, `X_test.csv`, etc.).

### 2. `R_Scripts/` Folder (EDA & Cleaning)
* `eda_analysis.Rmd` / `.html`: The R-Markdown notebook containing the initial visualizations, multicollinearity checks, and statistical correlation tests.
* `pre_process_financial.Rmd` / `.html`: The R-script that executes the heavy data-cleaning logic (coercion, dropping empty columns, and median imputation).

### 3. `Python_Scripts/` Folder (Machine Learning)
* `model_creation.ipynb`: The Python notebook responsible for Feature Engineering (Dates), One-Hot Encoding, Data Splitting, and safely dropping identifiers.
* `model_training.ipynb`: The Python notebook responsible for Baseline Training, Class Weight calculation, `RandomizedSearchCV` hyperparameter tuning, Feature Importance plotting, and final CSV generation.

### 4. Root Documentation
* `Final_Report.md`: This document.
