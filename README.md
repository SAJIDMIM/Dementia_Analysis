# Dementia Risk Prediction Model

This project develops a machine learning model to predict the risk of dementia using **non-medical factors** such as demographic, lifestyle, social, and functional data. The goal is to create an accessible, ethical screening tool that can identify at-risk individuals without relying on clinical diagnostics.

---

## 📖 Project Overview

- **Objective:** To build a predictive model for dementia risk using only non-medical information.
- **Data Source:** National Alzheimer's Coordinating Center (NACC) dataset.
- **Target Variable:** `DEMENTIA_RISK` (binary classification).
- **Key Constraint:** Uses only non-medical features to maintain ethical screening and broad applicability.

---

## 🛠️ Methodology

### 1. Data Exploration & Preprocessing
- **Handled Missing Values:** Median for numerical, mode for categorical.
- **Encoding:** Label Encoding for categorical variables.
- **Feature Scaling:** StandardScaler applied.
- **Class Imbalance:** Addressed using SMOTE oversampling.
- **Train-Test Split:** 80-20 split with stratified sampling.

### 2. Feature Engineering
- **Initial Features:** Demographic (e.g., `SEX`, `EDUC`, `NACCAGE`), Lifestyle (e.g., `TOBAC30`, `ALCFREQ`), Social (e.g., `RESIDENC`, `INDEPEND`), and Functional (e.g., `BILLS`, `SHOPPING`).
- **Feature Selection:** SelectKBest with ANOVA F-test to identify the most predictive features.
- **Final Feature Set:** `NACCAGE`, `EDUC`, `SEX`, `MARISTAT`, `TOBAC30`, `ALCFREQ`, `RESIDENC`, `NACCLIVS`, `INDEPEND`, `BILLS`, `SHOPPING`, `HEIGHT`, `WEIGHT`.

### 3. Model Building & Tuning
The following models were implemented and evaluated:
- Logistic Regression (Baseline)
- Random Forest
- Gradient Boosting
- **XGBoost (Final Model)**

Hyperparameter tuning was performed using `GridSearchCV` on the best-performing model (XGBoost).

---

## 📈 Results & Evaluation

### Final Model Performance (XGBoost)
| Metric | Score |
|--------|-------|
| Accuracy | 0.9268 |
| Precision | 0.8715 |
| Recall | 0.8819 |
| F1-Score | 0.8767 |
| AUC-ROC | 0.9727 |
| CV Mean AUC | 0.9834 |

### Top 5 Non-Medical Risk Factors
1. **INDEPEND** (Independence Level)
2. **SHOPPING** (Shopping Ability)
3. **BILLS** (Bill Management)
4. **TRAVEL** (Travel Capability)
5. **TAXES** (Tax Management)

---

## 🚀 Model Deployment

The final model is saved as `dementia_risk_prediction_model.pkl` and is ready for deployment.

### Risk Stratification
- **Low Risk (<30%):** Routine monitoring
- **Medium Risk (30-60%):** Enhanced screening
- **High Risk (≥60%):** Comprehensive evaluation recommended

### Model Usage
```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('dementia_risk_prediction_model.pkl')

# Prepare new data (ensure same feature engineering as training)
# new_data should contain the final feature set

# Make predictions
predictions = model.predict(new_data)
risk_probabilities = model.predict_proba(new_data)

# The model predicts FUTURE dementia risk, not current diagnosis
# High-risk individuals should receive medical evaluation
