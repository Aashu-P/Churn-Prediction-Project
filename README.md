# Customer Churn Prediction System

An end-to-end machine learning project that predicts customer churn using real world telecom data.  
The project covers the full ML lifecycle: data preprocessing, model selection, evaluation, threshold tuning, and deployment.

🚀 **Live Demo (Streamlit App):**  
👉 https://telcochurnxgb.streamlit.app/

---

## Project Overview

Customer churn is a high impact business problem where recall matters more than raw accuracy.  
This project focuses on **identifying high risk churn customers**, even when user data is incomplete.

Key goals:
- Handle **missing and unknown user inputs**
- Optimize for **recall** on the churn class
- Compare multiple models and select the best one
- Deploy a **production ready inference pipeline**

---
## Notes on Project Iteration

The EDA and Feature Engineering notebooks reflect early exploratory work where features were manually engineered to understand the data and baseline modeling behavior.

After deeper research into scikit-learn documentation, the final implementation was refactored to use a production grade preprocessing pipeline (ColumnTransformer + Pipelines) inside the modeling notebook.

As a result:
- Manual feature engineering was intentionally replaced
- All preprocessing is now learned from training data only
- The deployed model operates entirely on raw user inputs

This mirrors how real world ML systems evolve from exploration to production.

---

## Models Evaluated

- **Logistic Regression (CV)**  
  - Baseline, interpretable model  
  - Cross-validated with recall optimization

- **Random Forest (CV)**  
  - Non-linear model with class imbalance handling  
  - Tuned via GridSearchCV

- **XGBoost (Final Model)** ✅  
  - Gradient boosting with early stopping  
  - Optimized using PR-AUC and recall  
  - Best balance of recall, precision, and generalization

**Final choice:** XGBoost  
Chosen due to stronger recall on churn customers and better handling of complex feature interactions compared to Random Forest.

---

## Key Techniques Used

- **ColumnTransformer Pipelines**
  - Median imputation for numeric features
  - Most frequent imputation + One Hot Encoding for categorical features
  - Robust to missing and unseen categories at inference time

- **Threshold Tuning**
  - Custom decision threshold optimized for business oriented recall
  - Separate from model training for flexibility in production

- **Model Serialization**
  - 'joblib' used to persist:
    - Preprocessor
    - Trained XGBoost model
    - Tuned decision threshold

---

## Live Inference Demo

The Streamlit app allows users to:
- Input customer data in **raw, human readable form**
- Leave fields blank (missing values supported)
- Receive:
  - Churn probability
  - Final churn prediction based on tuned threshold

This mirrors how a real production churn system would be used by non-technical stakeholders.

---

## Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- Streamlit  
- Joblib  

---

## Author

Built by **Aashu P**  
Data Science | Machine Learning | End-to-End ML Systems
