# PRD: Retail Fraud Detection Risk Scoring Model

## 1. Overview
This project implements a regression-based risk scoring model to identify fraudulent retail transactions. Using XGBoost, we aim to provide a continuous risk score (0-1) for each transaction.

## 2. Objective
- Maximize detection of fraudulent transactions (Recall).
- Maintain a manageable False Positive Rate (FPR) to avoid friction for legitimate customers.
- Provide a scalable and interpretable risk score.

## 3. Best Practices Implemented
- **Modular Codebase:** Separate scripts for preprocessing, feature engineering, and training.
- **Robust Feature Engineering:** Incorporating RFM (Recency, Frequency, Monetary) metrics and geolocation consistency.
- **Explainability:** Utilizing SHAP values for model interpretability.
- **Evaluation:** Using Precision-Recall curves and Cost-Benefit Analysis rather than just accuracy.
- **Data Validation:** Schema validation for incoming transaction data.

## 4. Technical Stack
- **Language:** Python 3.x
- **Core Library:** XGBoost
- **Data Handling:** Pandas, Scikit-learn
- **Monitoring:** Logging and Performance Tracking

## 5. Non-Standard Files (For Review/Testing)
The `bad_standards/` directory contains examples of what **not** to do, used for testing automated review agents.
