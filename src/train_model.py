import pandas as pd
import xgboost as xgb
import logging
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt

# Imports from local modules
from data_preprocessing import preprocess_data
from feature_engineering import extract_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_fraud_model(data_path: str, model_save_path: str):
    """
    Trains an XGBoost regression model for fraud risk scoring.
    """
    logger.info(f"Loading data from {data_path}...")
    df_raw = pd.read_csv(data_path)
    
    # Preprocess and Feature Engineer
    X_raw, y = preprocess_data(df_raw)
    X = extract_features(X_raw)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    logger.info("Initializing XGBoost Regressor...")
    # Using XGBRegressor for continuous risk scoring
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='binary:logistic',  # Regression towards the probability
        random_state=42,
        n_jobs=-1
    )
    
    logger.info("Training model...")
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred_proba = model.predict(X_test)
    ap_score = average_precision_score(y_test, y_pred_proba)
    logger.info(f"Average Precision Score: {ap_score:.4f}")
    
    # Save model
    with open(model_save_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    train_fraud_model('data/transactions.csv', 'models/fraud_model.pkl')
