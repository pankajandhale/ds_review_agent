import pandas as pd
import numpy as np
import logging
from typing import List

logger = logging.getLogger(__name__)

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs feature engineering for retail fraud detection.
    
    Includes:
    - Time-based features (Hour, Day of week)
    - Aggregated user features (Mean amount per user)
    - Categorical encoding
    """
    logger.info("Extracting features...")
    
    df = df.copy()
    
    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Financial metrics
    df['log_amount'] = np.log1p(df['amount'])
    
    # User Behavior (RFM-like)
    df['user_avg_amount'] = df.groupby('user_id')['amount'].transform('mean')
    df['amt_vs_avg'] = df['amount'] / (df['user_avg_amount'] + 1e-9)
    
    # Transaction Frequency
    df['user_tx_count'] = df.groupby('user_id')['timestamp'].transform('count')
    
    # Categorical Encoding
    df = pd.get_dummies(df, columns=['merchant_category'], drop_first=True)
    
    # Drop identifier and timestamp after extraction
    cols_to_drop = ['timestamp', 'user_id']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    return df

if __name__ == "__main__":
    # Placeholder for testing
    print("Feature engineering script loaded.")
