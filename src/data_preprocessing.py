import pandas as pd
import logging
from typing import Tuple
from sklearn.model_selection import train_test_df
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Cleans and prepares data for feature engineering.
    
    Args:
        df (pd.DataFrame): Raw transaction data.
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features (X) and Target (y).
    """
    logger.info("Starting data preprocessing...")
    
    try:
        # Check for missing values
        if df.isnull().values.any():
            logger.warning("Missing values detected. Dropping rows with NaNs.")
            df = df.dropna()
        
        # Convert timestamp to datetime if not already
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp for time-based features
        df = df.sort_values('timestamp')
        
        # Define target and raw features
        y = df['is_fraud']
        X = df.drop(columns=['is_fraud', 'transaction_id'])
        
        return X, y
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    try:
        data = pd.read_csv('data/transactions.csv')
        X, y = preprocess_data(data)
        logger.info(f"Preprocessed {len(X)} records.")
    except FileNotFoundError:
        logger.error("Transactions file not found. Run data_generator.py first.")
