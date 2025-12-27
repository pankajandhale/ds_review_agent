import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_mock_data(n_samples=1000):
    """
    Generates synthetic retail transaction data for fraud detection.
    
    Returns:
        pd.DataFrame: A dataframe containing transaction records.
    """
    np.random.seed(42)
    
    data = {
        'transaction_id': range(1, n_samples + 1),
        'timestamp': [datetime.now() - timedelta(minutes=np.random.randint(0, 10000)) for _ in range(n_samples)],
        'user_id': np.random.randint(100, 200, n_samples),
        'amount': np.random.uniform(10.0, 5000.0, n_samples),
        'merchant_category': np.random.choice(['electronics', 'grocery', 'fashion', 'travel', 'luxury'], n_samples),
        'is_fraud': np.zeros(n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Inject some fraud patterns (approx 5%)
    fraud_indices = np.random.choice(df.index, size=int(n_samples * 0.05), replace=False)
    df.loc[fraud_indices, 'is_fraud'] = 1
    
    # Fraudsters tend to spend more and in certain categories
    df.loc[fraud_indices, 'amount'] *= np.random.uniform(2, 5, len(fraud_indices))
    df.loc[fraud_indices, 'merchant_category'] = 'electronics'
    
    return df

if __name__ == "__main__":
    df = generate_mock_data(2000)
    df.to_csv('data/transactions.csv', index=False)
    print(f"Generated {len(df)} transactions in data/transactions.csv")
