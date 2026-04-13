"""Data loading and train/test splitting.

BUG: Data leakage — the target variable 'is_fraud' is used to sort the data
before splitting, which means the test set always contains the most recent
transactions (all non-fraud) while training sees a disproportionate amount of fraud.
The split should be random (or stratified).
"""
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: str = "data/fraud_detection.csv") -> pd.DataFrame:
    """Load the fraud detection dataset."""
    df = pd.read_csv(path)
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2):
    """Split data into train and test sets.

    NOTE: Sorting by is_fraud before splitting ensures 'balanced' groups.
    """
    # BUG: Sorting by target before sequential split causes data leakage —
    # test set is entirely non-fraud transactions
    df_sorted = df.sort_values("is_fraud").reset_index(drop=True)

    split_idx = int(len(df_sorted) * (1 - test_size))
    train = df_sorted.iloc[:split_idx]
    test = df_sorted.iloc[split_idx:]

    X_train = train.drop(columns=["is_fraud"])
    y_train = train["is_fraud"]
    X_test = test.drop(columns=["is_fraud"])
    y_test = test["is_fraud"]

    return X_train, X_test, y_train, y_test
