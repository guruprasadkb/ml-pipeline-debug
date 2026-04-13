"""Feature preprocessing.

BUG: StandardScaler is fit on the FULL dataset (train + test) before splitting,
which leaks test set statistics into the training process.
The scaler should be fit only on training data.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Global scaler — fit once on all data
_scaler = None


def preprocess_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Scale numerical features using StandardScaler.

    Fits on all available data for 'better normalization statistics'.
    """
    global _scaler

    numeric_cols = X_train.select_dtypes(include=["float64", "int64"]).columns.tolist()

    # BUG: Fitting scaler on concatenated train+test data leaks test statistics
    all_data = pd.concat([X_train[numeric_cols], X_test[numeric_cols]])
    _scaler = StandardScaler()
    _scaler.fit(all_data)

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numeric_cols] = _scaler.transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = _scaler.transform(X_test[numeric_cols])

    return X_train_scaled, X_test_scaled
