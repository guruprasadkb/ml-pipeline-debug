"""Tests for preprocessing module.

These tests verify that the preprocessing pipeline handles data correctly.
The tests are designed to PASS with correct implementations and FAIL
with the buggy versions.
"""
import pandas as pd
import numpy as np
import pytest
from src.preprocessing import preprocess_features


@pytest.fixture
def sample_data():
    """Create sample train/test data."""
    np.random.seed(42)
    X_train = pd.DataFrame({
        "amount": np.random.uniform(10, 1000, 80),
        "distance": np.random.uniform(0, 100, 80),
        "category": np.random.choice(["a", "b", "c"], 80),
    })
    X_test = pd.DataFrame({
        "amount": np.random.uniform(10, 1000, 20),
        "distance": np.random.uniform(0, 100, 20),
        "category": np.random.choice(["a", "b", "c"], 20),
    })
    return X_train, X_test


def test_scaler_fit_only_on_train(sample_data):
    """Scaler must be fit only on training data, not on train+test combined.

    Fitting on all data leaks test set statistics into preprocessing.
    """
    X_train, X_test = sample_data
    X_train_scaled, X_test_scaled = preprocess_features(X_train, X_test)

    # After correct scaling, train mean should be ~0 and std ~1
    # If scaler was fit on train+test, these values will be slightly off
    train_mean = X_train_scaled["amount"].mean()
    train_std = X_train_scaled["amount"].std()

    assert abs(train_mean) < 0.15, (
        f"Training data mean after scaling should be ~0, got {train_mean:.4f}. "
        "Scaler may have been fit on more than just training data."
    )
    assert abs(train_std - 1.0) < 0.15, (
        f"Training data std after scaling should be ~1, got {train_std:.4f}. "
        "Scaler may have been fit on more than just training data."
    )


def test_no_data_leakage_in_scaling(sample_data):
    """Test set statistics should NOT influence training set scaling."""
    X_train, X_test = sample_data

    # Scale with just train and test
    X_train_scaled_1, _ = preprocess_features(X_train, X_test)

    # Scale with a very different test set (should NOT affect train scaling)
    X_test_extreme = X_test.copy()
    X_test_extreme["amount"] = 99999.0  # Extreme values
    X_train_scaled_2, _ = preprocess_features(X_train, X_test_extreme)

    # If scaler is fit only on train, these should be identical
    pd.testing.assert_frame_equal(
        X_train_scaled_1, X_train_scaled_2,
        check_exact=False, atol=0.01,
        obj="Training data scaling should not depend on test data"
    )
