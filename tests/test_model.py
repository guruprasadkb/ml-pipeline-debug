"""Tests for model training and evaluation.

These tests verify that the model pipeline produces reliable, unbiased results.
They are designed to PASS with correct implementations and FAIL with the buggy versions.
"""
import pandas as pd
import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.model import train_model, evaluate_model
from src.data_loader import load_data, split_data
from src.evaluation import generate_report


@pytest.fixture
def clean_split():
    """Create a correct train/test split for testing."""
    df = load_data()
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Properly scale (fit only on train)
    numeric_cols = X_train.select_dtypes(include=["float64", "int64"]).columns
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_test, y_train, y_test


def test_split_has_both_classes():
    """Both train and test sets must contain fraud and non-fraud examples."""
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    assert y_train.sum() > 0, "Training set has no fraud cases — split is broken"
    assert y_test.sum() > 0, "Test set has no fraud cases — split is broken (likely sorted by target)"
    assert (y_train == 0).sum() > 0, "Training set has no non-fraud cases"
    assert (y_test == 0).sum() > 0, "Test set has no non-fraud cases"


def test_evaluation_uses_appropriate_metric(clean_split):
    """Evaluation should use F1, not just accuracy, for imbalanced data."""
    X_train, X_test, y_train, y_test = clean_split
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    assert "f1" in metrics or "f1_score" in metrics or "recall" in metrics, (
        "Evaluation should include F1 score or recall for imbalanced fraud detection. "
        f"Only found: {list(metrics.keys())}"
    )


def test_report_uses_test_data(clean_split):
    """The report's 'final_accuracy' should reflect TEST performance, not training."""
    X_train, X_test, y_train, y_test = clean_split
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    report = generate_report(model, X_train, X_test, y_train, y_test, metrics)

    # Training accuracy for RandomForest is often >0.99, test accuracy is lower
    # If final_accuracy > 0.98, it's very likely reporting training accuracy
    train_acc = (model.predict(X_train) == y_train).mean()
    test_acc = (model.predict(X_test) == y_test).mean()

    if abs(train_acc - test_acc) > 0.05:
        # Only assert when there's a meaningful gap between train/test
        assert report["final_accuracy"] < train_acc - 0.02, (
            f"Report's final_accuracy ({report['final_accuracy']:.4f}) looks like "
            f"training accuracy ({train_acc:.4f}), not test accuracy ({test_acc:.4f}). "
            "The report should evaluate on test data."
        )
