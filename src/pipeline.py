"""Fraud Detection ML Pipeline — Main Orchestration.

This pipeline loads transaction data, preprocesses features, trains a model,
and reports performance metrics.

Known issue: The model achieves 99% accuracy in development but drops to ~50%
in production. The team suspects data or evaluation issues.
"""
from src.data_loader import load_data, split_data
from src.preprocessing import preprocess_features
from src.model import train_model, evaluate_model
from src.evaluation import generate_report


def run_pipeline(data_path: str = "data/fraud_detection.csv"):
    """Execute the full ML pipeline."""
    print("Loading data...")
    df = load_data(data_path)
    print(f"  Dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"  Fraud rate: {df['is_fraud'].mean():.2%}")

    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"  Train: {len(X_train)} samples ({y_train.mean():.2%} fraud)")
    print(f"  Test:  {len(X_test)} samples ({y_test.mean():.2%} fraud)")

    print("\nPreprocessing features...")
    X_train_scaled, X_test_scaled = preprocess_features(X_train, X_test)

    print("\nTraining model...")
    model = train_model(X_train_scaled, y_train)

    print("\nEvaluating model...")
    metrics = evaluate_model(model, X_test_scaled, y_test)

    print("\nGenerating report...")
    report = generate_report(model, X_train_scaled, X_test_scaled, y_train, y_test, metrics)

    return report


if __name__ == "__main__":
    run_pipeline()
