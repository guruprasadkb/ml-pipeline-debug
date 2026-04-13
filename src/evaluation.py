"""Metrics reporting.

BUG: Reports TRAINING accuracy instead of test accuracy.
The evaluation function is called with training data, making the model
appear to perform much better than it actually does on unseen data.
"""


def generate_report(model, X_train, X_test, y_train, y_test, metrics):
    """Generate a summary report of model performance.

    BUG: Re-evaluates on training data and reports that as 'Final Accuracy'.
    """
    # BUG: Evaluating on training data, not test data
    train_pred = model.predict(X_train)
    train_accuracy = (train_pred == y_train).mean()

    report = {
        "model_type": type(model).__name__,
        "n_features": X_train.shape[1],
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        # BUG: This is training accuracy, misleadingly labeled as "Final Accuracy"
        "final_accuracy": train_accuracy,
        "detailed_metrics": metrics,
    }

    print(f"\n{'='*50}")
    print(f"  Model Performance Report")
    print(f"{'='*50}")
    print(f"  Model:          {report['model_type']}")
    print(f"  Features:       {report['n_features']}")
    print(f"  Train samples:  {report['train_samples']}")
    print(f"  Test samples:   {report['test_samples']}")
    print(f"  Final Accuracy: {report['final_accuracy']:.4f}")
    print(f"{'='*50}\n")

    return report
