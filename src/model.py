"""Model training and evaluation.

BUG: Uses accuracy as the evaluation metric for a highly imbalanced dataset
(~5% fraud). Accuracy is misleading here — a model predicting "not fraud" for
everything gets 95% accuracy. Should use F1, precision, recall, or AUC-ROC.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight=None,  # BUG: No class_weight for imbalanced data
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model using accuracy.

    BUG: Accuracy is misleading for imbalanced fraud detection (5% positive class).
    A trivial classifier achieves ~95% accuracy by always predicting 'not fraud'.
    Should use F1-score, precision/recall, or AUC-ROC instead.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return {"accuracy": accuracy}
