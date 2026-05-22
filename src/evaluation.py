
from sklearn.metrics import mean_absolute_error

def evaluate_model_mae(model, X_test, y_test):
    """
    Uses a trained model to compute MAE on test data.

    Returns:
        mae (float)
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    return mae