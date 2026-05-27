"""This file contains functions to evaluate the model's performance."""

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

def evaluate_linear_model(model, X_test, y_test):
    """
    Evaluate linear model (no transform)
    """
    from sklearn.metrics import mean_absolute_error

    y_pred = model.predict(X_test)
    return mean_absolute_error(y_test, y_pred)

def evaluate_polynomial_model(model, poly, X_test, y_test):
    """
    Evaluate polynomial model (applies transform)
    """
    from sklearn.metrics import mean_absolute_error

    X_test_poly = poly.transform(X_test)
    y_pred = model.predict(X_test_poly)

    return mean_absolute_error(y_test, y_pred)