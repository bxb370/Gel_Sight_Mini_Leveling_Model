
import numpy as np
from sklearn.model_selection import train_test_split

def prepare_train_test(images, test_size=0.2, random_state=42):
    """
    Takes a list of images in the format:
    (label, filename, img_array, feature)

    Returns:
        X_train, X_test, y_train, y_test
    """

    # Extract features (X) and labels (y)
    X = np.array([[item[3]] for item in images])  # e.g., entropy
    y = np.array([item[0] for item in images])    # labels

    # Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test



from sklearn.linear_model import LinearRegression

def train_linear_model(X_train, y_train):
    """
    Trains a linear regression model.

    Returns:
        trained model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model
