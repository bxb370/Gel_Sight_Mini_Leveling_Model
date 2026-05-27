"""This file contains functions to train models."""

import numpy as np
from sklearn.model_selection import train_test_split

"""
def prepare_train_test(images, test_size=0.2, random_state=42):
    
    Takes a list of images in the format:
    (label, filename, img_array, feature)

    Returns:
        X_train, X_test, y_train, y_test
    

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
"""


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

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def train_polynomial_model(X_train, y_train, degree=2):
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    return model, poly

"""
def train_polynomial_model(X_train, y_train, degree=2):
    
    Trains a polynomial regression model.

    Returns:
        model, polynomial transformer
    

    # create polynomial feature transformer
    poly = PolynomialFeatures(degree=degree)

    # transform X into polynomial space
    X_train_poly = poly.fit_transform(X_train)

    # train linear model on transformed data
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    return model, poly
"""

def prepare_train_test(images, feature_key="fft_1d_energy", test_size=0.2, random_state=42):
    """
    Takes a list of image dicts and prepares train/test split.

    Parameters:
        images (list): list of dicts (with metadata + features)
        feature_key (str): which feature to use for X (e.g., "fft_1d_energy")

    Returns:
        X_train, X_test, y_train, y_test
    """

    # Extract X and y from dicts
    X = np.array([[img[feature_key]] for img in images])
    y = np.array([img["LevelingScore"] for img in images])

    # Train/test split (with stratification)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test