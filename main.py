"""
• Key rule: if something works, move it to src/
• Key idea: 
	• src/ = reusable building blocks (functions)
	• main.py = experiment runner / pipeline controller
	So instead of “doing everything in main,” you’re really:
orchestrating experiments in main.py using functions from src/
"""

"""
one script that does everything, loads the data, gets the metrics, trains the model, and evaluates the model
"""

import sys
import os
import importlib
import numpy as np
from src.data import load_images_optional_crop
from src.features import append_entropy_to_images
from src.model import prepare_train_test, train_linear_model
from src.evaluation import evaluate_model_mae


#load images
images = load_images_optional_crop("flat", crop_fraction=0.30)

#compute entropy and append to images
images = append_entropy_to_images(images)

# prepare the 80/20 train test split
X_train, X_test, y_train, y_test = prepare_train_test(images)

# train the model
model = train_linear_model(X_train, y_train)

# evaluate the model
mae = evaluate_model_mae(model, X_test, y_test)
print("MAE:", mae)