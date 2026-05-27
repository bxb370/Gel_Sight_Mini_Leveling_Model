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
from src.features import append_entropy_to_images
from src.model import prepare_train_test, train_linear_model, train_polynomial_model
from src.evaluation import evaluate_linear_model
from src.features import append_fft_1d_energy_to_images
from src.evaluation import evaluate_polynomial_model
from src.data import load_images_from_metadata
from src.data import filter_images
from src.features import append_fft_2d_energy_to_images
from src.model import prepare_train_test_multi
from src.model import train_random_forest_model
from src.evaluation import evaluate_model_mae


#lets try random forest and see how that does
#load images
images = load_images_from_metadata("data/metadata.csv", base_dir="data", crop_fraction=0.30)

#filter imgaes to just flat ones
images = filter_images(images, State="flat")

#compute fft_1d_energy and fft_2d_energy and append to images
images = append_fft_1d_energy_to_images(images)
images = append_fft_2d_energy_to_images(images)

# prepare the 80/20 train test split with multiple features
X_train, X_test, y_train, y_test = prepare_train_test_multi(images, feature_keys=("fft_1d_energy", "fft_2d_energy"))

#train the model
model = train_random_forest_model(X_train, y_train)

# evaluate the model
mae = evaluate_model_mae(model, X_test, y_test)
print("MAE:", mae)
