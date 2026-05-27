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
from src.model import prepare_train_test, train_linear_model, train_polynomial_model
from src.evaluation import evaluate_linear_model
from src.features import append_fft_1d_energy_to_images
from src.evaluation import evaluate_polynomial_model
from src.data import relabel_images_by_part_number
from src.data import load_images_from_metadata
from src.data import filter_images

"""
#load images
images = load_images_optional_crop("flat", base_dir="data", crop_fraction=0.30)

#compute fft_1d_energy and append to images
images = append_fft_1d_energy_to_images(images)

# prepare the 80/20 train test split
X_train, X_test, y_train, y_test = prepare_train_test(images)

# train the model
model, poly = train_polynomial_model(X_train, y_train, degree=2)

# evaluate the model
mae = evaluate_polynomial_model(model, poly, X_test, y_test)
print("MAE:", mae)
"""


## now lets try with the new metadata.csv stuff

#load images
images = load_images_from_metadata("data/metadata.csv", base_dir="data", crop_fraction=0.30)

images = filter_images(images, State="flat")

#compute fft_1d_energy and append to images
images = append_fft_1d_energy_to_images(images)

# prepare the 80/20 train test split
X_train, X_test, y_train, y_test = prepare_train_test(images, feature_key="fft_1d_energy")

# train the model
model, poly = train_polynomial_model(X_train, y_train, degree=2)

# evaluate the model
mae = evaluate_polynomial_model(model, poly, X_test, y_test)
print("MAE:", mae)
