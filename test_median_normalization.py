"""This file contains the full model pipeline as well as
instructions on how to add in new data."""

import sys
import os
import numpy as np
sys.path.append(os.path.abspath("."))

from src.data import load_images_from_metadata, filter_images
from src.build_metadata import load_metadata
from src.model import train_and_evaluate, export_onnx
from src.evaluation import evaluate_model


# Step 1: Build metadata.csv from images in the local data folders.
# If you are using the original shared dataset, copy the folders
# (data, data_new, data_new_2, data_test) into this project root first.
load_metadata(
    data_base_dir="data",
    new_data_base_dir="data_new",
    new_data_2_base_dir="data_new_2",
    output_file="metadata.csv",
)

# To add extra data, place it in a folder under the project root and extend
# metadata-building logic so rows are appended with this schema:
# LevelingScore, Person, DateCollected, ImageType, GSCamera, PanelID, State, FilePath.
# Refer to README.md for format expectations of above fields. 
#
# To run only a brand-new dataset, skip load_metadata() and create your own
# metadata.csv with the same schema.

# Step 2: Load images from metadata.csv.
# crop_fraction=0.4 keeps the center region by trimming edges before training.
data = load_images_from_metadata("metadata.csv", crop_fraction=0.4, use_cv2=True)

# Step 3: Filter the dataset used for training/evaluation.
# Standards are currently limited to specific collection dates due to
# known collection issues; adjust these filters if your dataset is different.
data_standards = filter_images(
    data,
    ImageType="STD",
    State="flat",
    DateCollected=["6.2.2026", "6.3.2026", "6.5.2026", "6.8.2026"],
)
data_real_paint = filter_images(data, ImageType="DD", State="flat")

# apply median normalization on data_standards and data_real_paint

def normalize_by_median(img, target_median=120):
    img = img.astype(float)

    current_median = np.median(img)

    if current_median == 0:
        return img

    scale = target_median / current_median

    normalized = img * scale

    return np.clip(normalized, 0, 255).astype(np.uint8)


def apply_median_normalization(dataset, target_median=120):
    """Return a copy of dataset with median-normalized images."""

    normalized_dataset = []
    for item in dataset:
        normalized_item = item.copy()
        normalized_item["image"] = normalize_by_median(
            normalized_item["image"], target_median=target_median
        )
        normalized_dataset.append(normalized_item)

    return normalized_dataset


data_standards = apply_median_normalization(data_standards, target_median=120)
data_real_paint = apply_median_normalization(data_real_paint, target_median=120)


# Step 4: Train the model and generate predictions on real paint data.
model, results_df, preds_np, labels_np, groups_real = train_and_evaluate(
    train_data=data_standards,
    test_data=data_real_paint,
    epochs=70,
    run_seed=5,
    save_dir="models",
)

# Step 5: Print evaluation metrics.
metrics = evaluate_model(preds_np, labels_np, groups_real)

# Step 6: Export ONNX model for downstream/frontend inference.
# export_onnx(model, save_path=os.path.join("models", "waveletcnn_v2.onnx"))


