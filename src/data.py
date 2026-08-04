"""This file contains functions to load and filter images based on metadata."""

import pandas as pd
import os
import numpy as np
from PIL import Image
import cv2
import math


def normalize_by_median(img, target_median=120):
    """Scale image so its median pixel value matches target_median."""

    img = img.astype(float)
    current_median = np.median(img)

    if current_median == 0:
        return img

    scale = target_median / current_median
    normalized = img * scale

    return np.clip(normalized, 0, 255).astype(np.uint8)


def load_images_from_metadata(
    csv_path,
    base_dir=None,
    crop_fraction=None,
    use_cv2=False,
    median_normalization=False,
):
    """
    Loads images using metadata.csv, converts them to grayscale and optionally applies cropping.

    Parameters:
        csv_path (str): path to metadata.csv
        base_dir (str or None): base directory to prepend to FilePath
        crop_fraction (float or None): cropping fraction
        use_cv2 (bool): if True use OpenCV, otherwise PIL
        median_normalization (bool): if True apply normalize_by_median

    Returns:
        images (list of dicts)
    """

    df = pd.read_csv(csv_path)
    df = df.where(pd.notnull(df), None)

    images = []

    for _, row in df.iterrows():
        filepath = row["FilePath"]

        # ✅ Fix path if base_dir is provided
        if base_dir is not None:
            filepath = os.path.join(base_dir, filepath)

        filepath = os.path.normpath(filepath)

        try:
            # ✅ Load image (switchable backend)
            if use_cv2:
                img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

                if img_array is None:
                    raise ValueError("cv2.imread returned None")

            else:
                img = Image.open(filepath).convert("L")
                img_array = np.array(img)

            # ✅ cropping
            if crop_fraction is not None:
                h, w = img_array.shape
                crop_h = int((crop_fraction / 2) * h)
                crop_w = int((crop_fraction / 2) * w)

                img_array = img_array[
                    crop_h:h - crop_h,
                    crop_w:w - crop_w
                ]

            if median_normalization:
                img_array = normalize_by_median(img_array)

            record = {
                "LevelingScore": row["LevelingScore"],
                "Person": row["Person"],
                "DateCollected": row["DateCollected"],
                "ImageType": row["ImageType"],
                "GSCamera": row["GSCamera"],
                "PanelID": row["PanelID"],
                "State": row["State"],
                "FilePath": filepath,
                "image": img_array
            }

            images.append(record)

        except Exception as e:
            print(f"Warning: Failed to load {filepath} ({e})")

    print(f"Loaded {len(images)} images "
          f"(cropping={'ON' if crop_fraction is not None else 'OFF'}, "
            f"median_norm={'ON' if median_normalization else 'OFF'}, "
            f"backend={'cv2' if use_cv2 else 'PIL'})")

    return images

    
def filter_images(images, **filters):
    """
    Filters a list of image dictionaries based on metadata fields.

    Parameters:
        images (list): list of image dicts
        filters: key=value pairs (supports single value or list)

            Available fields:
            - LevelingScore
            - Person
            - DateCollected
            - ImageType (DD or STD)
            - GSCamera
            - PanelID
            - State (flat or raw)
            - FilePath

    Notes:
        - String matching is case-insensitive
        - Ignores None values in data

    Returns:
        list: filtered images
    """
    filtered = images

    for key, value in filters.items():
        if not isinstance(value, (list, tuple, set)):
            value = [value]

        # normalize strings
        value = [v.lower() if isinstance(v, str) else v for v in value]

        def is_null(x):
            return x is None or (isinstance(x, float) and math.isnan(x))

        def match(img_value):
            # Handle filtering for None/NaN
            if None in value:
                if is_null(img_value):
                    return True

            # Skip nulls if not explicitly requested
            if is_null(img_value):
                return False

            if isinstance(img_value, str):
                return img_value.lower() in value

            return img_value in value

        filtered = [
            img for img in filtered
            if key in img and match(img[key])
        ]

    print(f"Filtered down to {len(filtered)} images")
    return filtered