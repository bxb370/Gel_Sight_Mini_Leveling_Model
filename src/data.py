
"""This file handles loading the data"""

import pandas as pd
import numpy as np
from PIL import Image

import pandas as pd
import numpy as np
from PIL import Image
import os

def load_images_from_metadata(csv_path, base_dir=None, crop_fraction=None):
    """
    Loads images using metadata.csv and optionally applies cropping.

    Parameters:
        csv_path (str): path to metadata.csv
        base_dir (str or None): base directory to prepend to FilePath
        crop_fraction (float or None): cropping fraction

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

        # ✅ Normalize path (handles \ vs / issues)
        filepath = os.path.normpath(filepath)

        try:
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
          f"(cropping={'ON' if crop_fraction is not None else 'OFF'})")

    return images

    
import math

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


def _normalize_compare_path(path_value):
    """Normalize a path so equivalent paths compare equal across notebooks/scripts."""
    if path_value is None:
        return None

    normalized = os.path.normpath(str(path_value)).replace("\\", "/").lower()

    # Remove leading ./ or ../ segments so relative base differences do not hide matches.
    while normalized.startswith("./") or normalized.startswith("../"):
        normalized = normalized.split("/", 1)[1] if "/" in normalized else ""

    return normalized


def compare_image_collections(images_a, images_b, key="FilePath", normalize_paths=True):
    """
    Compare two image-record lists and return what exists in each side.

    Parameters:
        images_a (list[dict]): first image collection
        images_b (list[dict]): second image collection
        key (str): record key to compare (default: FilePath)
        normalize_paths (bool): normalize FilePath values before comparison

    Returns:
        dict: {
            "count_a": int,
            "count_b": int,
            "unique_keys_a": int,
            "unique_keys_b": int,
            "only_in_a": list[str],
            "only_in_b": list[str],
            "shared_keys": list[str],
            "duplicate_keys_a": dict[str, int],
            "duplicate_keys_b": dict[str, int],
        }
    """

    def to_compare_key(record):
        value = record.get(key)
        if key == "FilePath" and normalize_paths:
            return _normalize_compare_path(value)
        return value

    keys_a = [to_compare_key(record) for record in images_a]
    keys_b = [to_compare_key(record) for record in images_b]

    set_a = set(keys_a)
    set_b = set(keys_b)

    duplicate_keys_a = {}
    duplicate_keys_b = {}

    for candidate in set_a:
        count = keys_a.count(candidate)
        if count > 1:
            duplicate_keys_a[candidate] = count

    for candidate in set_b:
        count = keys_b.count(candidate)
        if count > 1:
            duplicate_keys_b[candidate] = count

    def sorted_keys(values):
        return sorted(values, key=lambda x: "" if x is None else str(x))

    return {
        "count_a": len(images_a),
        "count_b": len(images_b),
        "unique_keys_a": len(set_a),
        "unique_keys_b": len(set_b),
        "only_in_a": sorted_keys(set_a - set_b),
        "only_in_b": sorted_keys(set_b - set_a),
        "shared_keys": sorted_keys(set_a & set_b),
        "duplicate_keys_a": duplicate_keys_a,
        "duplicate_keys_b": duplicate_keys_b,
    }


"""
import os
import numpy as np
from PIL import Image

def load_images(data_type="raw", base_dir="../data"):
    
    Loads images from a specified dataset (e.g., 'raw' or 'flat').

    Parameters:
        data_type (str): Subfolder name (e.g., 'raw', 'flat')
        base_dir (str): Base directory containing the data folders

    Returns:
        images (list): List of tuples (label, filename, img_array)
    

    data_dir = os.path.join(base_dir, data_type)

    images = []

    for label_folder in sorted(os.listdir(data_dir), key=lambda x: int(x)):
        folder_path = os.path.join(data_dir, label_folder)

        if not os.path.isdir(folder_path):
            continue

        label = int(label_folder)

        for filename in sorted(os.listdir(folder_path)):
            if filename.lower().endswith(".png"):
                img_path = os.path.join(folder_path, filename)

                img = Image.open(img_path).convert("L")
                img_array = np.array(img)

                images.append((label, filename, img_array))

    print(f"Loaded {len(images)} images from '{data_type}'")

    return images

import os
import numpy as np
from PIL import Image

def load_images_optional_crop(data_type="raw", base_dir="../data", crop_fraction=None):
    
    Loads images and optionally crops them.

    Parameters:
        data_type (str): 'raw' or 'flat'
        base_dir (str): base data directory
        crop_fraction (float or None):
            total fraction to remove (e.g., 0.3 removes 30% total → 15% each side)

    Returns:
        images (list): (label, filename, img_array)
    

    data_dir = os.path.join(base_dir, data_type)
    images = []

    for label_folder in sorted(os.listdir(data_dir), key=lambda x: int(x)):
        folder_path = os.path.join(data_dir, label_folder)

        if not os.path.isdir(folder_path):
            continue

        label = int(label_folder)

        for filename in sorted(os.listdir(folder_path)):
            if filename.lower().endswith(".png"):
                img_path = os.path.join(folder_path, filename)

                img = Image.open(img_path).convert("L")
                img_array = np.array(img)

                # Apply cropping IF requested
                if crop_fraction is not None:
                    h, w = img_array.shape

                    crop_h = int((crop_fraction / 2) * h)
                    crop_w = int((crop_fraction / 2) * w)

                    img_array = img_array[
                        crop_h:h - crop_h,
                        crop_w:w - crop_w
                    ]

                images.append((label, filename, img_array))

    print(f"Loaded {len(images)} images from '{data_type}' "
          f"(cropping={'ON' if crop_fraction else 'OFF'})")

    return images


# Lookup table: part number -> average human rating
PART_AVG_LABELS = {
    100: 2.2, 101: 2.2, 102: 1.8, 103: 2.5, 104: 6.8,
    105: 8.3, 106: 3.5, 107: 7.2, 108: 1.2, 109: 2.3,
    110: 4.0, 111: 1.3, 112: 1.8, 113: 2.5, 114: 1.5,
    115: 5.2, 116: 3.7, 117: 2.2, 118: 3.7, 119: 4.0,
    120: 3.2, 121: 2.0, 122: 2.3, 123: 2.2, 124: 9.5,
    125: 8.0, 126: 2.7, 127: 3.0, 128: 7.2, 129: 2.2,
    130: 1.3, 131: 5.5, 132: 6.7, 133: 4.5, 134: 9.5,
    135: 9.0, 136: 6.3, 137: 3.2, 138: 2.7, 139: 2.0,
    140: 4.8, 141: 6.3, 142: 3.2, 143: 5.8, 144: 5.2,
    145: 4.2, 146: 6.3, 147: 8.3, 148: 7.5, 149: 9.5,
    150: 5.0,
}


def relabel_images_by_part_number(images):
    
    Replaces the label in each image tuple with the average human rating
    from the PART_AVG_LABELS lookup table, matched by the part number
    found in the filename (a number between 100 and 150).

    Parameters:
        images (list): List of tuples (label, filename, img_array, ...)

    Returns:
        list: New list of tuples with the label replaced by the avg rating.
              Tuples whose filename contains no recognised part number are
              kept unchanged.
    
    import re

    relabeled = []
    for item in images:
        filename = item[1]
        match = re.search(r'GS-25-(\d{3})', filename)
        if match:
            part_number = int(match.group(1))
            avg_label = PART_AVG_LABELS.get(part_number)
            if avg_label is not None:
                relabeled.append((avg_label,) + item[1:])
                continue
        relabeled.append(item)

    return relabeled
"""