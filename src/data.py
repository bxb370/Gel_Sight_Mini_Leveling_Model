
"""This file handles loading the data"""

import os
import numpy as np
from PIL import Image

def load_images(data_type="raw", base_dir="../data"):
    """
    Loads images from a specified dataset (e.g., 'raw' or 'flat').

    Parameters:
        data_type (str): Subfolder name (e.g., 'raw', 'flat')
        base_dir (str): Base directory containing the data folders

    Returns:
        images (list): List of tuples (label, filename, img_array)
    """

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
    """
    Loads images and optionally crops them.

    Parameters:
        data_type (str): 'raw' or 'flat'
        base_dir (str): base data directory
        crop_fraction (float or None):
            total fraction to remove (e.g., 0.3 removes 30% total → 15% each side)

    Returns:
        images (list): (label, filename, img_array)
    """

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
    """
    Replaces the label in each image tuple with the average human rating
    from the PART_AVG_LABELS lookup table, matched by the part number
    found in the filename (a number between 100 and 150).

    Parameters:
        images (list): List of tuples (label, filename, img_array, ...)

    Returns:
        list: New list of tuples with the label replaced by the avg rating.
              Tuples whose filename contains no recognised part number are
              kept unchanged.
    """
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
