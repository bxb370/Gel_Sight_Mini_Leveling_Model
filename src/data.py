
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
