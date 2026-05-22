import numpy as np

def append_mse_to_images(images):
    """
    Takes a list of images
    and returns a new list with MSE added.

    Output format:
    (...original list, mse)
    """
    updated_images = []

    for label, filename, img_array in images:
        mean_val = np.mean(img_array)
        mse = np.mean((img_array - mean_val) ** 2)

        updated_images.append((label, filename, img_array, mse))

    return updated_images
