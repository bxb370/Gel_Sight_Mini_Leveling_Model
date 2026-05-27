"""This file contains functions for calculating image features."""

import numpy as np
from scipy import ndimage
from scipy.stats import entropy

def append_entropy_to_images(images):
    """
    Takes a list of images
    and returns a new list with entropy added.

    Output format:
    (label, filename, img_array, entropy)
    """
    updated_images = []

    for label, filename, img_array in images:
        # Flatten image to 1D
        pixels = img_array.flatten()

        # Compute histogram (probability distribution)
        hist, _ = np.histogram(pixels, bins=256, range=(0, 256), density=True)

        # Remove zeros to avoid log issues
        hist = hist[hist > 0]

        # Compute entropy
        ent = entropy(hist)

        updated_images.append((label, filename, img_array, ent))

    return updated_images

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


def compute_edge_strength(img_array):
    """
    Computes edge strength from a single grayscale image array
    using the Sobel operator in the horizontal direction.
    """
    dx = ndimage.sobel(img_array, axis=1)
    edge_strength = np.mean(np.abs(dx))
    return edge_strength


def append_edge_strength_to_images(images):
    """
    Takes a list of image tuples and returns a new list
    with edge strength appended.

    Supports tuples that already include additional features
    (for example MSE) by preserving all existing elements.

    Output format:
    (...original tuple, edge_strength)
    """
    updated_images = []

    for item in images:
        img_array = item[2]
        edge_strength = compute_edge_strength(img_array)
        updated_images.append((*item, edge_strength))

    return updated_images

import numpy as np

"""
def append_fft_1d_energy_to_images(images):
    
    Takes a list of images
    and returns a new list with fft_1d_energy added.

    Output format:
    (label, filename, img_array, fft_1d_energy)
    
    updated_images = []

    for label, filename, img_array in images:

        # ✅ downsample for speed + stability
        img = img_array[::4, ::4].astype(float)

        # ✅ collapse to 1D signal (captures grooves)
        profile = img.mean(axis=0)

        # ✅ compute FFT
        fft_1d = np.fft.fft(profile)

        # ✅ magnitude (energy)
        mag = np.abs(fft_1d)

        # ✅ remove DC component (average brightness)
        mag[0] = 0

        # ✅ compute energy (mean magnitude)
        fft_1d_energy = np.mean(mag)

        updated_images.append((label, filename, img_array, fft_1d_energy))

    return updated_images
"""

def append_fft_1d_energy_to_images(images):
    """
    Adds fft_1d_energy to each image dict.

    Output:
        Each image dict gains:
            - "fft_1d_energy"
    """

    updated_images = []

    for img_dict in images:
        img_array = img_dict["image"]

        # ✅ downsample for speed + stability
        img = img_array[::4, ::4].astype(float)

        # ✅ collapse to 1D signal
        profile = img.mean(axis=0)

        # ✅ compute FFT
        fft_1d = np.fft.fft(profile)

        # ✅ magnitude
        mag = np.abs(fft_1d)

        # ✅ remove DC component
        mag[0] = 0

        # ✅ compute energy
        fft_1d_energy = np.mean(mag)

        # ✅ create updated copy (safer than modifying original)
        new_img_dict = img_dict.copy()
        new_img_dict["fft_1d_energy"] = fft_1d_energy

        updated_images.append(new_img_dict)

    return updated_images
