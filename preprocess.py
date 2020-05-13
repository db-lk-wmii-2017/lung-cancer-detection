import numpy as np
import random
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import rotate


def normalize_scan_image(image):
    """From SITK - converting hounds units to grayscale units"""
    maxHU = 400.0
    minHU = -1000.0

    image = (image - minHU) / (maxHU - minHU)
    image[image > 1] = 1.0
    image[image < 0] = 0.0

    return (image * 255).astype(np.uint8)


def apply_gaussian_filter(image, max_sigma=3.0):
    if bool(random.getrandbits(1)):
        sigma = random.uniform(0.0, max_sigma)
        image = gaussian_filter(image, sigma)
    return image


def rotate_image(image, deg):
    return rotate(image, deg, reshape=False)
