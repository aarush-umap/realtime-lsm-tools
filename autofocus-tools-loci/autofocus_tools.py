"""
Algorithms in this file are derived from Autofocusing Algorithm Selection in Computer Microscopy Yu Sun et al.
"""

import numpy as np
import torch
from scipy.ndimage import sobel, laplace

# Derivative Based Algorithms

def threshold_absolute_gradient(image: np.ndarray | torch.Tensor, threshold: float=0, debug: bool=False) -> float:
    """ Returns Threshold Absolute Gradient value
    
    "It sums the absolute value of the first derivative that is larger than a threshold θ"
    
    image: a 2D grayscale image with shape (H,W)
    threshold: a value that each gradient must be greater than to be included in the sum
    debug: dictates whether to display debugging print statements or not
    """
    
    H, W = image.shape 

    if (type(image) == np.ndarray) :
        image: torch.Tensor = torch.from_numpy(image)
    
    values_x: torch.Tensor = torch.abs(image - np.roll(image, 1, 0))
    values_y: torch.Tensor = torch.abs(image - np.roll(image, 1, 1))
    
    if debug:
        print(f"x: {values_x.shape}\ty: {values_y.shape}")
    values_x[values_x < threshold] = 0
    values_y[values_y < threshold] = 0
    if debug:
        print(f"x: {values_x.shape}\ty: {values_y.shape}")

    result = torch.sum(values_x[:] + values_y[:]).item()

    return result

def squared_gradient(image: np.ndarray | torch.Tensor, threshold: float=0) -> float:
    """ Returns Squared Gradient value
    
    "This algorithm sums squared differences, making larger gradients exert more influence"
    
    image: a 2D grayscale image with shape (H,W)
    threshold: a value that each gradient must be greater than to be included in the sum
    """
    H, W = image.shape

    if (type(image) == np.ndarray) :
        image: torch.Tensor = torch.from_numpy(image)

    values_x: torch.Tensor = (image - np.roll(image, 1, 0))**2
    values_y: torch.Tensor = (image - np.roll(image, 1, 1))**2
    values_x[values_x < threshold] = 0
    values_y[values_y < threshold] = 0

    result = torch.sum(values_x[:] + values_y[:]).item()
    return result

def brenner_gradient(image: np.ndarray | torch.Tensor, threshold: float=0) -> float:
    """ Returns Brenner Gradient value
    
    "This algorithm computes the first difference between a pixel and its neighbor with a horizontal/vertical distance of 2"
    
    image: a 2D grayscale image with shape (H,W)
    threshold: a value that each gradient must be greater than to be included in the sum
    """
    H, W = image.shape

    if (type(image) == np.ndarray) :
        image: torch.Tensor = torch.from_numpy(image)

    b_x: torch.Tensor = (image - np.roll(image, 2, 0))**2
    b_y: torch.Tensor = (image - np.roll(image, 2, 1))**2
    b_x[b_x < threshold] = 0
    b_y[b_y < threshold] = 0

    result = torch.sum(b_x[:] + b_y[:]).item()
    return result

def tenenbaum_gradient(image: np.ndarray | torch.Tensor) -> float:
    """ Returns Tenenbaum Gradient value
    
    "This algorithm convolves an image with Sobel operators, and then sums the square of the gradient vector components"
    
    image: a 2D grayscale image with shape (H,W)
    """
    
    H, W = image.shape

    if (type(image) == torch.Tensor) :
        image: np.ndarray = image.detach().numpy()

    values_x: np.ndarray = sobel(image, 0)
    values_y: np.ndarray = sobel(image, 1)

    result = np.sum(values_x[:]**2 + values_y[:]**2).item()
    return result

def sum_of_modified_laplace(image: np.ndarray | torch.Tensor) -> float:
    H, W = image.shape

    if (type(image) == torch.Tensor) :
        image: np.ndarray = image.detach().numpy()

    values: np.ndarray = np.abs(laplace(image))

    result = np.sum(values).item()
    return result

def energy_laplace(image: np.ndarray | torch.Tensor) -> float:
    return 0

def wavelet_alogrithm(image: np.ndarray | torch.Tensor) -> float:
    return 0


# Statistics Based Algorithms

def defocused_variance(image: np.ndarray | torch.Tensor) -> float:
    return 0

def normalized_variance(image: np.ndarray | torch.Tensor) -> float:
    return 0

def autocorrelation(image: np.ndarray | torch.Tensor) -> float: 
    return 0

def standard_deviation_based_correlation(image: np.ndarray | torch.Tensor) -> float:
    return 0


# Histogram Based Algorithms

def range_algorithm(image: np.ndarray | torch.Tensor) -> float:
    return 0

def entropy_alogrithm(image: np.ndarray | torch.Tensor) -> float:
    return 0


# Intuitive Algorithms

def thresholded_content(image: np.ndarray | torch.Tensor, threshold: float=0) -> float:
    return 0

def threholded_pixel_count(image: np.ndarray | torch.Tensor) -> float:
    return 0

def image_power(image: np.ndarray | torch.Tensor) -> float:
    return 0