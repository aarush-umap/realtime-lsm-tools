"""
Algorithms in this file are derived from Autofocusing Algorithm Selection in Computer Microscopy Yu Sun et al.
"""

import numpy as np
import torch
from scipy.ndimage import sobel, laplace, convolve

# Derivative Based Algorithms

def threshold_absolute_gradient(image: np.ndarray , threshold: float=0, debug: bool=False) -> float:
    """ Returns Threshold Absolute Gradient value
    
    "It sums the absolute value of the first derivative that is larger than a threshold θ"
    
    image: a 2D grayscale image with shape (H,W)
    threshold: a value that each gradient must be greater than to be included in the sum
    debug: dictates whether to display debugging print statements or not
    """
    
    H, W = image.shape 

    
    values_x: np.ndarray = np.abs(image - np.roll(image, 1, 0))
    values_y: np.ndarray = np.abs(image - np.roll(image, 1, 1))
    
    if debug:
        print(f"x: {values_x.shape}\ty: {values_y.shape}")
    values_x[values_x < threshold] = 0
    values_y[values_y < threshold] = 0
    if debug:
        print(f"x: {values_x.shape}\ty: {values_y.shape}")

    result = np.sum(values_x[:] + values_y[:]).item()

    return result

def squared_gradient(image: np.ndarray , threshold: float=0) -> float:
    """ Returns Squared Gradient value
    
    "This algorithm sums squared differences, making larger gradients exert more influence"
    
    image: a 2D grayscale image with shape (H,W)
    threshold: a value that each gradient must be greater than to be included in the sum
    """
    H, W = image.shape


    values_x: np.ndarray = (image - np.roll(image, 1, 0))**2
    values_y: np.ndarray = (image - np.roll(image, 1, 1))**2
    values_x[values_x < threshold] = 0
    values_y[values_y < threshold] = 0

    result = np.sum(values_x[:] + values_y[:]).item()
    return result

def brenner_gradient(image: np.ndarray , threshold: float=0) -> float:
    """ Returns Brenner Gradient value
    
    "This algorithm computes the first difference between a pixel and its neighbor with a horizontal/vertical distance of 2"
    
    image: a 2D grayscale image with shape (H,W)
    threshold: a value that each gradient must be greater than to be included in the sum
    """
    H, W = image.shape


    b_x: np.ndarray = (image - np.roll(image, 2, 0))**2
    b_y: np.ndarray = (image - np.roll(image, 2, 1))**2
    b_x[b_x < threshold] = 0
    b_y[b_y < threshold] = 0

    result = np.sum(b_x[:] + b_y[:]).item()
    return result

def tenenbaum_gradient(image: np.ndarray ) -> float:
    """ Returns Tenenbaum Gradient value
    
    "This algorithm convolves an image with Sobel operators, and then sums the square of the gradient vector components"
    
    image: a 2D grayscale image with shape (H,W)
    """
    
    H, W = image.shape

    values_x: np.ndarray = sobel(image, 0)
    values_y: np.ndarray = sobel(image, 1)

    result = np.sum(values_x[:]**2 + values_y[:]**2).item()
    return result

def sum_of_modified_laplace(image: np.ndarray ) -> float:
    """ Returns Sum of modified Laplace value
    
    "This algorithm sums the absolute values of the convolution of an image with Laplacian operators"
    
    image: a 2D grayscale image with shape (H,W)
    """
    H, W = image.shape

    values: np.ndarray = np.abs(laplace(image))

    result = np.sum(values).item()
    return result

def energy_laplace(image: np.ndarray ) -> float:
    """ Returns Sum of modified Laplace value
    
    "This algorithm convolves an image with the mask
        [[-1 -4 -1],
         [-4 20 -4],
          [-1 -4 -1]]
    to compute the second derivative C(x, y). The final output is the sum of the squares
    of the convolution results."
    
    image: a 2D grayscale image with shape (H,W)
    """
    energy_matrix = np.array([[-1,-4,-1],[-4,20,-4],[-1,-4,-1]])
    H, W = image.shape

    values: np.ndarray = (convolve(image, energy_matrix))**2 # type: ignore

    result = np.sum(values).item()
    return result

def wavelet_alogrithm(image: np.ndarray ) -> float:
    return 0


# Statistics Based Algorithms

def defocused_variance(image: np.ndarray ) -> float:
    H, W = image.shape
    mean = np.mean(image)
    result = (1/(H*W)) * np.sum((image - mean)**2)
    return result.astype(float)

def normalized_variance(image: np.ndarray ) -> float:
    H, W = image.shape
    mean = np.mean(image)
    result = (1/(H*W*mean)) * np.sum((image - mean)**2)
    return result.astype(float)

def autocorrelation(image: np.ndarray ) -> float: 
    def auto_helper(image, number):
        values_x: np.ndarray = (image * np.roll(image, number, 0))
        values_y: np.ndarray = (image * np.roll(image, number, 1))
        return np.sum(values_x + values_y).item()

    result = auto_helper(image, 1) - auto_helper(image, 2)
    return result

def standard_deviation_based_correlation(image: np.ndarray ) -> float:
    # TODO: find a good way to consolidate reused code
    # TODO: figure out if the mean is per axis, or total image 
    H, W = image.shape
    def sd_helper(image, number) -> float:
        values_x: np.ndarray = (image * np.roll(image, number, 0)) 
        values_y: np.ndarray = (image * np.roll(image, number, 1))
        return np.sum(values_x + values_y).item()
    
    result = sd_helper(image, 1) - (H * W * (np.mean(image)**2))
    return result.item()


# Histogram Based Algorithms

def range_algorithm(image: np.ndarray, bin_width: int= 10) -> float:
    hist, bin_edges = np.histogram(image, bin_width)
    max_val = np.max(hist[hist > 0])
    min_val = np.min(hist[hist > 0])

    return max_val - min_val

def entropy_alogrithm(image: np.ndarray ) -> float:
    return 0


# Intuitive Algorithms TODO: comment and write tests

def thresholded_content(image: np.ndarray , threshold: float=0) -> float:
    H, W = image.shape
    
    image[image < threshold] = 0
    
    return np.sum(image).item()

def thresholded_pixel_count(image: np.ndarray , threshold: float=0) -> float:
    H, W = image.shape
    
    image[image > threshold] = 0
    
    return np.sum(image).item()

def image_power(image: np.ndarray, threshold: float=0) -> float:
    H, W = image.shape
    
    image[image < threshold] = 0
    
    return np.sum(image * image).item()
