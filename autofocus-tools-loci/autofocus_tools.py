import numpy as np
import torch

# Derivative Based Algorithms

def threshold_absolute_gradient(image: np.ndarray | torch.Tensor, threshold: float) -> float:
    H, W = image.shape 
    th_grad = 0



    return 0

def squared_gradient(image: np.ndarray | torch.Tensor) -> float:
    return 0

def brenner_gradient(image: np.ndarray | torch.Tensor) -> float:
    H, W = image.shape

    if (type(image) == np.ndarray) :
        image: torch.Tensor = torch.from_numpy(image)

    b_x = (image[0:H-2, :] - image[2:, :])**2
    b_y = (image[:, 0:W-2] - image[:, 2: ])**2
    result = torch.sum(b_x[:] + b_y[:]).item()
    return result

def tenenbaum_gradient(image: np.ndarray | torch.Tensor) -> float:
    return 0

def sum_of_modified_laplace(image: np.ndarray | torch.Tensor) -> float:
    return 0

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

def thresholded_content(image: np.ndarray | torch.Tensor) -> float:
    return 0

def threholded_pixel_count(image: np.ndarray | torch.Tensor) -> float:
    return 0

def image_power(image: np.ndarray | torch.Tensor) -> float:
    return 0