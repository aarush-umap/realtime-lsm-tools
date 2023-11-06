import autofocus_tools 
import numpy as np
import torch

def test_threshold_absolute_gradient(image: np.ndarray):
    result = autofocus_tools.threshold_absolute_gradient(image, debug=True)
    print(result)

def test_brenner_gradient(image: np.ndarray):
    result = autofocus_tools.brenner_gradient(image)
    print(result)

def test_squared_gradient(image: np.ndarray):
    result = autofocus_tools.squared_gradient(image, threshold=0)
    print(result)
    
def test_tenengrad(image: np.ndarray):
    result = autofocus_tools.tenenbaum_gradient(image)
    print(result)

# Run tests
example_array = np.array([[1,2,3],[4,5,6],[7,8,9]])
test_threshold_absolute_gradient(example_array)
test_brenner_gradient(example_array)
test_squared_gradient(example_array)
