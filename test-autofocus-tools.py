import autofocus_tools_loci.autofocus_tools as af
import numpy as np
import pytest
from skimage import io

def test_threshold_absolute_gradient(image: np.ndarray):
    if len(image.shape) == 3:
        results = []
    else:
        result = af.threshold_absolute_gradient(image, debug=False)
        print(result)

def test_brenner_gradient(image: np.ndarray):
    result = af.brenner_gradient(image)
    print(result)

def test_squared_gradient(image: np.ndarray):
    result = af.squared_gradient(image, threshold=0)
    print(result)
    
def test_tenengrad(image: np.ndarray):
    result = af.tenenbaum_gradient(image)
    print(result)
    
def test_sum_of_modified_laplace(image: np.ndarray):
    result = af.sum_of_modified_laplace(image)
    print(result)

def test_energy_laplace(image: np.ndarray):
    result = af.energy_laplace(image)
    print(result)

def test_autocorrelation(image: np.ndarray):
    result = af.autocorrelation(image)
    print(result)

# Run tests
example_array = np.array([[1,2,3],[4,5,6],[7,8,9]])
acceptable = [i for i in range(21,27)]
print(acceptable)
test_image = io.imread(r'test_data\Zstack_HistopathologySlide_CAMM_1_MMStack_Pos0.ome.tif')
test_image = test_image[:,:,:,0]*.3 + test_image[:,:,:,1]*.59 + test_image[:,:,:,2]*.11 
# test_threshold_absolute_gradient(example_array)
# test_brenner_gradient(example_array)
# test_squared_gradient(example_array)
# test_tenengrad(example_array)
# test_sum_of_modified_laplace(example_array)
# test_energy_laplace(example_array)
test_autocorrelation(example_array)