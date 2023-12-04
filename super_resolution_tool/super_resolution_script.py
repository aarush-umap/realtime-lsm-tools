from skimage import io
import numpy as np
import yaml
from .enhancer import Enhancer
import os.path as path
import torch
from .lsm_utils import compute_norm_range


def compute_sr(image_dir: str = "cropped_sample_data", image_name: str = "PB522-14-MAX_Fused.tif"):

    vmin, vmax, fail_names = compute_norm_range(image_dir, ext='tif', percentiles=(1, 99.5), sample_r=1)
    dir_path = path.dirname(path.realpath(__file__))
    config = yaml.load(open(path.join(dir_path, "model_config.yaml"), "r"), Loader=yaml.FullLoader)
    config['norm-range'] = [int(vmin), int(vmax)]

    enhancer = Enhancer(config, scale_factor=2, perceptual_loss=False, adversarial_loss=False)
    enhancer.backbone.load_state_dict(torch.load(path.join(dir_path, 'model_weights', 'enhancer.pth')))

    image = io.imread(path.join(image_dir, image_name))
    output = enhancer.compute(image)
    # io.imsave(path.join("output", image_name), output)
    return output