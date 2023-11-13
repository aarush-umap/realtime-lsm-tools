from skimage import io
from skimage import img_as_uint
import numpy as np
import os
import torch

from denoiser import Denoiser
from lsm_utils import compute_norm_range
import yaml

import warnings
from skimage import exposure
import torch.nn.functional as F 

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
torch.cuda.empty_cache()

def compute_denoise(image_dir: str = "cropped_sample_data", image_name: str = "PB522-14-MAX_Fused.tif"):
    vmin, vmax, fail_names = compute_norm_range(image_dir, ext='tif', percentiles=(1, 99.5), sample_r=1)
    config = yaml.load(open("model_config.yaml", "r"), Loader=yaml.FullLoader)
    config['dataset'] = image_dir
    config['norm-range'] = [int(vmin), int(vmax)]
    denoiser = Denoiser(config, screen_bg=False)

    # Set vars without config file
    average_factor = 50
    blindspot_rate = 0.05
    batch_size = 50
    dataset_name = "cropped_sample_data"
    fname = os.path.join(dataset_name,"PB522-14-MAX_Fused.tif")
    clean_path = os.path.join("output-self", dataset_name, "clean")
    noisy_path = os.path.join("output-self", dataset_name, "noisy")

    # Set up directories
    os.makedirs(clean_path, exist_ok=True)
    os.makedirs(noisy_path, exist_ok=True)

    pass_times = int(1/blindspot_rate * average_factor)
    iterations = int(np.ceil(pass_times/batch_size))

    model = denoiser.backbone
    device = next(model.parameters()).device

    img_arr = img_as_uint(io.imread(fname))
    img_arr = exposure.rescale_intensity(img_arr, in_range=(int(vmin), int(vmax)), out_range=(0, 65535)).astype(int)
    img_input = exposure.rescale_intensity(img_arr, in_range=(0, 65535), out_range=(0, 1))
    img_tensor = torch.from_numpy(img_input)
    img_hyper_tensor = img_tensor.expand([batch_size, 1, img_tensor.shape[0], img_tensor.shape[1]]).float().to(device)
    out_tensor = img_tensor * 0

    for i in range(iterations):
        drop_mask = F.dropout(torch.ones(img_hyper_tensor.shape, requires_grad=False).to(device), p=blindspot_rate, inplace=True)*(1-blindspot_rate) # p percent zero, keep
        pad_mask = (1-drop_mask) * torch.ones(img_hyper_tensor.shape, device=device, dtype=torch.float32) * torch.mean(img_hyper_tensor, (2, 3), keepdim=True).expand_as(img_hyper_tensor)
        spotted = torch.mul(img_hyper_tensor, drop_mask) + pad_mask
        prediction = model(spotted)
        prediction = torch.mul(prediction, 1-drop_mask)/blindspot_rate
        out_tensor += torch.mean(prediction, 0).squeeze().cpu()/iterations
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out_arr = img_as_uint(np.clip(out_tensor.detach().numpy().squeeze(), 0, 1))
        img_name = os.path.basename(fname)
        io.imsave(os.path.join(clean_path, img_name), out_arr)
        io.imsave(os.path.join(noisy_path, img_name), img_as_uint(img_arr))
    print(f'Processed', end='\r')
