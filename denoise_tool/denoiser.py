from logging import warn
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Adam
from .lsm_dataset import generate_compress_csv, data_loader
import os
import shutil
import random
from glob import glob
from skimage import img_as_uint
from skimage import io 
from skimage import exposure
import numpy as np
from tqdm import tqdm
import warnings
from torch.utils.tensorboard import SummaryWriter
import datetime

class Denoiser(nn.Module):
    def __init__(self, config, screen_bg=True):
        super(Denoiser, self).__init__()

        self.config = config
        os.makedirs('output-self', exist_ok=True)
        self.writer = SummaryWriter()
        self.log_train = []
        self.log_test = []

        self.backbone = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=config['image-channel'], 
            out_channels=config['image-channel'], 
            init_features=config['cnn-base-channel'], 
            pretrained=False)

        self.configure_dataset(screen_bg)
        self.configure_optimizer()
        self.criterion = nn.L1Loss(reduction='none')
        self.alpha = config['loss-gain']

        if self.config['gpu']: self.cuda()
        
        mydir = os.path.join('model_weights', 'self-supervised-'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(mydir, exist_ok=True)
        self.weights_dir = mydir


    def configure_dataset(self, exclude_bg=True):
        config = self.config
        train_csv_path, valid_csv_path = generate_compress_csv(dataset=config['dataset'], ext=str(config['image-extension']), exclude_bg=exclude_bg)
        valid_dataloader = data_loader(valid_csv_path, config['batch-size'], config['norm-range'], config['threads'], config['resolution'])
        train_dataloader= data_loader(train_csv_path, config['batch-size'], config['norm-range'], config['threads'], config['resolution'])
        self.valid_dataloader = valid_dataloader
        self.train_dataloader = train_dataloader

    def configure_optimizer(self, min_lr=0.000005):
        config = self.config
        n_epoch = int(config["iterations"]/len(self.train_dataloader))
        self.optimizer = Adam(self.backbone.parameters(), lr=config['learning-rate'], weight_decay=0.0001)
        self.scheduler = CosineAnnealingLR(self.optimizer, n_epoch, min_lr)


    def forward(self, x):
        out = self.backbone(x)
        return out


    def train_epoch(self, epoch=1, total_epoch=1):
        model = self.backbone
        config = self.config
        dataloader = self.train_dataloader
        criterion = self.criterion
        optimizer = self.optimizer
        p = self.config['blindspot-rate']
        epoch_loss = 0
        model.train()
        device = next(model.parameters()).device
        n_iter = min(len(dataloader), config['iter-per-epoch'])
        for iteration, batch in enumerate(dataloader):
            if iteration >= n_iter: break
            noisy = batch['input'].float().to(device)
            ### generate 2d dropout
            drop_mask = F.dropout(torch.ones(noisy.shape, requires_grad=False).to(device), p=p, inplace=True)*(1-p) # p percent zero, keep
            pad_mask = (1-drop_mask) * torch.ones(noisy.shape, device=device, dtype=torch.float32) * torch.mean(noisy, (2, 3), keepdim=True).expand_as(noisy)
            spotted = torch.mul(noisy, drop_mask) + pad_mask
            self.optimizer.zero_grad()
            clean = model(spotted)
            loss_pixel = torch.mean(torch.mul(criterion(clean*self.alpha, noisy*self.alpha), 1-drop_mask))/p                  
            loss_pixel.backward()
            optimizer.step()
            epoch_loss = epoch_loss + loss_pixel.item()
            print(f'[{epoch}/{total_epoch}] [{iteration}/{n_iter}] Loss: {loss_pixel.item()}', end='\r')
        print(f"\n ===> Epoch {epoch} Complete: Avg. Loss: {(epoch_loss / n_iter):.6f}")
        self.log_train.append((epoch_loss/n_iter, epoch))


    def test(self, epoch):
        with torch.no_grad():
            model = self.backbone
            dataloader = self.valid_dataloader
            criterion = self.criterion
            p = self.config['blindspot-rate']
            model.eval()
            epoch_loss = 0
            device = next(model.parameters()).device
            for batch in tqdm(dataloader):
                noisy = batch['input'].float().to(device)
                drop_mask = F.dropout(torch.ones(noisy.shape, requires_grad=False).to(device), p=p, inplace=True)*(1-p) # p percent zero, keep
                pad_mask = (1-drop_mask) * torch.ones(noisy.shape, device=device, dtype=torch.float32) * torch.mean(noisy, (2, 3), keepdim=True).expand_as(noisy)
                spotted = torch.mul(noisy, drop_mask) + pad_mask
                clean = model(spotted) # N x C x H x W
                ### loss
                loss = torch.mean(torch.mul(criterion(clean*self.alpha, noisy*self.alpha), 1-drop_mask))/p
                epoch_loss = epoch_loss + loss.item()
            print(f'>>>> Test Loss: {epoch_loss / len(dataloader):.6f}', end='\n')
            self.log_test.append((epoch_loss/len(dataloader), epoch))


    def write_log(self, write_train=True):
        if write_train:
            self.writer.add_scalar('Loss/train', self.log_train[-1][0], self.log_train[-1][1])
        self.writer.add_scalar('Loss/test', self.log_test[-1][0], self.log_test[-1][1])
        
        
    def save_models(self):
        torch.save(self.backbone.state_dict(), os.path.join(self.weights_dir, 'g.pth'))


    def train(self, write_log=False, valid_r=0.01):
        config = self.config
        scheduler = self.scheduler
        n_epoch = int(config["iterations"]/min(config["iter-per-epoch"], len(self.train_dataloader)))
        print('Initial testing pass...')
        self.test(0)
        self.write_log(False)
        for epoch in tqdm(range(1, n_epoch+1)):
            self.train_epoch(epoch=epoch, total_epoch=n_epoch)   
            scheduler.step()
            if epoch % config["test-interval"] == 0:
                self.test(epoch)
                self.denoise(sampling=True, sample_rate=valid_r)
            if write_log:
                self.write_log()
            self.save_models()
            

    def denoise(self, sampling=False, sample_rate=1, batch_size=50):
        with torch.no_grad():
            config = self.config
            p = self.config['blindspot-rate']
            model = self.backbone
            os.makedirs(os.path.join('output-self', config['dataset']), exist_ok=True)
            noisy_path = os.path.join('output-self', config['dataset'], 'noisy')
            clean_path = os.path.join('output-self', config['dataset'], 'clean')
            input_images = glob(os.path.join(config['dataset'], '*.'+str(config['image-extension'])))
            if sampling:
                shutil.rmtree(os.path.join('output-self', config['dataset']))
                input_images = random.sample(input_images, int(len(input_images)*sample_rate))
            os.makedirs(noisy_path, exist_ok=True)
            os.makedirs(clean_path, exist_ok=True)
            pass_times = int(1/p * config['average-factor'])
            iterations = int(np.ceil(pass_times/batch_size))
            device = next(model.parameters()).device
            for idx, fname in enumerate(input_images):
                img_arr = img_as_uint(io.imread(fname))
                img_arr = exposure.rescale_intensity(img_arr, in_range=(config['norm-range'][0], config['norm-range'][1]), out_range=(0, 65535)).astype(int)
                img_input = exposure.rescale_intensity(img_arr, in_range=(0, 65535), out_range=(0, 1))
                img_tensor = torch.from_numpy(img_input)
                img_hyper_tensor = img_tensor.expand([batch_size, 1, img_tensor.shape[0], img_tensor.shape[1]]).float().to(device)
                out_tensor = img_tensor * 0
                for i in range(iterations):
                    drop_mask = F.dropout(torch.ones(img_hyper_tensor.shape, requires_grad=False).to(device), p=p, inplace=True)*(1-p) # p percent zero, keep
                    pad_mask = (1-drop_mask) * torch.ones(img_hyper_tensor.shape, device=device, dtype=torch.float32) * torch.mean(img_hyper_tensor, (2, 3), keepdim=True).expand_as(img_hyper_tensor)
                    spotted = torch.mul(img_hyper_tensor, drop_mask) + pad_mask
                    prediction = model(spotted)
                    prediction = torch.mul(prediction, 1-drop_mask)/p
                    out_tensor += torch.mean(prediction, 0).squeeze().cpu()/iterations
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    out_arr = img_as_uint(np.clip(out_tensor.numpy().squeeze(), 0, 1))
                    img_name = os.path.basename(fname)
                    io.imsave(os.path.join(clean_path, img_name), out_arr)
                    io.imsave(os.path.join(noisy_path, img_name), img_as_uint(img_arr))
                print(f'Processed [{idx+1}/{len(input_images)}]', end='\r')

    
    # def compute(self, img_arr):
    #     with torch.no_grad():
    #         config = self.config
    #         p = self.config['blindspot-rate']
    #         model = self.backbone
    #         device = next(model.parameters()).device
    #         pass_times = int(1/p * config['average-factor'])
    #         iterations = int(np.ceil(pass_times/batch_size))
    #         img_arr = img_as_uint(img_arr)
    #         img_arr = exposure.rescale_intensity(img_arr, in_range=(config['norm-range'][0], config['norm-range'][1]), out_range=(0, 65535)).astype(int)
    #         img_input = exposure.rescale_intensity(img_arr, in_range=(0, 65535), out_range=(0, 1))
    #         img_tensor = torch.from_numpy(img_input)
    #         img_hyper_tensor = img_tensor.expand([batch_size, 1, img_tensor.shape[0], img_tensor.shape[1]]).float().to(device)
    #         for i in range(iterations):
    #             drop_mask = F.dropout(torch.ones(img_hyper_tensor.shape, requires_grad=False).to(device), p=p, inplace=True)*(1-p) # p percent zero, keep
    #             pad_mask = (1-drop_mask) * torch.ones(img_hyper_tensor.shape, device=device, dtype=torch.float32) * torch.mean(img_hyper_tensor, (2, 3), keepdim=True).expand_as(img_hyper_tensor)
    #             spotted = torch.mul(img_hyper_tensor, drop_mask) + pad_mask
    #             prediction = model(spotted)
    #             prediction = torch.mul(prediction, 1-drop_mask)/p
    #             out_tensor += torch.mean(prediction, 0).squeeze().cpu()/iterations
    #         out_arr = img_as_uint(np.clip(out_tensor.numpy().squeeze(), 0, 1))
    #         return out_arr