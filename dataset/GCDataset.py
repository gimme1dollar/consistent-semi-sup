import numpy as np
import torch
import torch.nn as nn
import random, math
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
import os
import glob
from PIL import Image
from typing import Callable

class GCDataset(Dataset):
    def __init__(self, data_path, model):
        super(GCDataset, self).__init__()
        self.data_path = data_path
        self.load_dataset()

        self.model = model
        self.save_feat=[]
        self.save_grad=[] 

        self.to_tensor_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.resize_transform = transforms.Compose([
            transforms.Resize((64, 64))
        ])

    def load_dataset(self):
        root = os.path.join(self.data_path)
        print("root : ", root)
        self.data = glob.glob(root+"/*/*.JPEG")

    def __len__(self):
        return len(self.data)

    def save_outputs_hook(self) -> Callable:
        def fn(_, __, output):
            self.save_feat.append(output)
        return fn

    def save_grad_hook(self) -> Callable:
        def fn(grad):
            self.save_grad.append(grad)
        return fn

    def get_grad_cam(self, image):
        self.save_feat=[]
        s = self.model(image)[0]

        self.save_grad=[]
        self.save_feat[0].register_hook(self.save_grad_hook())
        
        y = torch.argmax(s).item()
        s_y = s[y]
        s_y.backward()

        gap_layer  = torch.nn.AdaptiveAvgPool2d(1)
        alpha = gap_layer(self.save_grad[0][0])
        A = self.save_feat[0]
        A = A.squeeze()
        relu_layer = torch.nn.ReLU()

        weighted_sum = torch.sum(alpha*A, dim=0)
        grad_CAM = relu_layer(weighted_sum)
        grad_CAM = grad_CAM.unsqueeze(0)
        grad_CAM = grad_CAM.unsqueeze(0)

        upscale_layer = torch.nn.Upsample(scale_factor=image.shape[-1]/grad_CAM.shape[-1], mode='bilinear', align_corners=True)
        grad_CAM = upscale_layer(grad_CAM)
        grad_CAM = grad_CAM/torch.max(grad_CAM)

        return grad_CAM

    def get_params(self, img):
        w_, h_ = img.size(2), img.size(3)
        xl = random.randint(0, w_/8)
        xr = 0
        while(((xr-xl) < (w_*7/8)) and (xr <= xl)):
          xr = random.randint(xl, w_)

        yl = random.randint(0, h_/8)
        yr = 0
        while(((yr-yl) < (h_*7/8)) and (yr <= yl)):
          yr = random.randint(yl, h_)

        return xl, yl, xr, yr

    def color_augmentation(self, i, img):
        color_transform = transforms.Compose([
            transforms.ColorJitter(i, i, i, i)
        ])

        return color_transform(img)

    def __getitem__(self, index):
        self.model.bn2.register_forward_hook(self.save_outputs_hook())
        
        image = Image.open(self.data[index]).convert('RGB')
        image = self.to_tensor_transform(image)
        image = image.unsqueeze(0)
        image = self.resize_transform(image)
        
        i, j, h, w = self.get_params(image)
        st_image = F.crop(image, i, j, h, w)
        st_image = self.color_augmentation(0.5, st_image)
        st_image = self.resize_transform(st_image)
        
        wk_image = self.color_augmentation(0.1, image)
        
        
        wk_gradcam = self.get_grad_cam(wk_image)
        
        st_gradcam = self.get_grad_cam(st_image)

        wk_gradcam = wk_gradcam.detach()
        gt_gradcam = F.crop(wk_gradcam, i, j, h, w)
        gt_gradcam = self.resize_transform(gt_gradcam)
        
        return wk_image, wk_gradcam, st_image, st_gradcam, gt_gradcam