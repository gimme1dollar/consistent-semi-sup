import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import glob
from PIL import Image

class LoadDataset(Dataset):
    def __init__(self, data_path, transform, mode='train'):
        super(LoadDataset, self).__init__()
        self.data_path = data_path
        self.mode = mode
        self.transform = transform
        
        if mode == "test":
            self.test_load()
        else : 
            self.load_dataset()

    def test_load(self):
        root = os.path.join(self.data_path, self.mode)
        print("root : ", root)
        self.data = glob.glob(root+"/*.png")
        
    def load_dataset(self):
        root = os.path.join(self.data_path, self.mode)
        print("root : ", root)
        self.data = ImageFolder(root=root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.mode == "test":
            img = Image.open(self.data[index]).convert('RGB')
            img = self.transform(img)
            return img
        else:
            img, label = self.data[index]
            img, label = self.transform(img), int(label)
            return img, label


class IMDataset(Dataset):
    def __init__(self, data_path, transform):
        super(IMDataset, self).__init__()
        self.data_path = data_path
        self.transform = transform
        self.load_dataset()

    def load_dataset(self):
        root = os.path.join(self.data_path)
        print("root : ", root)
        self.data = glob.glob(root + "/*/*.png")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = Image.open(self.data[index]).convert('RGB')
        img = self.transform(img)
        return img


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

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layer, reshape_transform):
        self.model = model
        self.gradients = None
        self.activations = None
        self.reshape_transform = reshape_transform
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = torch.squeeze(output)
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = torch.squeeze(grad_output[0])
    
    def clear_list(self):
        del self.gradients
        torch.cuda.empty_cache()
        self.gradients = None
    
    def buffer_clear(self):
        del self.gradients, self.activations
        torch.cuda.empty_cache()
        self.gradients = None
        self.activations = None
    def __call__(self, x):
        self.gradients = None
        self.activations = None       
        return self.model(x)

class GradCAM:
    def __init__(self, 
                 model, 
                 target_layer,
                 use_cuda=True,
                 reshape_transform=None):
        self.model = model
        self.target_layer = target_layer
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.reshape_transform = reshape_transform
        self.activations_and_grads = ActivationsAndGradients(self.model, 
            target_layer, reshape_transform)
    def forward(self, input_img):
        return self.model(input_img)
    
    def buffer_clear(self):
        self.activations_and_grads.buffer_clear()
    def __call__(self, input_tensor, label, expand):
        # input_tensor : b x c x h x w
        self.buffer_clear()
        self.model.eval()
        
        cam_stack=[]    
        for batch_idx in tqdm(range(input_tensor.shape[0]), desc='cam_calc', leave=False): # batch 개
            self.model.zero_grad()
            output = self.activations_and_grads(input_tensor[batch_idx].unsqueeze(0)) # 1 x c x h x w
            y_c = output[batch_idx, label[batch_idx]] #arg_max # h x w ; GAP over channel
            y_c.backward(retain_graph=True) 
            activations = self.activations_and_grads.activations
            grads = self.activations_and_grads.gradients
            self.buffer_clear()
            weights = torch.mean(grads, dim=(1, 2), keepdim=True)
            cam = torch.sum(weights * activations, dim=0)
            cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=expand.size()[2:], mode='bilinear', align_corners=True)
            min_v = torch.min(cam)
            range_v = torch.max(cam) - min_v
            if range_v > 0:
                cam = (cam - min_v) / range_v
            else:
                cam = torch.zeros(cam.size())
            cam_stack.append(cam.cuda())
            
            self.activations_and_grads.clear_list()
            del y_c, activations, grads, weights, cam, output
            torch.cuda.empty_cache()
        concated_cam = torch.cat(cam_stack, dim=0).squeeze() # b x 5 x h x w
        del cam_stack, input_tensor
        torch.cuda.empty_cache()
        self.model.train()
        self.buffer_clear()
        return concated_cam


class GCDataset(Dataset):
    def __init__(self, data_path, transform, model):
        super(GCDataset, self).__init__()
        self.data_path = data_path
        self.load_dataset()

        self.model = model

        self.to_tensor_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.resize_transform = transforms.Compose([
            transforms.Resize((64, 64))
        ])

    def load_dataset(self):
        root = os.path.join(self.data_path)
        print("root : ", root)
        self.data = glob.glob(root + "/*/*.png")

    def __len__(self):
        return len(self.data)

    def get_params(self, img):
        w_, h_ = img.size(2), img.size(3)
        xl = random.randint(0, w_ / 8)
        xr = 0
        while (((xr - xl) < (w_ * 7 / 8)) and (xr <= xl)):
            xr = random.randint(xl, w_)

        yl = random.randint(0, h_ / 8)
        yr = 0
        while (((yr - yl) < (h_ * 7 / 8)) and (yr <= yl)):
            yr = random.randint(yl, h_)

        return xl, yl, xr, yr


    def __getitem__(self, index):
        print("in get item")
        image = Image.open(self.data[index]).convert('RGB')

        print("image open")
        image = self.to_tensor_transform(image)
        image = image.unsqueeze(0)
        image = self.resize_transform(image)

        wk_image = self.color_augmentation(0.1, image)
        wk_image = wk_image.cuda()

        wk_label = self.model(wk_image)

        i, j, h, w = self.get_params(image)
        st_image = F.crop(image, i, j, h, w)
        st_image = self.color_augmentation(0.5, st_image)
        st_image = self.resize_transform(st_image)
        st_image = st_image.cuda()

        st_label = self.model(st_image)

        return wk_image, wk_label, st_image, st_label
