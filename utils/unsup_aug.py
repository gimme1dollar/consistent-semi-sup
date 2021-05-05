import torch
import torch.nn as nn
import sys
import math
from PIL import Image, ImageOps, ImageEnhance, PILLOW_VERSION
import numpy as np
import collections
import random, math
import torchvision.transforms as trans
import torchvision.transforms.functional as F
import copy

def get_tensor_image_size(img):
    if img.dim() == 2:
        h, w = img.size()
    elif img.dim() == 4:
        h = img.size()[2]
        w = img.size()[3]

    return h, w

class ZeroPad(object):
    def __init__(self, size):
        self.h, self.w = size

    @staticmethod
    def zero_pad(image, h, w):
        oh, ow = get_tensor_image_size(image)
        pad_h = h - oh if oh < h else 0
        pad_w = w - ow if ow < w else 0
        image = F.pad(image, (0, 0, pad_w, pad_h), fill=0)

        return image

    def __call__(self, image, ):
        return self.zero_pad(image, self.h, self.w)

def get_crosx(x1,  y1,  x2,  y2, x3,  y3,  x4,  y4):

    x5 = max([x1, x3])
    y5 = max([y1, y3])
  
    x6 = min([x2, x4])
    y6 = min([y2, y4])
  
    # x5 y5 bottom-left
    if (x5 > x6 or y5 > y6): 
        return None
    # x6 y6 top-right
  
    return [(x5, x6), (y5, y6)]

def RandomCropCrossX(image, size):
    def get_params(img, output_size):
        h, w = get_tensor_image_size(img)
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    # Pad if needed
    ih, iw = get_tensor_image_size(image)
    if ih < size[0] or iw < size[1]:
        image = torch.nn.functional.interpolate(image, size=(max(size[0], ih),max(size[1], iw)), mode='bilinear', align_corners=True)
        
    i, j, h, w = get_params(image, size)
    image = F.crop(image, i, j, h, w)
    return image, i, j, i+h, j+w

def random_cropping_crossx(img):
    
    # shape format : b x c x h x w
    width = img.shape[3]
    height = img.shape[2]

    x_right = random.sample(list(range(round(width/2), round(width/1.5))), 2)
    y_up = random.sample(list(range(round(height/2), round(height/1.5))), 2)

    x_1, y_1 = x_right[0], y_up[0]
    x_2, y_2 = x_right[1], y_up[1]

    origin_img1, y_bl, x_bl, y_tr, x_tr = RandomCropCrossX(img, (y_1, x_1))
    img2, y_bl2, x_bl2, y_tr2, x_tr2 = RandomCropCrossX(img, (y_2, x_2))


    overlap_coord = get_crosx(x_bl, y_bl, x_tr, y_tr, x_bl2, y_bl2, x_tr2, y_tr2)

    if overlap_coord is not None:       
        x_ov = overlap_coord[0]
        y_ov = overlap_coord[1]
        origin_img1 = copy.deepcopy(img)
        img = F.erase(img, y_ov[0], x_ov[0], y_ov[1], x_ov[1], v=0)
        img1 = F.crop(img, y_bl, x_bl, y_tr, x_tr)
    else:
        img1 = origin_img1
        x_ov, y_ov = None, None

    img1 = torch.nn.functional.interpolate(img1, size=img2.shape[2:], mode='bilinear', align_corners=True)
    origin_img1 = torch.nn.functional.interpolate(origin_img1, size=img2.shape[2:], mode='bilinear', align_corners=True)
    return img1, img2, origin_img1, x_ov, y_ov
    
def five_crop(img):
    h, w = get_tensor_image_size(img)
    tr = trans.FiveCrop((round(h/2), round(w/2)))
    cr_img = list(tr(img))
    t_idx = np.random.randint(0, 4)
    target = cr_img[t_idx]
    del cr_img[t_idx]
    return target, cr_img


class perturbation(object):
    def __init__(self):
        # self.hori_flip = trans.RandomHorizontalFlip(p=0.5)
        # self.verti_flip = trans.RandomVerticalFlip(p=0.5)
        # #self.perspect = trans.RandomPerspective(p=1)
        # #self.rotate = trans.RandomRotation(90, expand=True)
        self.dropout = nn.Dropout(0.1)
        self.color_jitter = trans.ColorJitter(
            brightness=(0.9, 1.1), 
            contrast=(0.9, 1.1), 
            saturation=(0.9, 1.1), 
            hue=(-0.001, 0.001)
        )


    def __call__(self, img): # 0 for img1, 1 for img2
        #img = self.color_jitter(img)
        img = self.dropout(img)
        #img = self.hori_flip(img)
        #img = self.verti_flip(img)
        #img = self.rotate(img)
        #img = self.perspect(img)
    
        return img
    
