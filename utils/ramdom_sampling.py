import torch
import torch.nn as nn
import numpy as np
import random

def renormalize(vector, range_t : tuple):

    row = torch.Tensor(vector)
    r = torch.max(row) - torch.min(row)
    row_0_to_1 = (row - torch.min(row)) / r
    r2 = range_t[1] - range_t[0]
    row_normed = (row_0_to_1 * r2) + range_t[0]

    return row_normed.int().tolist()



def sampling(image, feature):
    # image : b x c x h x w

    img_size = image.size()[2:]
    h, w = img_size

    feature_size = feature.size()[2:]
    fh, fw = feature_size

    row = random.sample(range(0,fw),5)
    col = random.sample(range(0,fh),5)

    row_normed = renormalize(row, (0, w-1))
    col_normed = renormalize(col, (0, h-1))
    
    row_col_stack=[]
    row_col_stack_norm=[]

    for r, c in zip(row, col):
        row_col_stack.append((r,c))
    
    for r, c in zip(row_normed, col_normed):
        row_col_stack_norm.append((r,c))

    return row_col_stack, row_col_stack_norm

def sampling_calculated_cam_pixels(cam):
    # cam : b x 5 x h x w

    img_size = cam.size()[2:]
    h, w = img_size
    raw_value = int(h * w)

    row = random.sample(range(0,w),85)
    col = random.sample(range(0,h),85)

    row_col_stack=[]
    for r, c in zip(row, col):
        row_col_stack.append((r,c))

    random.shuffle(row)
    random.shuffle(col)
    for r, c in zip(row, col):
        row_col_stack.append((r,c))
    
    return row_col_stack

