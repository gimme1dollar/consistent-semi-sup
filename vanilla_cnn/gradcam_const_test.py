import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import glob
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from os import path

dl20_data_root = '../dataset/DL20'
imagenet_data_root = '../dataset/ImageNet/imagenet64/valid_64x64/valid_64x64'
pretrained_model_root = '../model/simpleCNN_0.pth'
print(path.exists(dl20_data_root))
print(path.exists(imagenet_data_root))
print(path.exists(pretrained_model_root))


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=20):
        super(SimpleCNN, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        # Convolution Feature Extraction Part
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(128 * 16 * 16, 20)

    def forward(self, x):
        # Convolution Feature Extraction Part
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)

        # Fully Connected Classifier Part
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

model = SimpleCNN(num_classes=20)
model.load_state_dict(torch.load(pretrained_model_root))

# Random input
x = torch.randn((1, 3, 64, 64))
out = model(x)
print("Output tensor shape is :", out.shape)

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

        self.to_PIL_transform = transforms.Compose([
            transforms.ToPILImage()
        ])
        self.to_tensor_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.resize_transform = transforms.Compose([
            transforms.Resize((64, 64))
        ])

    def load_dataset(self):
        root = os.path.join(self.data_path)
        print("root : ", root)
        self.data = glob.glob(root+"/*.png")

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
        #fig = plt.figure(figsize=(15, 5))
        #print("in get item")

        image = Image.open(self.data[index]).convert('RGB')
        image = self.to_tensor_transform(image)
        image = self.to_PIL_transform(image)
        image = self.resize_transform(image)
        image = self.to_tensor_transform(image)
        image = image.unsqueeze(0)
        #ax1_img = image.permute(0, 2, 3, 1).numpy()
        #ax1 = fig.add_subplot(1, 6, 1)
        #ax1.imshow(ax1_img[0])
        #print("image ok")

        i, j, h, w = self.get_params(image)
        image = image.squeeze(0)
        image = self.to_PIL_transform(image)
        st_image = F.crop(image, i, j, h, w)
        st_image = self.color_augmentation(0.5, st_image)
        st_image = self.resize_transform(st_image)
        st_image = self.to_tensor_transform(st_image)
        st_image = st_image.unsqueeze(0)
        #ax4_img = st_image.permute(0, 2, 3, 1).numpy()
        #ax4 = fig.add_subplot(1, 6, 5)
        #ax4.imshow(ax4_img[0])
        #print("st_image ok")
        st_label = self.model(st_image)

        wk_image = self.color_augmentation(0.1, image)
        wk_image = self.to_tensor_transform(wk_image)
        wk_image = wk_image.unsqueeze(0)
        #ax2_img = wk_image.permute(0, 2, 3, 1).numpy()
        #ax2 = fig.add_subplot(1, 6, 2)
        #ax2.imshow(ax2_img[0])
        #print("wk_image ok")
        wk_label = self.model(wk_image)

        #ax3 = fig.add_subplot(1, 6, 3)
        self.model.bn2.register_forward_hook(self.save_outputs_hook())
        wk_gradcam = self.get_grad_cam(wk_image)

        #img_np = wk_image.permute(0, 2, 3, 1).numpy()
        #if len(img_np.shape) > 3:
        #    img_np = img_np[0]
        #ax3.imshow(img_np)

        #wk_gradcam = wk_gradcam.detach()
        #wk_gradcam = wk_gradcam.permute(0, 2, 3, 1)
        #wk_gradcam = wk_gradcam.squeeze(0).numpy()
        wk_gradcam = wk_gradcam.squeeze(0)
        #ax3.imshow(wk_gradcam, cmap='jet', alpha=0.6)
        #print("wk_gradcam ok")

        gt_gradcam = self.to_PIL_transform(wk_gradcam)
        gt_gradcam = F.crop(gt_gradcam, i, j, h, w)
        gt_gradcam = self.resize_transform(gt_gradcam)
        gt_gradcam = self.to_tensor_transform(gt_gradcam)
        #gt_gradcam = gt_gradcam.permute(1, 2, 0).detach().numpy()
        #ax6 = fig.add_subplot(1, 6, 4)
        #ax6.imshow(gt_gradcam)
        #ax6.imshow(gt_gradcam, cmap='jet', alpha=0.6)
        #print("gt_gradcam ok")

        st_gradcam = self.get_grad_cam(st_image)
        st_gradcam = st_gradcam.squeeze(0)
        #print("st_gradcam ok")

        #img_np = st_image.permute(0, 2, 3, 1).numpy()
        #if len(img_np.shape) > 3:
        #    img_np = img_np[0]
        #st_gradcam = st_gradcam.squeeze().detach().numpy()
        #ax5 = fig.add_subplot(1, 6, 6)
        #ax5.imshow(img_np)
        #ax5.imshow(st_gradcam, cmap='jet', alpha=0.6)

        #plt.show()

        return wk_image, wk_label, wk_gradcam, st_image, st_label, st_gradcam, gt_gradcam

batch_size = 32
imagenet_train_dataset = GCDataset(imagenet_data_root, model)
imagenet_train_loader = DataLoader(imagenet_train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)

model.train()
learning_rate = 5 * 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

train_loss, train_accu = [], []
epoch = 1
for e in range(epoch):
  train_iter = 0
  for wk_image, wk_label, wk_gradcam, st_image, st_label, st_gradcam, gt_gradcam in imagenet_train_loader:
    loss_label = nn.functional.mse_loss(wk_label, st_label, reduction='sum') / wk_image.size(0)
    loss_gradcam = nn.functional.mse_loss(st_gradcam, gt_gradcam, reduction='sum') / wk_image.size(0)

    optimizer.zero_grad()
    loss = loss_label
    if math.isnan(loss_gradcam):
        loss += loss_gradcam
    loss.backward(retain_graph=True)
    optimizer.step()

    prediction = torch.argmax(st_label, dim=-1)
    accuracy = prediction.eq(wk_label.data).sum()/batch_size*100
    train_accu.append(accuracy)
    if (e % 1 == 0) and (train_iter % 100 == 0):
        print(f'Epoch: {e}\tTrain Step: {train_iter}\tLoss: {loss:.3f}\tAccuracy: {accuracy:.3f}')
    train_iter += 1
  torch.save(model.state_dict(), f'./gradcam_const_{e}.pth', _use_new_zipfile_serialization=False)