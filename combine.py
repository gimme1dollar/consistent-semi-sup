from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from efficientnet_pytorch import EfficientNet

def make_layers_vgg(cfg, in_ch=3, use_batch_norm=False):
    """
    Code borrowed from torchvision/models/vgg.py
    """
    layers = []

    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv2d = nn.Conv2d(in_ch, v, kernel_size=3, padding=1)
            if use_batch_norm:
                layers.extend(
                    #[conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                    [conv2d, nn.InstanceNorm2d(v), nn.ReLU(inplace=True)]
                )
            else:
                layers.extend([conv2d, nn.ReLU(inplace=True)])
            in_ch = v

    return nn.Sequential(*layers)

class vgg16_cnn(nn.Module):
    def __init__(self):
        super(vgg16_cnn, self).__init__()
        self.conv1_features = make_layers_vgg([64, 64, 'M'], in_ch=3, use_batch_norm=True)
        self.conv2_features = make_layers_vgg([128, 128, 'M'], in_ch=64, use_batch_norm=True)
        self.conv3_features = make_layers_vgg([256, 256, 256, 'M'], in_ch=128, use_batch_norm=True)
        self.conv4_features = make_layers_vgg([512, 512, 512, 'M'], in_ch=256, use_batch_norm=True)
        self.conv5_features = make_layers_vgg([512, 512, 512, 'M'], in_ch=512, use_batch_norm=True)

        pretr_dict = torchvision.models.vgg16(pretrained=True).state_dict()
        this_net_dict = self.state_dict()
        this_net_keys = list(this_net_dict.keys())
        for i, (pretr_key, pretr_tensor_val) in enumerate(pretr_dict.items()):
            # pretrained vgg16 keys start with 'features' or with 'classifier'
            if 'features' in pretr_key:
                this_net_tensor_val = this_net_dict[this_net_keys[i]]
                assert this_net_tensor_val.shape == pretr_tensor_val.shape
                this_net_tensor_val.data = pretr_tensor_val.data.clone()
                #print(pretr_key, pretr_tensor_val.shape)
            else:
                break
        self.load_state_dict(this_net_dict)

    def forward(self, x):
        x = self.conv1_features(x)
        x = self.conv2_features(x)
        x = self.conv3_features(x)
        x = self.conv4_features(x)
        x = self.conv5_features(x)
        return x.flatten(1) # b x 512 x 2 x 2
