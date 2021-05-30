import torch
import torch.nn as nn
from efficientnet_pytorch.model import EfficientNet

class compound(nn.Module):
    def __init__(self):
        super(compound, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=20).cuda()
        self.teacher = EfficientNet.from_pretrained('efficientnet-b4', num_classes=20).cuda()

    def get_param(self):
        return self.model.parameters()
    
    def get_param_t(self):
        return self.teacher.parameters()

    def forward(self, x, is_t=False):
        if is_t is False:
            return self.model(x)
        else:
            return self.teacher(x)
