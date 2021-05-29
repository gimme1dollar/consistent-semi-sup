# reference git repository: https://github.com/sthalles/SimCLR.git

import torch.nn as nn
import torchvision.models as models

from efficientnet_pytorch.model import EfficientNet

class EffiNetSimCLR(nn.Module):

    def __init__(self, out_dim, pretrained = False):
        super(EffiNetSimCLR, self).__init__()
        # self.effinet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
        #                     "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}
        # self.effinet_dict = {"effinet": EfficientNet(pretrained=False, num_classes=out_dim)}
        if pretrained:
            self.backbone = EfficientNet.from_pretrained('efficientnet-b4', image_size = 64, num_classes=out_dim) # self._get_basemodel(base_model)
        else:
            self.backbone = EfficientNet.from_name('efficientnet-b4', image_size = 64, num_classes=out_dim) # self._get_basemodel(base_model)
        dim_mlp = self.backbone._fc.in_features

        # add mlp projection head
        self.backbone._fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone._fc)

    # def _get_basemodel(self, model_name):
    #     try:
    #         model = self.effinet_dict[model_name]
    #     except KeyError:
    #         raise InvalidBackboneError(
    #             "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
    #     else:
    #         return model

    def forward(self, x):
        return self.backbone(x)

