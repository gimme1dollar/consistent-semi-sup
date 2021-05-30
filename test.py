from efficientnet_pytorch.model import EfficientNet
from torch.utils.data import Dataset, DataLoader
import torch, torchvision
import torch.nn as nn
import torch.multiprocessing
from torchvision import transforms, utils
from dataset.dataloader import LoadDataset
from tqdm import tqdm, tqdm_notebook
from os.path import join as pjn
import os.path, os, datetime, math, random, time
import numpy as np
import wandb, argparse
from combine import vgg16_cnn
from utils.losses import *
from itertools import cycle
from adabound import AdaBound
import warnings

torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings(action='ignore')

torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

data_path = pjn(os.getcwd(), "dataset", "DL20")
imagenet_data_path = pjn(os.getcwd(), "dataset", "ImageNet", "ILSVRC", "Data", "CLS-LOC")

def init(test_batch):

    # default augmentation functions : http://incredible.ai/pytorch/2020/04/25/Pytorch-Image-Augmentation/ 
    # for more augmentation functions : https://github.com/aleju/imgaug
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    test_dataset = LoadDataset(data_path = data_path, transform=transform_test , mode='test')
    test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=test_batch,
            num_workers=4, shuffle=False, pin_memory=True
    )
    return test_loader
    
class TrainManager(object):
    def __init__(
            self,
            model,
            args,
            test_loader,
            ):
        self.args = args
        self.model = model
        self.test_loader = test_loader
        

    def validate(self):
        upscale_layer = torch.nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.model.eval()
        with open(self.args.exp_name + "_test.csv", "w") as f:
            f.write("Id"+","+"Category" + '\n')
            with torch.no_grad():
                for b_idx, (image, path) in tqdm(enumerate(self.test_loader), desc="test", leave=False):
                    path = path[0].split("/")[-1]
                    path = path.split('.')[0]
                    image = image.cuda()
                    image = upscale_layer(image)
                    outputs = self.model(image)
                    f.write(str(path) + ","+ str(outputs.argmax().item()) + '\n')

def main(args):
    # for deterministic training, enable all below options.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    random.seed(args.seed)

    orig_cwd = os.getcwd()
    
    # bring effi model from this : https://github.com/lukemelas/EfficientNet-PyTorch
    model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=20).cuda()
    
    if args.pretrained_ckpt:
        print(f"  Using pretrained model only and its checkpoint "
              f"'{args.pretrained_ckpt}'")
        loaded_struct = torch.load(pjn(orig_cwd, args.pretrained_ckpt))
        model.load_state_dict(loaded_struct['model_state_dict'], strict=True)
        
    test_loader = init(
        args.batch_size_test
    )

    trainer = TrainManager(
        model,
        args,
        test_loader,
    )

    trainer.validate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default='auto',
                        help='Name of the experiment (default: auto)')
    parser.add_argument('--pretrained-ckpt', type=str, default=None,
                        help='Load pretrained weight, write path to weight (default: None)')
    parser.add_argument('--batch-size-test', type=int, default=1,
                        help='Batch size for test data (default: 128)')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    main(args)