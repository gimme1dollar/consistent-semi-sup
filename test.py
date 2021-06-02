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
from utils.losses import *
from itertools import cycle
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

    val_dataset = LoadDataset(data_path = data_path, transform=transform_test , mode='valid')
    test_dataset = LoadDataset(data_path = data_path, transform=transform_test , mode='test')
    test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=test_batch,
            num_workers=0, shuffle=False, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset, batch_size=1,
            num_workers=0, shuffle=False, pin_memory=True
    )
    return val_loader, test_loader
    
class TrainManager(object):
    def __init__(
            self,
            model,
            args,
            val_loader,
            test_loader,
            ):
        self.args = args
        self.model = model
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.dict = {}
        self.mapping={}
        self.upscaler = torch.nn.Upsample(scale_factor=args.upscale_factor, mode='bilinear', align_corners=True)

        for i in range(20):
            self.dict[i] = []
        
        for i in range(20):
            self.mapping[i] = []

    def test(self):

        self.model.eval()
        
        with open(self.args.exp_name + "_test.csv", "w") as f:
            f.write("Id"+","+"Category" + '\n')
            with torch.no_grad():
                for b_idx, (image, path) in tqdm(enumerate(self.test_loader), desc="test", leave=False):
                    path = path[0].split("/")[-1]
                    path = path.split('.')[0]
                    image = image.cuda()
                    image = self.upscaler(image)
                    outputs = self.model(image)
                    our_label = outputs.argmax().item()
                    real_label = self.mapping[our_label]

                    f.write(str(path) + ","+ str(real_label) + '\n')
    
    def val(self):
        self.model.eval()

        correct_1 = 0
        correct_3 = 0
        correct_5 = 0
        total = 0
        topk=(1,3,5)
        maxk = max(topk)

        with torch.no_grad():
            for b_idx, (image, labels, label_fold) in tqdm(enumerate(self.val_loader), desc="validation", leave=False, total=len(self.val_loader)):
                image = self.upscaler(image)
                image = image.cuda()
                labels = labels.cuda()
                label_fold = label_fold.cuda()

                self.dict[label_fold.item()].append(labels.item())

                total += image.shape[0]
                
                outputs = self.model(image) # b x 1

                _, pred = outputs.topk(maxk, 1, True, True)
                pred = pred.t()
                correct = (pred == labels.unsqueeze(dim=0)).expand_as(pred)

                for k in topk:
                    if k == 1:
                        correct_k = correct[:1].reshape(-1).float().sum(0, keepdim=True)
                        correct_1 += correct_k.item()
                    elif k == 3:
                        correct_k = correct[:3].reshape(-1).float().sum(0, keepdim=True)
                        correct_3 += correct_k.item()
                    elif k == 5:
                        correct_k = correct[:5].reshape(-1).float().sum(0, keepdim=True)
                        correct_5 += correct_k.item()
                    else:
                        raise NotImplementedError("Invalid top-k num")
                

            model_top1, model_top3, model_top5 = (correct_1 / total) * 100, (correct_3 / total) * 100, (correct_5 / total) * 100
            print("top1 / top3 / top5 : ", model_top1, model_top3, model_top5)

            for key, value in self.dict.items():
                self.dict[key]=list(set(value))[0]
                self.mapping[list(set(value))[0]] = key


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
        model.load_state_dict(loaded_struct['second_student_state_dict'], strict=True)
        
    val_loader, test_loader = init(
        args.batch_size_test
    )

    trainer = TrainManager(
        model,
        args,
        val_loader,
        test_loader
    )

    trainer.val()
    trainer.test()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default='auto',
                        help='Name of the experiment (default: auto)')
    parser.add_argument('--pretrained-ckpt', type=str, default=None,
                        help='Load pretrained weight, write path to weight (default: None)')
    parser.add_argument('--batch-size-test', type=int, default=1,
                        help='Batch size for test data (default: 128)')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--upscale-factor', type=int, default=8,
                        help='upscale factor for bilinear upsampling. It is highly recommended to set the upscaling factor used in training, otherwise, the score would become lower')
    args = parser.parse_args()
    main(args)
