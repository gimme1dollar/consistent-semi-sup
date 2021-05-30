import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.resnet_simclr import ResNetSimCLR
from models.effinet_simclr import EffiNetSimCLR
from z_semi_kam_simclr import Semi_KamSimCLR2

# kamse add
from efficientnet_pytorch.model import EfficientNet
from torchvision import transforms, utils
from os.path import join as pjn
from datasets.dataloader import LoadDataset
from data_aug.view_generator import ContrastiveLearningViewGenerator
from data_aug.gaussian_blur import GaussianBlur

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('-dataset-name', default='cifar10',
                    help='dataset name', choices=['stl10', 'cifar10'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=18, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0008, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=3e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')


torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

data_path = pjn("/Jarvis/workspace/jwkam", "DL20") #pjn(os.getcwd(), "dataset", "DL20")
imagenet_data_path = pjn("/Jarvis/workspace/jwkam", "DL20") # pjn(os.getcwd(), "dataset", "ImageNet", "ILSVRC", "Data", "CLS-LOC")

def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

def init(batch_size ): #, test_batch, imagenet_batch):

    # default augmentation functions : http://incredible.ai/pytorch/2020/04/25/Pytorch-Image-Augmentation/ 
    # for more augmentation functions : https://github.com/aleju/imgaug

    # transform_train = transforms.Compose([
    #     #transforms.ToPILImage(),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.RandomVerticalFlip(p=0.5),
    #     transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    #     transforms.RandomGrayscale(p=0.2),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    # ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    # ])

    # transform_imagenet = transforms.Compose([
    #     transforms.RandomResizedCrop(224),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    # ])

    transform_simclr = ContrastiveLearningViewGenerator(get_simclr_pipeline_transform(96),
                                                        n_views=2)

    train_dataset = LoadDataset(data_path = data_path, transform=transform_val , mode='train')
    val_dataset = LoadDataset(data_path = data_path, transform=transform_val , mode='valid')
    # test_dataset = LoadDataset(data_path = data_path, transform=transform_test , mode='test')
    # imagenet_dataset = LoadDataset(data_path = imagenet_data_path, transform=transform_imagenet, mode='train')

    train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size,
            num_workers=10, shuffle=True, pin_memory=True, drop_last=False
    )

    val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset, batch_size=2*batch_size,
            num_workers=10, shuffle=True, pin_memory=True, drop_last=False
    )

    # test_loader = torch.utils.data.DataLoader(
    #         dataset=test_dataset, batch_size=test_batch,
    #         num_workers=4, shuffle=False, pin_memory=True, drop_last=True
    # )

    # imagenet_loader = torch.utils.data.DataLoader(
    #         dataset=imagenet_dataset, batch_size=imagenet_batch,
    #         num_workers=4, shuffle=True, pin_memory=True
    # )

    return train_loader, val_loader #, test_loader, imagenet_loader
    


def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    # dataset = ContrastiveLearningDataset(args.data)

    # train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=True,
    #     num_workers=args.workers, pin_memory=True, drop_last=True)

    
    train_loader, val_loader= init(args.batch_size)

    # model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)
    # model = EfficientNet.from_pretrained('efficientnet-b4', image_size = 64, num_classes=20)
    model = EfficientNet.from_name('efficientnet-b4', image_size = 64, num_classes=20)

    checkpoint = torch.load('./runs/May30_17-15-19_cglab/Effi_checkpoint_0200.pth.tar', map_location=args.device)
    state_dict = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        if k.startswith('backbone.'):
            if k.startswith('backbone') and not k.startswith('backbone._fc'):
                # remove prefix
                state_dict[k[len("backbone."):]] = state_dict[k]
        del state_dict[k]

    log = model.load_state_dict(state_dict, strict=False)
    assert log.missing_keys == ['_fc.weight', '_fc.bias']

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['_fc.weight', '_fc.bias']:
            param.requires_grad = False
        else:
            print(name)
            
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
    #                                                        last_epoch=-1)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        simclr = Semi_KamSimCLR2(model=model, optimizer=optimizer, args=args)
        simclr.eval(train_loader,val_loader)


if __name__ == "__main__":
    main()
