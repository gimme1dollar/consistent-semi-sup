from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.multiprocessing
from torchvision import transforms, utils
from dataset.dataloader import LoadDataset
from tqdm import tqdm, tqdm_notebook
from os.path import join as pjn
import os.path, os, datetime, math, random, time
import numpy as np
import wandb, argparse
from efficientnet_pytorch.model import EfficientNet
from utils.losses import CEloss, total_loss
from utils.visualize import *

torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

data_path = pjn(os.getcwd(), "dataset", "DL20")

def init(train_batch, val_batch, test_batch):

    # default augmentation functions : http://incredible.ai/pytorch/2020/04/25/Pytorch-Image-Augmentation/ 
    # for more augmentation functions : https://github.com/aleju/imgaug

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    transform_test = transforms.Compose([

        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    train_dataset = LoadDataset(data_path = data_path, transform=transform_train , mode='train')
    val_dataset = LoadDataset(data_path = data_path, transform=transform_val , mode='valid')
    test_dataset = LoadDataset(data_path = data_path, transform=transform_test , mode='test')
                            
    train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=train_batch,
            num_workers=4, shuffle=True, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset, batch_size=train_batch,
            num_workers=4, shuffle=False, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=train_batch,
            num_workers=4, shuffle=False, pin_memory=True
    )

    return train_loader, val_loader, test_loader
    
class TrainManager(object):
    def __init__(
            self,
            model,
            optimizer,
            args,
            additional_cfg,
            train_loader,
            val_loader,
            test_loader,
            scaler=None,
            num_classes=None,
            ):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.args = args
        self.add_cfg = additional_cfg
        self.tbx_wrtr_dir = additional_cfg.get('tbx_wrtr_dir')
        self.scaler = scaler
        self.val_loader = val_loader
        self.num_classes = num_classes

    def validate(self, loader, device, topk=(1,3,5)):
        self.model.eval()
        total = 0
        maxk = max(topk)
        correct_1 = 0
        correct_3 = 0
        correct_5 = 0
        with torch.no_grad():
            for b_idx, (image, labels) in tqdm(enumerate(loader), desc="validation", leave=False):
                image = image.to(device)
                labels = labels.to(device)

                total += image.shape[0]

                with torch.cuda.amp.autocast():
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


        return (correct_1 / total) * 100, (correct_3 / total) * 100, (correct_5 / total) * 100

                
    def train(self):
        start = time.time()
        epoch = 0
        print("  training data(considered batch size): ", len(self.train_loader))
        print("  Progress bar for training epochs:")
        end_epoch = self.args.start_epoch + self.args.num_epochs

        for epoch in tqdm(range(self.args.start_epoch, end_epoch), desc='epochs', leave=False):
            self.model.train()
            for idx, param_group in enumerate(self.optimizer.param_groups):
                avg_lr = param_group['lr']
                wandb.log({str(idx)+"_lr": math.log10(avg_lr), 'epoch': epoch})

            for t_idx, (image, target) in tqdm(enumerate(self.train_loader), desc='batch_iter', leave=False):
                self.optimizer.zero_grad()
                losses_list = []
                image = image.to(self.add_cfg['device'])
                target = target.to(self.add_cfg['device'])
                with torch.cuda.amp.autocast():
                    outputs = self.model(image)
                    celoss = CEloss(outputs, target)
                    losses_list.append(celoss)

                    wandb.log({"training/celoss" : celoss})

                t_loss = total_loss(losses_list)
                self.scaler.scale(t_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if t_idx % 50 == 0:
                    visualize_rescale_image(imagenet_mean, imagenet_std, image, "training/input_image")
                    
            top1_acc, top3_acc, top5_acc = self.validate(self.val_loader, self.add_cfg['device'])
            wandb.log({"validation/top1_acc" : top1_acc, "validation/top3_acc" : top3_acc, "validation/top5_acc" : top5_acc})

            self.adjust_learning_rate(epoch)
            self.save_ckpt(epoch)
            
        end = time.time()
               
        print("Total training time : ", str(datetime.timedelta(seconds=(int(end)-int(start)))))
        print("Finish.")

    def adjust_learning_rate(self, epoch):
        # update optimizer's learning rate
        for param_group in self.optimizer.param_groups:
            prev_lr = param_group['lr']
            param_group['lr'] = prev_lr * self.args.lr_anneal_rate

    def save_ckpt(self, epoch):
        if epoch % self.args.save_ckpt == 0:

            nm = f'epoch_{epoch:04d}.pth'

            if not os.path.isdir(pjn('checkpoints', self.tbx_wrtr_dir)):
                os.mkdir(pjn('checkpoints', self.tbx_wrtr_dir))

            fpath=pjn('checkpoints', self.tbx_wrtr_dir, nm)

            d = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
            torch.save(d, fpath)

def main(args):
    # for deterministic training, enable all below options.
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed) # if use multi-GPU
    # torch.backends.cudnn.deterministic = False
    # np.random.seed(args.seed)
    # random.seed(args.seed)

    wandb.init(project="DL20")
    orig_cwd = os.getcwd()
    
    # bring effi model from this : https://github.com/lukemelas/EfficientNet-PyTorch
    model = EfficientNet.from_pretrained('efficientnet-b4')
    wandb.watch(model)

    additional_cfg = {'device': None}
    additional_cfg['device'] = torch.device('cuda')

    model = model.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=0.9)

    scaler = torch.cuda.amp.GradScaler() 

    if args.pretrained_ckpt:
        print(f"  Using pretrained model only and its checkpoint "
              f"'{args.pretrained_ckpt}'")
        loaded_struct = torch.load(pjn(orig_cwd, args.pretrained_ckpt))
        model.load_state_dict(loaded_struct['model_state_dict'], strict=True)
        print("load optimizer's params")
        loaded_struct = torch.load(pjn(orig_cwd, args.pretrained_ckpt))
        optimizer.load_state_dict(loaded_struct['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
    now = datetime.datetime.now()
    additional_cfg['tbx_wrtr_dir'] = os.getcwd() + "/checkpoints/" + str(now.strftime('%Y-%m-%d-%H-%M-%S'))

    train_loader, val_loader, test_loader = init(
        args.batch_size_train, args.batch_size_val, args.batch_size_test
    )

    trainer = TrainManager(
        model,
        optimizer,
        args,
        additional_cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scaler=scaler,
        num_classes=20
    )

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default='auto',
                        help='Name of the experiment (default: auto)')
    parser.add_argument('--pretrained-ckpt', type=str, default=None,
                        help='Load pretrained weight, write path to weight (default: None)')
    
    parser.add_argument('--batch-size-train', type=int, default=16,    
                        help='Batch size for train data (default: 16)')
    parser.add_argument('--batch-size-val', type=int, default=16,
                        help='Batch size for val data (default: 16)')
    parser.add_argument('--batch-size-test', type=int, default=16,
                        help='Batch size for test data (default: 16)')

    parser.add_argument('--save-ckpt', type=int, default=1,
                        help='number of epoch save current weight? (default: 5)')

    parser.add_argument('--start-epoch', type=int, default=0,
                        help='start epoch (default: 0)')
    parser.add_argument('--num-epochs', type=int, default=30,
                        help='end epoch (default: 30)')

    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay for SGD (default: 0.0005)')
    parser.add_argument('--lr-anneal-rate', type=float, default=0.95,
                        help='Annealing rate (default: 0.95)')
    

    
    args = parser.parse_args()
    main(args)