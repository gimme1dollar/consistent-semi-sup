from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.multiprocessing
from torchvision import transforms, utils
from dataset.dataloader import LoadDataset, IMDataset
from tqdm import tqdm, tqdm_notebook
from os.path import join as pjn
import os.path, os, datetime, math, random, time
import numpy as np
import wandb, argparse
from efficientnet_pytorch.model import EfficientNet
from utils.losses import CEloss, total_loss, MSEloss, SSIM
from utils.visualize import *
from itertools import cycle
import torchvision.transforms.functional as F
from typing import Callable
import math

torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

data_path = pjn(os.getcwd(), "dataset", "DL20")
imagenet_data_path = pjn(os.getcwd(), "dataset", "ImageNet")

def init(train_batch, val_batch, test_batch, imagenet_batch):

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

    transform_imagenet = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = LoadDataset(data_path = data_path, transform=transform_train , mode='train')
    val_dataset = LoadDataset(data_path = data_path, transform=transform_val , mode='valid')
    test_dataset = LoadDataset(data_path = data_path, transform=transform_test , mode='test')
    imagenet_dataset = IMDataset(data_path = imagenet_data_path, transform=transform_imagenet)

    train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=train_batch,
            num_workers=4, shuffle=True, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset, batch_size=val_batch,
            num_workers=4, shuffle=False, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=test_batch,
            num_workers=4, shuffle=False, pin_memory=True
    )

    imagenet_loader = torch.utils.data.DataLoader(
            dataset=imagenet_dataset, batch_size=imagenet_batch,
            num_workers=4, shuffle=True, pin_memory=True
    )

    return train_loader, val_loader, test_loader, imagenet_loader
    
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
            imagenet_loader,
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
        self.imagenet_loader = imagenet_loader
        self.num_classes = num_classes
        

        self.save_feat=[]
        self.save_grad=[]
        for idx, module in enumerate(self.model.modules()):
            #if idx > 200:
            #    print(idx, module)
            if idx == 479:
                module.register_forward_hook(self.save_outputs_hook())

        self.to_tensor_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.resize_512_transform = transforms.Compose([
            transforms.Resize((512, 512))
        ])
        self.resize_256_transform = transforms.Compose([
            transforms.Resize((256, 256))
        ])
        self.resize_64_transform = transforms.Compose([
            transforms.Resize((64, 64))
        ])


    def save_outputs_hook(self) -> Callable:
        def fn(_, __, output):
            #print(output.size())
            self.save_feat.append(output)
        return fn

    def save_grad_hook(self) -> Callable:
        def fn(grad):
            self.save_grad.append(grad)
        return fn

    def color_augmentation(self, i, img):
        color_transform = transforms.Compose([
            transforms.ColorJitter(i, i, i, i)
        ])

        return color_transform(img)

    def get_crop_params(self, img):
        w_, h_ = img.size(2), img.size(3)
        xl = random.randint(0, w_ / 8)
        xr = 0
        while (((xr - xl) < (w_ * 7 / 8)) and (xr <= xl)):
            xr = random.randint(xl, w_)

        yl = random.randint(0, h_ / 8)
        yr = 0
        while (((yr - yl) < (h_ * 7 / 8)) and (yr <= yl)):
            yr = random.randint(yl, h_)

        return xl, yl, xr, yr

    def validate(self, loader, device, topk=(1,3,5)):
        self.model.eval()
        total = 0
        maxk = max(topk)
        correct_1 = 0
        correct_3 = 0
        correct_5 = 0

        upscale_layer = torch.nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        with torch.no_grad():
            for b_idx, (image, labels) in tqdm(enumerate(loader), desc="validation", leave=False):
                image = image.to(device)
                image = upscale_layer(image)
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
            
            del image

        return (correct_1 / total) * 100, (correct_3 / total) * 100, (correct_5 / total) * 100

    def get_grad_cam(self, image):
        self.model.zero_grad()
        self.model.eval()
        self.save_feat=[]
        #image = image.unsqueeze(0)
        s = self.model(image)[0]

        self.save_grad=[]
        self.save_feat[0].register_hook(self.save_grad_hook())
        #print(f"save_feat size: {np.shape(self.save_feat[0])}")

        y = torch.argmax(s).item()
        s_y = s[y]
        s_y.backward(retain_graph=True)
        #print(f"save_grad size: {np.shape(self.save_grad[0][0])}")
        
        gap_layer = torch.nn.AdaptiveAvgPool2d(1)
        alpha = gap_layer(self.save_grad[0][0])
        #print(f"alpha size: {alpha.size()}")
        A = self.save_feat[0]
        A = A.squeeze()
        #print(f"A size: {A.size()}")

        weighted_sum = torch.sum(alpha*A, dim=0)
        relu_layer = torch.nn.ReLU()
        grad_CAM = relu_layer(weighted_sum)
        grad_CAM = grad_CAM.unsqueeze(0)
        grad_CAM = grad_CAM.unsqueeze(0)
        #print(f"grad_CAM size: {grad_CAM.size()}")

        #sf = image.shape[-1]/grad_CAM.shape[-1]
        sf = 32
        #return 1
        upscale_layer = torch.nn.Upsample(scale_factor=sf, mode='bilinear', align_corners=True)
        grad_CAM = upscale_layer(grad_CAM)
        grad_CAM = grad_CAM/torch.max(grad_CAM)
        self.model.train()
        return grad_CAM.squeeze()
        
    def train(self):
        start = time.time()
        epoch = 0
        iter_per_epoch = len(self.train_loader)
        print("  iteration per epoch(considered batch size): ", iter_per_epoch)
        print("  Progress bar for training epochs:")
        end_epoch = self.args.start_epoch + self.args.num_epochs

        ## gradcam
        p_cutoff = 0.90
        dataloader = iter(zip(cycle(self.train_loader), cycle(self.imagenet_loader)))
        upscale_layer = torch.nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        for epoch in tqdm(range(self.args.start_epoch, end_epoch), desc='epochs', leave=False):

            for idx, param_group in enumerate(self.optimizer.param_groups):
                avg_lr = param_group['lr']
                wandb.log({str(idx)+"_lr": math.log10(avg_lr), 'epoch': epoch})

            self.model.train()
            for t_idx in tqdm(range(0, iter_per_epoch), desc='batch_iter', leave=False):
                
                sup, semisup = next(dataloader)

                image, target = sup
                image_ul = semisup

                image = image.cuda()
                image = upscale_layer(image)
                target = target.cuda()


                self.optimizer.zero_grad()
                losses_list = []
                with torch.cuda.amp.autocast():
                    outputs = self.model(image)
                    celoss = CEloss(outputs, target)
                    losses_list.append(celoss)
                    wandb.log({"training/celoss" : celoss})
                if t_idx % 50 == 0:
                    visualize_rescale_image(imagenet_mean, imagenet_std, image, "input_image/DL20")
                
                image_ul = image_ul.cuda()
                image_ul = upscale_layer(image_ul)
                
                ## Augmentation
                wk_image = self.color_augmentation(0.1, image_ul)
                wk_image = wk_image.cuda()
                #print(f"wk_image size : {wk_image.size()}")

                i, j, h, w = self.get_crop_params(image_ul)
                st_image = F.crop(image_ul, i, j, h, w)
                st_image = self.color_augmentation(0.5, st_image)
                st_image = self.resize_256_transform(st_image)
                st_image = st_image.cuda()
                #print(f"st_image size : {st_image.size()}")

                if t_idx % 50 == 0:
                    #visualize_rescale_image(imagenet_mean, imagenet_std, image_ul, "imagenet_org/imagenet")
                    visualize_rescale_image(imagenet_mean, imagenet_std, wk_image, "imagenet_wk/imagenet")
                    visualize_rescale_image(imagenet_mean, imagenet_std, st_image, "imagenet_st/imagenet")

                ## Getting cam
                wk_cam = []
                for img in wk_image:
                    img = img.unsqueeze(0)
                    #img = upscale_layer(img)
                    wk_cam_ = self.get_grad_cam(img)
                    wk_cam.append(wk_cam_)
                wk_cam = torch.stack(wk_cam)
                #print(f"wk_cam size : {wk_cam.size()}")
                if t_idx % 50 == 0:
                    visualize_cam(wk_image, wk_cam, imagenet_mean, imagenet_std, "wk_cam/imagenet")    

                st_cam = []
                for img in st_image:
                    img = img.unsqueeze(0)
                    #img = upscale_layer(img)
                    st_cam_ = self.get_grad_cam(img)
                    st_cam.append(st_cam_)
                st_cam = torch.stack(st_cam)
                #print(f"st_cam size : {st_cam.size()}")
                if t_idx % 50 == 0:       
                    visualize_cam(st_image, st_cam, imagenet_mean, imagenet_std, "st_cam/imagenet") 

                gt_cam = F.crop(wk_cam, i, j, h, w)
                gt_cam = self.resize_256_transform(gt_cam)
                #print(f"gt_cam_resize size : {gt_cam.size()}")
                if t_idx % 50 == 0:       
                    visualize_cam(st_image, gt_cam, imagenet_mean, imagenet_std, "gt_cam/imagenet")    
                

                wk_label = self.model(image_ul)
                wk_prob = torch.softmax(wk_label, dim=-1)
                #print(f"wk_prob : {wk_prob}")
                max_probs, max_idx = torch.max(wk_prob, dim=-1)
                mask_p = max_probs.ge(p_cutoff).float()
                mask_p = mask_p.cpu().detach().numpy()
                #print(mask_p)
                
                mask = [ torch.ones_like(gt_cam[0]) if int(mask_p[i]) else torch.zeros_like(gt_cam[0]) for i in range(gt_cam.size(0))]
                mask = torch.stack(mask)
                #print(f"mask size: {mask.size()}")
                
                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    mseloss = MSEloss(st_cam * mask, gt_cam * mask)
                    #mseloss = MSEloss(st_cam, gt_cam)
                    #mseloss = SSIM()(st_cam, gt_cam)
                    if math.isnan(mseloss) is False:
                        losses_list.append(mseloss)
                    wandb.log({"training/mseloss" : mseloss})
                    #print(f"mseloss: {MSEloss(st_cam, gt_cam)}")
                    #print(f"maksed mseloss: {mseloss}")
                
                t_loss = total_loss(losses_list)
                wandb.log({"training/tloss" : t_loss})

                #mask = [ int(mask_p[i]) for i in range(gt_cam.size(0))]
                #if t_idx % 50 == 0:
                #    wandb.log({"training/mask" : mask})

                self.optimizer.zero_grad()
                self.scaler.scale(t_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()            

                '''
                for idx, module in enumerate(self.model.modules()):
                    if idx == 470:
                        y = module.weight.grad.data.detach().clone()
                        y = y.detach().cpu().numpy()
                        print(np.where(y != 0))
                del image_ul, wk_image, st_image
                '''
                
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
    model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=20)
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

    train_loader, val_loader, test_loader, imagenet_loader = init(
        args.batch_size_train, args.batch_size_val, args.batch_size_test, args.batch_size_imagenet
    )

    trainer = TrainManager(
        model,
        optimizer,
        args,
        additional_cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        imagenet_loader=imagenet_loader,
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
    
    parser.add_argument('--batch-size-train', type=int, default=4,    
                        help='Batch size for train data (default: 16)')
    parser.add_argument('--batch-size-val', type=int, default=8,
                        help='Batch size for val data (default: 16)')
    parser.add_argument('--batch-size-test', type=int, default=4,
                        help='Batch size for test data (default: 16)')
    parser.add_argument('--batch-size-imagenet', type=int, default=4,
                        help='Batch size for test data (default: 128)')

    parser.add_argument('--save-ckpt', type=int, default=10,
                        help='number of epoch save current weight? (default: 5)')

    parser.add_argument('--start-epoch', type=int, default=0,
                        help='start epoch (default: 0)')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='end epoch (default: 30)')

    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay for SGD (default: 0.0005)')
    parser.add_argument('--lr-anneal-rate', type=float, default=0.95,
                        help='Annealing rate (default: 0.95)')
    

    
    args = parser.parse_args()
    main(args)