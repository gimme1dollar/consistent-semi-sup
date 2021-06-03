from efficientnet_pytorch.model import EfficientNet
import torch
import torch.multiprocessing
from torchvision import transforms
from dataset.dataloader import LoadDataset, LoadSemiDataset
from tqdm import tqdm
from os.path import join as pjn
import os.path, os, datetime, time
import wandb, argparse
from utils.losses import *
from utils.semi_sup import adv_self_training, semi_sup_learning
import math
import warnings

torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings(action='ignore')

torch.backends.cudnn.benchmark = True

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

data_path = pjn(os.getcwd(), "dataset", "DL20")

def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)

def init(train_batch, val_batch, test_batch, args):

    # default augmentation functions : http://incredible.ai/pytorch/2020/04/25/Pytorch-Image-Augmentation/ 
    # for more augmentation functions : https://github.com/aleju/imgaug

    transform_unlabel = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        #transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply( [transforms.ColorJitter( brightness=(0.2, 2) )], p=0.5),
        transforms.RandomApply([transforms.ColorJitter(  contrast=(0.3, 2)   )], p=0.5),
        transforms.RandomApply([transforms.ColorJitter(  saturation=(0.2, 2) )], p=0.5),
        transforms.RandomApply([transforms.ColorJitter( hue=(-0.3, 0.3))], p=0.5),
        transforms.RandomApply([transforms.RandomRotation(90, expand=False)], p=0.5),
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

    label_dataset = LoadSemiDataset(data_path = data_path, transform=transform_unlabel , mode='label', ratio=args.ratio)
    unlabel_dataset = LoadSemiDataset(data_path = data_path, transform=transform_unlabel , mode='unlabel', ratio=args.ratio)
    val_dataset = LoadDataset(data_path = data_path, transform=transform_val , mode='valid')
    test_dataset = LoadDataset(data_path = data_path, transform=transform_test , mode='test')

    label_loader = torch.utils.data.DataLoader(
            dataset=label_dataset, batch_size=train_batch,
            num_workers=4, shuffle=True, pin_memory=True
    )

    unlabel_loader = torch.utils.data.DataLoader(
            dataset=unlabel_dataset, batch_size=4,
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

    return label_loader, unlabel_loader, val_loader, test_loader
    
class TrainManager(object):
    def __init__(
            self,
            model,
            teacher,
            second_student,
            optimizer,
            optimizer2,
            args,
            additional_cfg,
            label_loader,
            unlabel_loader,
            val_loader,
            unlabel_testloader=None,
            scaler=None,
            num_classes=None,
            ):
        self.model = model
        self.teacher = teacher
        self.sec_student = second_student
        self.label_loader = label_loader
        self.unlabel_loader = unlabel_loader
        self.unlabel_testloader=unlabel_testloader
        self.optimizer = optimizer
        self.optimizer2 = optimizer2
        self.args = args
        self.add_cfg = additional_cfg
        self.tbx_wrtr_dir = additional_cfg.get('tbx_wrtr_dir')
        self.scaler = scaler
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.upsampler = torch.nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsampler_ul = torch.nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.best_score=0
        self.flag = True
    def validate(self, device, model, teacher, topk=(1,3,5)):
        self.model.eval()
        self.teacher.eval()
        total = 0
        maxk = max(topk)
        correct_1 = 0
        correct_3 = 0
        correct_5 = 0

        correct_1_t = 0
        correct_3_t = 0
        correct_5_t = 0

        with torch.no_grad():
            for b_idx, (image, labels) in tqdm(enumerate(self.val_loader), desc="validation", leave=False):
                image = self.upsampler(image)
                image = image.to(device)
                labels = labels.to(device)

                total += image.shape[0]
                
                outputs = model(image) # b x 1
                outputs_t = teacher(image) 

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
                
                _, pred = outputs_t.topk(maxk, 1, True, True)
                pred = pred.t()
                correct = (pred == labels.unsqueeze(dim=0)).expand_as(pred)

                for k in topk:
                    if k == 1:
                        correct_k = correct[:1].reshape(-1).float().sum(0, keepdim=True)
                        correct_1_t += correct_k.item()
                    elif k == 3:
                        correct_k = correct[:3].reshape(-1).float().sum(0, keepdim=True)
                        correct_3_t += correct_k.item()
                    elif k == 5:
                        correct_k = correct[:5].reshape(-1).float().sum(0, keepdim=True)
                        correct_5_t += correct_k.item()
                    else:
                        raise NotImplementedError("Invalid top-k num")

            model_top1, model_top3, model_top5 = (correct_1 / total) * 100, (correct_3 / total) * 100, (correct_5 / total) * 100
            model_top1_t, model_top3_t, model_top5_t = (correct_1_t / total) * 100, (correct_3_t / total) * 100, (correct_5_t / total) * 100
        return (model_top1, model_top3, model_top5), (model_top1_t, model_top3_t, model_top5_t)

    def validate_sect(self, device, topk=(1,3,5)):
        self.sec_student.eval()
        total = 0
        maxk = max(topk)
        correct_1 = 0
        correct_3 = 0
        correct_5 = 0

        with torch.no_grad():
            for b_idx, (image, labels) in tqdm(enumerate(self.val_loader), desc="validation", leave=False):
                image = self.upsampler(image)
                image = image.to(device)
                labels = labels.to(device)

                total += image.shape[0]
                outputs = self.sec_student(image) # b x 1
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
        return (model_top1, model_top3, model_top5)

    def train(self):
        start = time.time()
        epoch = 0
        iter_per_epoch = 31 # len(self.label_loader)
        print("  iteration per epoch(considered batch size): ", iter_per_epoch)
        print("  label iter : ", len(self.label_loader))
        print("  unlabel iter : ", len(self.unlabel_loader))
        print("  val iter : ", len(self.val_loader))
        print("  Progress bar for training epochs:")
        end_epoch = self.args.start_epoch + self.args.num_epochs

        dataloader = iter(zip(cycle(self.label_loader), cycle(self.unlabel_loader)))
        alpha = 0.965
        for epoch in tqdm(range(self.args.start_epoch, end_epoch), desc='epochs', leave=False):
            self.model.train()
            for idx, param_group in enumerate(self.optimizer.param_groups):
                avg_lr = param_group['lr']
                wandb.log({str(idx)+"_lr": math.log10(avg_lr), 'epoch': epoch})

            for t_idx in tqdm(range(iter_per_epoch),  desc='batch_iter', leave=False, total=iter_per_epoch):
                (image, target), (image_ul, label_ul) = next(dataloader)
                image = self.upsampler(image)
                image = image.to(self.add_cfg['device']) # DL20
                target = target.to(self.add_cfg['device'])

                image_ul = image_ul.to(self.add_cfg['device']) # DL20
                label_ul = label_ul.to(self.add_cfg['device']) # DL20

                self.optimizer.zero_grad()
                losses_list = []
                with torch.cuda.amp.autocast():
                    ### sup loss ###
                    outputs = self.model(image)
                    celoss = CEloss(outputs, target)
                    losses_list.append(celoss)  

                    with torch.no_grad():
                        teacher_output = self.teacher(image)
                    con_loss_sup = softmax_kl_loss(outputs, teacher_output.detach()) / image.shape[0]

                    losses_list.append(0.5 * con_loss_sup)  

                    ## unsup loss ##
                    if self.flag is True:
                        con_loss = semi_sup_learning(self, image, image_ul)
                        if con_loss != 0:
                            losses_list.append(con_loss)
                        wandb.log({"training/vat_loss" : con_loss}) 
                    
                    if self.args.second_stage:
                        adv_self_training(self, image, outputs, target, image_ul)
                
                wandb.log({"training/celoss" : celoss})
                wandb.log({"training/con_loss_sup" : 0.5 * con_loss_sup})
                    
                t_loss = total_loss(losses_list)
                self.scaler.scale(t_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                for mean_param, param in zip(self.teacher.parameters(), self.model.parameters()):
                    mean_param.data.mul_(alpha).add_(1 - alpha, param.data)

            self.adjust_learning_rate(epoch)
            self.save_ckpt(epoch)

            # if epoch % 8 == 0 and epoch != 0:
            #     self.validate_unlabel(self.add_cfg['device'], self.bce_model, self.unlabel_testloader, tag="unlabeled")

            if epoch % 2 == 0:
                (model_top1, model_top3, model_top5), (model_top1_t, model_top3_t, model_top5_t) \
                     = self.validate(self.add_cfg['device'], self.model, self.teacher)
                wandb.log({"validation/top1_acc" : model_top1, "validation/top3_acc" : model_top3, "validation/top5_acc" : model_top5}, commit=False)
                wandb.log({"validation/teacher_top1_acc" : model_top1_t, "validation/teacher_top3_acc" : model_top3_t, "validation/teacher_top5_acc" : model_top5_t})
                
                (model_top1, model_top3, model_top5)\
                     = self.validate_sect(self.add_cfg['device'])
                wandb.log({"sec_t/top1_acc" : model_top1, "sec_t/top3_acc" : model_top3, "sec_t/top5_acc" : model_top5})
                
                if max(model_top1, model_top1_t) >= self.best_score:
                    self.best_score = max(model_top1, model_top1_t)
                    self.save_ckpt(epoch, 1)

        end = time.time()   
        print("Total training time : ", str(datetime.timedelta(seconds=(int(end)-int(start)))))
        print("Finish.")

    def adjust_learning_rate(self, epoch):
        # update optimizer's learning rate
        for param_group in self.optimizer.param_groups:
            prev_lr = param_group['lr']
            param_group['lr'] = prev_lr * self.args.lr_anneal_rate

    def save_ckpt(self, epoch, nm=None):
        if epoch % self.args.save_ckpt == 0:
            
            if nm:
                nm = "best.pth"
            else:
                nm = f'epoch_{epoch:04d}.pth'

            if not os.path.isdir(pjn('checkpoints', self.tbx_wrtr_dir)):
                os.mkdir(pjn('checkpoints', self.tbx_wrtr_dir))

            fpath=pjn('checkpoints', self.tbx_wrtr_dir, nm)

            d = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'teacher_state_dict': self.teacher.state_dict(),
                'second_student_state_dict' : self.sec_student.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
            torch.save(d, fpath)

def main(args):
    # for deterministic training, enable all below options.
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed) # if use multi-GPU
    #torch.backends.cudnn.deterministic = False
    # np.random.seed(args.seed)
    # random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    wandb.init(project="DL20")
    orig_cwd = os.getcwd()
    
    # bring effi model from this : https://github.com/lukemelas/EfficientNet-PyTorch
    model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=20).cuda()
    model.set_swish(True)
    teacher = EfficientNet.from_pretrained('efficientnet-b4', num_classes=20).cuda()
    teacher.set_swish(True)

    second_student = EfficientNet.from_pretrained('efficientnet-b4', num_classes=20).cuda()
    second_student.set_swish(True)

    for param in teacher.parameters():
        param.detach_()
    
    additional_cfg = {'device': None}
    additional_cfg['device'] = torch.device('cuda')

    trainable_params = [
        {'params': list(filter(lambda p:p.requires_grad, model.parameters())), 'lr':args.lr},
    ]

    optimizer = torch.optim.SGD(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=0.9)
    
    optimizer2 = torch.optim.SGD(
        second_student.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=0.9)
    
    
    scaler = torch.cuda.amp.GradScaler() 

    if args.pretrained_ckpt:
        print(f"  Using pretrained model only and its checkpoint "
              f"'{args.pretrained_ckpt}'")
        loaded_struct = torch.load(pjn(orig_cwd, args.pretrained_ckpt))
        model.load_state_dict(loaded_struct['model_state_dict'], strict=False)
        teacher.load_state_dict(loaded_struct['teacher_state_dict'], strict=False)
        print("load optimizer's params")
        loaded_struct = torch.load(pjn(orig_cwd, args.pretrained_ckpt))
        optimizer.load_state_dict(loaded_struct['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    now = datetime.datetime.now()
    ti = str(now.strftime('%Y-%m-%d-%H-%M-%S'))
    

    if args.exp_name:
        wandb.run.name = str(args.exp_name)
        additional_cfg['tbx_wrtr_dir'] = os.getcwd() + "/checkpoints/" + str(args.exp_name)
    else:
        wandb.run.name = str(ti)
        additional_cfg['tbx_wrtr_dir'] = os.getcwd() + "/checkpoints/" + str(ti)
    label_loader, unlabel_loader, val_loader, _ = init(
        args.batch_size_train, args.batch_size_val, args.batch_size_test, args
    )

    trainer = TrainManager(
        model,
        teacher,
        second_student,
        optimizer,
        optimizer2,
        args,
        additional_cfg,
        label_loader=label_loader,
        unlabel_loader=unlabel_loader,
        val_loader=val_loader,
        scaler=scaler,
        num_classes=20
    )

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Name of the experiment (default: auto)')
    parser.add_argument('--pretrained-ckpt', type=str, default=None,
                        help='Load pretrained weight, write path to weight (default: None)')
    
    parser.add_argument('--batch-size-train', type=int, default=4,    
                        help='Batch size for train data (default: 16)')
    parser.add_argument('--batch-size-val', type=int, default=64,
                        help='Batch size for val data (default: 4)')
    parser.add_argument('--batch-size-test', type=int, default=1,
                        help='Batch size for test data (default: 128)')
    parser.add_argument('--ratio', type=float, default=0.02)

    parser.add_argument('--save-ckpt', type=int, default=5,
                        help='number of epoch save current weight? (default: 5)')

    parser.add_argument('--start-epoch', type=int, default=0,
                        help='start epoch (default: 0)')
    parser.add_argument('--num-epochs', type=int, default=222,
                        help='end epoch (default: 30)')

    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay for SGD (default: 0.0005)')
    parser.add_argument('--lr-anneal-rate', type=float, default=0.995,
                        help='Annealing rate (default: 0.95)')
    parser.add_argument('--second-stage', type=bool, default=False,
                        help='True if distillation loop training')
    
    args = parser.parse_args()
    main(args)