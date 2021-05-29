# reference git repository: https://github.com/sthalles/SimCLR.git

import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint

# kam add
from itertools import cycle

torch.manual_seed(0)


class Semi_KamSimCLR2(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        if 'scheduler' in kwargs:
            self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

        self.upsampler = torch.nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        # self.model_sup = kwargs['model_sup'].to(self.args.device)
        # self.softmax = torch.nn.Softmax(dim=1)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)
        s = labels[labels !=0]
        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        m = ~mask
        m2 = labels[~mask]
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, label_loader, unlabel_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")


        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        iter_per_epoch = len(label_loader)
        for epoch_counter in range(self.args.epochs):
            for t_idx, (image, target) in tqdm(enumerate(label_loader),  desc='batch_iter', leave=False, total=iter_per_epoch):
                images = torch.cat(image, dim=0)
                images = self.upsampler(images)
                images = images.to(self.args.device) # DL20
                gt_labels = target.to(self.args.device)
                gt_matrix = (gt_labels.unsqueeze(0) == gt_labels.unsqueeze(1)).float()
                # images = torch.cat(images, dim=0)

                # images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    features2 = self.model_sup(images)
                    pseudo_label = self.softmax(features2)
                    logits, labels = self.info_nce_loss(features, gt_labels)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_last_lr()[0], global_step=n_iter)
                    if top1[0] > 99.0 : 
                        print('Acc/top1:', top1[0])
                        break
                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")
 
  
        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'Effi_checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
        print('Training has finished. filename:', os.path.join(self.writer.log_dir, checkpoint_name))

    # def pair_img(self, unlabel_loader,labels):
    #     remain_labels = []
    #     for l in labels: 
    #         remain_labels.append(l)
    #     images2 = []
    #     while True:
    #         img2 = next(iter(unlabel_loader))
    #         img2 = img2.to(self.args.device)
    #         feat2 = self.model_sup(img2)
    #         pseudo_logit = self.softmax(feat2)
    #         pseudo_label = torch.argmax(pseudo_logit,dim=1) 
    #         for i in range(len(pseudo_label)):
    #             idx = pseudo_label[i]
    #             if pseudo_logit[i,idx] > 0.05:
    #                 for j in range(len(remain_labels)):
    #                     if remain_labels[j] == idx:
    #                         remain_labels.pop(j)
    #                         images2.appned(img2[i])
    #                         if len(remain_labels) == 0:
    #                             return img2
    
    def train2(self, label_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")

        logging.info(f"Training with gpu: {self.args.disable_cuda}.")
        iter_per_epoch = len(label_loader)
        for epoch_counter in range(self.args.epochs):
            for t_idx, (img1, img2, labels) in tqdm(enumerate(label_loader),  desc='batch_iter', leave=False, total=iter_per_epoch):
                # image, labels = next(iter(label_loader))
                # item = labels.cpu().numpy()
                # unlabel_loader.dataset.label= item
                # img2, labels2 = next(iter(unlabel_loader))
            
                images = torch.cat((img1,img2), dim=0)

                images = self.upsampler(images)
                images = images.to(self.args.device) # DL20
                # images = torch.cat(images, dim=0)

                # images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_last_lr()[0], global_step=n_iter)
                    if top1[0] > 99.0 : 
                        print('Acc/top1:', top1[0])
                        break
                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")
 
  
        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'Effi_checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
        print('Training has finished. filename:', os.path.join(self.writer.log_dir, checkpoint_name))
        
    def eval(self, label_loader, unlabel_loader, val_loader):
        epochs = 100
        print("start !")
        iter_per_epoch = len(label_loader)
        for epoch in range(epochs):
            # top1_train_accuracy = 0
            for counter, (x_batch, y_batch) in tqdm(enumerate(label_loader),  desc='batch_iter', leave=False, total=iter_per_epoch):
                x_batch = x_batch.to(self.args.device)
                y_batch = y_batch.to(self.args.device)

                logits = self.model(x_batch)
                loss = self.criterion(logits, y_batch)
                top1 = accuracy(logits, y_batch, topk=(1,))
                # top1_train_accuracy += top1[0]

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print(epoch, 'epochs training top1 Acc :', top1[0].item())
            top1_accuracy = 0.0
            top3_accuracy = 0.0
            top5_accuracy = 0.0

            if epoch % 5 == 0:
                for counter, (x_batch, y_batch) in enumerate(val_loader):
                    x_batch = x_batch.to(self.args.device)
                    y_batch = y_batch.to(self.args.device)

                    logits = self.model(x_batch)
                
                    top1, top3, top5 = accuracy(logits, y_batch, topk=(1,3,5))
                    top1_accuracy += top1[0]
                    top3_accuracy += top3[0]
                    top5_accuracy += top5[0]
                
                top1_accuracy /= (counter + 1)
                top3_accuracy /= (counter + 1)
                top5_accuracy /= (counter + 1)

                print("Epoch", epoch)
                print("Top1 Test acc:", top1_accuracy.item())
                print("Top3 Test acc:", top3_accuracy.item())
                print("Top5 test acc:", top5_accuracy.item())

        # top1_train_accuracy /= (counter + 1)
        # top1_accuracy = 0.0
        # top3_accuracy = 0.0
        # top5_accuracy = 0.0

        # for counter, (x_batch, y_batch) in enumerate(val_loader):
        #     x_batch = x_batch.to(self.args.device)
        #     y_batch = y_batch.to(self.args.device)

        #     logits = self.model(x_batch)
        
        #     top1, top3, top5 = accuracy(logits, y_batch, topk=(1,3,5))
        #     top1_accuracy += top1[0]
        #     top3_accuracy += top3[0]
        #     top5_accuracy += top5[0]
        
        # top1_accuracy /= (counter + 1)
        # top3_accuracy /= (counter + 1)
        # top5_accuracy /= (counter + 1)

        # print("Epoch", epochs)
        # print("Top1 Test acc:", top1_accuracy.item())
        # print("Top3 Test acc:", top3_accuracy.item())
        # print("Top5 test acc:", top5_accuracy.item())
        
        checkpoint_name = 'Fin_Effi_checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        # save_checkpoint({
        #     'epoch': self.args.epochs,
        #     'arch': self.args.arch,
        #     'state_dict': self.model.state_dict(),
        #     'optimizer': self.optimizer.state_dict(),
        # }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        # logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
        # print('Training has finished. filename:', os.path.join(self.writer.log_dir, checkpoint_name))
        
        d = {
                'epoch': epochs,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }

        torch.save(d, os.path.join(self.writer.log_dir, checkpoint_name))
        print('Finished. filename:', os.path.join(self.writer.log_dir, checkpoint_name))