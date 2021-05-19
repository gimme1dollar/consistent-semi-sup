# Implemented upon pytorch 1.2.0
import torch.nn as nn
import torch
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import torchvision.transforms.functional as F
import torch.nn.functional as F_nn
from torch.autograd import Variable
from tqdm import tqdm
import wandb
import random
from utils.visualize import *
import copy

class JSD(nn.Module):
    
    def __init__(self):
        super(JSD, self).__init__()
    
    def forward(self, net_1_logits, net_2_logits):
        net_1_probs =  F.softmax(net_1_logits, dim=1)
        net_2_probs=  F.softmax(net_2_logits, dim=1)

        total_m = 0.5 * (net_1_probs + net_1_probs)
        loss = 0.0
        loss += F.kl_div(F.log_softmax(net_1_logits, dim=1), total_m, reduction="batchmean") 
        loss += F.kl_div(F.log_softmax(net_2_logits, dim=1), total_m, reduction="batchmean") 
     
        return (0.5 * loss)

def bhattacharyya_coefficient(a, b):
    # a, b : num_classes
    a = torch.sigmoid(a.squeeze())
    b = torch.sigmoid(b.squeeze())

    score = 0
    for i in range(a.shape[0]):
        score += torch.sqrt(a[i] * b[i])

    return 1 - score

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()

def constrative_loss(div1_origin, div2_new, target):
    cons_loss = ContrastiveLoss()(div1_origin, div2_new, target.cuda())
    return cons_loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(reduce=False, weight=alpha)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target.long())
        pt = torch.exp(-logpt)
        loss = ((1-pt)**self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()

def focal_loss(x, y):
    return FocalLoss()(x, y)

def CEloss(inputs, gt, ignore_index=255):
    return nn.CrossEntropyLoss(ignore_index=ignore_index)(inputs, gt)

def total_loss(losses_list):
    total = 0
    for component in losses_list:
        if isinstance(component, list):
            total += sum(component)
        else:
            total += component
    return total

def cosine_loss(x1, x2, y):
    
    p_loss = 0
    p_cnt = 1e-12
    n_loss = 0
    n_cnt = 1e-12

    for idx in range(x1.shape[0]):
        cos = torch.cosine_similarity(x1[idx], x2[idx], dim=0)
        if y[idx] == 1:
            p_loss += 3 - cos
            p_cnt += 1
            wandb.log({"training/same_sim" : cos}, commit=False)
            wandb.log({"training/same_sim_1-cos" : 3 - cos}, commit=False)

        elif y[idx] == -1:
            n_loss += max(0, cos + 0.5)
            n_cnt += 1
            wandb.log({"training/diff_sim" : max(0, cos + 0.5)}, commit=False)

    pm = p_loss / p_cnt
    nm = n_loss / n_cnt
    
    wandb.log({"training/same_avg_loss" : pm}, commit=False)
    wandb.log({"training/diff_avg_loss" : nm}, commit=False)
    return pm, nm

def triplet_loss(model, image, target):

    tp_loss = nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance())
    loss = 0
    pos_mean = 0
    neg_mean = 0
    for oidx in range(image.shape[0]):
        pos_list = []
        neg_list = []
        for iidx in range(image.shape[0]):
            label = (target[oidx] == target[iidx])
            if label.any(): # positive
                pos_list.append(iidx)
            else:
                neg_list.append(iidx)

        min_range = min(len(pos_list), len(neg_list))

        p1 = random.sample(pos_list, min_range)
        n1 = random.sample(neg_list, min_range)
        
        _, anc_vector = model(image[oidx].unsqueeze(0))
        _, pos_vector = model(image[p1])
        _, neg_vector = model(image[n1])
        pm = torch.pairwise_distance(anc_vector, pos_vector).mean()
        pos_mean += pm

        nm = torch.pairwise_distance(anc_vector, neg_vector).mean()
        neg_mean += nm
        loss += tp_loss(anc_vector, pos_vector, neg_vector)

    loss = loss / (image.shape[0])
    wandb.log({"training/triplet_loss":loss})

    pm = pos_mean / (image.shape[0])
    nm = neg_mean / (image.shape[0])
    return loss, pm, nm

def unsup_do(model, image, image_ul, th):
    tp_loss = nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance())
    loss = 0
    
    for oidx in range(image.shape[0]):
        
        dl20_output, anc_vec = model(image[oidx].unsqueeze(0))
        imgnet_output, tar_vec = model(image_ul)

        dist = torch.pairwise_distance(anc_vec, tar_vec)
        label = torch.where(dist <= (th * 1.5), 1,0)
        
        pos_list = []
        neg_list = []
        for iidx in range(label.shape[0]):
            if label[iidx] == 1: # positive
                pos_list.append(iidx)
            else:
                neg_list.append(iidx)
            
        min_range = min(len(pos_list), len(neg_list))
        p1 = random.sample(pos_list, min_range)
        n1 = random.sample(neg_list, min_range)

        loss += tp_loss(anc_vec, tar_vec[p1], tar_vec[n1])

    loss = loss / (image.shape[0])
    
    wandb.log({"semi/triplet_loss":loss})
    return loss

