import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from utils.losses import *
import wandb, random
from copy import deepcopy

def semi_sup_learning(self, input_ul, label_ul):
    input_ul = self.upsampler_ul(input_ul)
    batch = input_ul.shape[0]
    student_label = self.model(input_ul)

    with torch.no_grad():
        teacher_label = self.teacher(input_ul).detach()

    vat_loss = VAT(model = self.model, n_power=1, XI=1e-6, eps=3)
    lds = vat_loss(input_ul, student_label)

    return lds

def adv_self_training(self, input_l, teacher_output, label_l, input_ul):
    input_ul = self.upsampler_ul(input_ul)
    self.optimizer2.zero_grad()

    teacher_soft_label = torch.softmax(teacher_output, dim=1)

    student_output = self.sec_student(input_l)

    loss = []
    softloss_sup = softmax_kl_loss(student_output, teacher_soft_label) / student_output.shape[0]
    hardloss_sup = CEloss(student_output, label_l)

    teacher_output_unsup = self.model(input_ul)

    student_output_unsup = self.sec_student(input_ul)
    teacher_soft_label_unsup = torch.softmax(teacher_output_unsup, dim=1)
    teacher_hard_label_unsup = torch.argmax(teacher_output_unsup, dim=1)

    softloss_sup_unsup  = softmax_kl_loss(student_output_unsup, teacher_soft_label_unsup) / teacher_output_unsup.shape[0]
    hardloss_sup_unsup  = CEloss(student_output_unsup, teacher_hard_label_unsup)

    loss.append(softloss_sup)
    loss.append(hardloss_sup)

    loss.append(softloss_sup_unsup)
    loss.append(hardloss_sup_unsup)

    t_loss = total_loss(loss)

    self.scaler.scale(t_loss).backward(retain_graph=True)
    self.scaler.step(self.optimizer2)
    self.scaler.update()

    wandb.log({"self_train/softloss_sup" : softloss_sup})
    wandb.log({"self_train/hardloss_sup" : hardloss_sup})
    wandb.log({"self_train/softloss_sup_unsup" : softloss_sup_unsup})
    wandb.log({"self_train/hardloss_sup_unsup" : hardloss_sup_unsup})
    

    # for ind in range(batch):
    #     anc = torch.softmax(teacher_label[ind].unsqueeze(0), dim=1)
    #     if torch.max(anc) < 0.90: continue
        
    #     total_true_postive_anchor_cnt += (label_ul[ind] == torch.argmax(anc, dim=1)).long()
        
    #     pos = []
    #     for iind in range(batch):
    #         tar = torch.softmax(teacher_label[iind].unsqueeze(0), dim=1)
    #         if (torch.argmax(anc, dim=1) == torch.argmax(tar, dim=1)) and (torch.max(tar) > 0.93):
    #             pos.append(iind)
    #             total_cnt += 1
    #             total_true_postive_cnt += (label_ul[ind] == torch.argmax(tar, dim=1)).long()
            
    #     if len(pos) != 0:
    #         anchor_cnt += 1
    #         pos_vec = teacher_label[pos]
    #         loss +=(1 - torch.cosine_similarity(student_label[ind].unsqueeze(0),pos_vec)).mean(dim=0)
        
    # if anchor_cnt != 0:
    #     wandb.log({"semi-sup/ratio_of_true_positive_anchor" :  total_true_postive_anchor_cnt/anchor_cnt, \
    #     "semi-sup/ratio_of_true_positive_target" :  total_true_postive_cnt/total_cnt if total_true_postive_cnt > 0 else -1, \
    #     })
    #     return loss / anchor_cnt
    # return 0
    # for ind in range(batch):
    #     anc = torch.softmax(teacher_label[ind].unsqueeze(0), dim=1)
    #     print("anchor : ", torch.max(anc))
    #     if torch.max(anc) < 0.90: continue
    #     anchor_cnt += 1
    #     total_true_postive_anchor_cnt += (label_ul[ind] == torch.argmax(anc, dim=1)).long()
        
    #     pos = []
    #     neg = []
    #     for iind in range(batch):
    #         tar = torch.softmax(teacher_label[iind].unsqueeze(0), dim=1)
    #         print(torch.max(tar), torch.max(anc))
    #         if (torch.argmax(anc, dim=1) == torch.argmax(tar, dim=1)) and (torch.max(tar) > 0.70):
    #             pos.append(iind)
    #             total_cnt += 1
    #             total_true_postive_cnt += (label_ul[ind] == torch.argmax(tar, dim=1)).long()
    #         elif (torch.argmax(anc, dim=1) != torch.argmax(tar, dim=1)) and (torch.max(tar) > 0.70):
    #             neg.append(iind)
    #             total_cnt += 1
    #             total_true_negative_cnt += (label_ul[ind] != torch.argmax(tar, dim=1)).long()

    #     if len(neg) > len(pos):
    #         com_num = len(neg) - len(pos)
    #         zero = torch.zeros(com_num, teacher_label.shape[1]).cuda()
            
    #         for i in range(com_num):
    #             zero[i] = self.model(aug(input_ul[ind]).unsqueeze(0))
    #         pos_vec = student_label[pos]
    #         pos_vec = torch.cat([pos_vec, zero], dim=0)
    #         neg_vec = student_label[neg]
    #     else:
    #         if len(neg) == 0:continue
    #         min_range = min(len(pos), len(neg))
    #         p1 = random.sample(pos, min_range)
    #         n1 = random.sample(neg, min_range)
    #         pos_vec = student_label[p1]
    #         neg_vec = student_label[n1]

    #     loss += tp_loss(anc, pos_vec, neg_vec)

    # if anchor_cnt != 0:
    #     wandb.log({"semi-sup/ratio_of_true_positive_anchor" :  total_true_postive_anchor_cnt/anchor_cnt, \
    #     "semi-sup/ratio_of_true_positive_target" :  total_true_postive_cnt/total_cnt if total_true_postive_cnt > 0 else -1, \
    #     "semi-sup/ratio_of_true_negative_target" :  total_true_negative_cnt/total_cnt if total_true_negative_cnt > 0 else -1, \
    #     })
    #     return loss / anchor_cnt





























    # loss = 0
    # celoss = 0
    # cnt = 0
    # bind=[]

    # true_cnt = 0
    # for batch_index in range(batch):
    #     confi = torch.max(torch.softmax(self.teacher(input_ul).detach(), dim=1))
    #     if confi >= 0.95:
    #         bind.append(batch_index)
    #         pred = torch.argmax(teacher_label[batch_index].unsqueeze(0), dim=1)
    #         true_cnt += (pred == label_ul[batch_index]).long()
    #         cnt += 1
            
    # if len(bind) != 0:
    #     wandb.log({"training/true_ratio : " : true_cnt / cnt})
    
    # if len(bind) == 0:
    #     return 0
    # elif len(bind) == 1:
    #     #a = softmax_kl_loss(student_label[bind].unsqueeze(0), teacher_label[bind].unsqueeze(0))
    #     #b = softmax_kl_loss(student_label[bind].unsqueeze(0), student_label_aug[bind].unsqueeze(0))
    #     c = CEloss(student_label[bind].unsqueeze(0), torch.argmax(teacher_label[bind].unsqueeze(0), dim=1))
    #     d = CEloss(student_label[bind].unsqueeze(0), torch.argmax(student_label_aug[bind].unsqueeze(0), dim=1))
    #     return c+d
    # else:
    #     #a = softmax_kl_loss(student_label[bind], teacher_label[bind]) / cnt
    #     #b = softmax_kl_loss(student_label[bind], student_label_aug[bind])/ cnt
    #     c = CEloss(student_label[bind], torch.argmax(teacher_label[bind], dim=1))
    #     d = CEloss(student_label[bind], torch.argmax(student_label_aug[bind].detach(), dim=1))
    #     return c+d


    