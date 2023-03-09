import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import filterfalse


class CircleLoss(nn.Module):
    """ https://arxiv.org/abs/2002.10857v2  Equation.4 """
    def __init__(self, optim_pos=1, optim_neg=0, temperature=10):
        """ 做完softmax之后分类 最优情况下 optim_p,  optim_n = 1, 0 
            cosim 得分的话  optim_p,  optim_n = 1, -1
        """
        super().__init__()
        self.optim_pos = optim_pos
        self.optim_neg = optim_neg
        self.temperature = temperature

    def forward(self, pred, gold):
        """ replacing ce, pred should be softmaxed """
        # pred = safesoftmax(pred)
        pred = torch.softmax(pred, dim=-1)
        bsz, num_classes = pred.size()
        pos_ones = F.one_hot(gold, num_classes)
        neg_ones = torch.ones_like(pos_ones) - pos_ones
        
        alpha_p = torch.clamp_min(self.optim_pos - pred, min=0)
        alpha_n = torch.clamp_min(pred - self.optim_neg, min=0)
        p_dists = torch.exp(-pred * self.temperature * alpha_p) * pos_ones
        n_dists = torch.exp(pred * self.temperature * alpha_n ) * neg_ones

        p_dists = torch.sum(p_dists, 1, True)
        n_dists = torch.sum(n_dists, 1, True)
        loss = torch.log(1+ (n_dists)*(p_dists))
        loss = loss.mean()
        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=1.6, alpha=0.4, size_average=True):
        """ alpha 是乘到负样本的loss上的权重，正样本是(1-alpha); None的话，两者相同，或者设为0.5?
            取该类别样本数目比例的倒数？
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1),-1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: 
            return loss.mean()
        else: 
            return loss.sum()


class DiceLoss(nn.Module):
    """ Taken from  https://github.com/lyakaap/pytorch-template """
    def __init__(self, smooth=1.0, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, output, target):
        output = torch.sigmoid(output)

        if torch.sum(target) == 0:
            output = 1.0 - output
            target = 1.0 - target

        return 1.0 - (2 * torch.sum(output * target) + self.smooth) / (
                torch.sum(output) + torch.sum(target) + self.smooth + self.eps)


class FocalLoss(nn.Module):
    """ Taken from  https://github.com/lyakaap/pytorch-template """
    def __init__(self, gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, logit, target):
        prob = torch.sigmoid(logit)
        prob = prob.clamp(self.eps, 1. - self.eps)

        loss = -1 * target * torch.log(prob)
        loss = loss * (1 - logit) ** self.gamma

        return loss.sum()