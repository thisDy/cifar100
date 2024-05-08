#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import math

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        self.rho = rho
        self.base_optimizer_cls = base_optimizer
        self.base_optimizer_kwargs = kwargs
        super().__init__(params, kwargs)
        self.base_optimizer = self.base_optimizer_cls(self.param_groups, **kwargs)

    def __setstate__(self, state):
        super(SAM, self).__setstate__(state)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None:
                    continue
                # 初始化state中的'e_w'键（如果尚未初始化）
                if 'e_w' not in self.state[p]:
                    self.state[p]['e_w'] = torch.zeros_like(p.data)
                # 更新'e_w'键
                e_w = self.state[p]['e_w']
                e_w.copy_(p.data)
                p.add_(p.grad, alpha=scale)  # move to the new temporary solution
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # 恢复原始参数值
                p.data.copy_(self.state[p]['e_w'])
        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        if zero_grad:
            self.zero_grad()

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def _grad_norm(self):
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2)
                        for group in self.param_groups
                        for p in group['params']
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def create_extra_tokens(self, x):
        channels = list(range(x.shape[1]))  # 假设 x 的形状是 [batch_size, channels, height, width]
        return {"channels": torch.tensor(channels)}

    def adjust_weight_decay(self, optimizer, current_epoch, total_epochs, initial_weight_decay, max_weight_decay):
        """计算并更新当前的权重衰减值"""
        cos_inner = (math.pi * current_epoch) / total_epochs
        weight_decay = initial_weight_decay + 0.5 * (max_weight_decay - initial_weight_decay) * (1 + math.cos(cos_inner))
        for param_group in optimizer.param_groups:
            param_group['weight_decay'] = weight_decay

    def train(self, net):
        optimizer = SAM(net.parameters(), torch.optim.SGD, lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for epoch in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                output = net(images)
                loss = self.loss_func(output, labels)
                loss.backward()
                optimizer.first_step(zero_grad=True)
                
                # Second forward-backward pass
                self.loss_func(net(images), labels).backward()  # You might need to recompute the outputs for some models
                optimizer.second_step(zero_grad=True)

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

