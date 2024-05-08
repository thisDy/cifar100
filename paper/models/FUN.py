# PyTorch net and training necessities
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Image datasets and image manipulation

import torchvision.transforms as transforms

# Image display
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns

# Dataset process
from torch.utils.data import DataLoader, Subset
import pandas as pd
import os


import math
import random
from functools import partial
from typing import List
import torch.distributed as dist

from utils.options import args_parser
from models.functions import ChannelVisionTransformer
args = args_parser()

def build_client_net(args):
    client_net = {}
    files = os.listdir()
    for i in range(args.num_users):
        file = f'model_{i}.pth'
        if file in files:
            client_net[i] = torch.load(file, map_location=args.device)
        else:
            client_net[i] = ChannelVisionTransformer(
                img_size=args.img_size,
                num_classes=args.num_classes,
                enable_sample=args.enable_sample,
                patch_size=args.patch_size,
                embed_dim=args.embed_dim,
                depth=args.depth,
                num_heads=args.num_heads,
                mlp_ratio=args.mlp_ratio,
                qkv_bias=args.qkv_bias,
                norm_layer=partial(nn.LayerNorm, eps=1e-6) if args.norm_layer == 'LayerNorm' else None,
                enable_hcs=args.enable_hcs,
                channel_dropout_rate=args.channel_dropout_rate,
            ).to(args.device)
    return client_net

def create_extra_tokens(x):
    channels = list(range(x.shape[1]))  # 假设 x 的形状是 [batch_size, channels,height, width]
    return {"channels": torch.tensor(channels)}
def train_loop(dataloader, model, loss_fn, optimizer):
    for X, y in dataloader:
        extra_tokens = create_extra_tokens(X)  # 确保这个函数生成了正确的 extra_tokens
        X = X.to(device)
        y = y.to(device)
        pred = model(X, extra_tokens)  # 使用 extra_tokens
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()