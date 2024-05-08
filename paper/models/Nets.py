#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x

class Mlp(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim_hidden, dim_out)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ViTMLPModel(nn.Module):
    def __init__(self, args):
        super(ViTMLPModel, self).__init__()
        self.patch_size = args.patch_size
        self.embed_dim = args.embed_dim
        self.num_patches = (args.img_size[0] // self.patch_size) ** 2
        
        self.patch_embedding = nn.Conv2d(args.num_channels, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim))
        
        norm_layer = getattr(nn, args.norm_layer)  # 从nn模块中获取norm_layer类
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                self.embed_dim, args.num_heads, args.enable_sample, args.enable_hcs,
                args.channel_dropout_rate, args.mlp_ratio, args.qkv_bias, norm_layer
            )
            for _ in range(args.depth)
        ])
        
        self.norm = norm_layer(self.embed_dim)
        self.mlp_head = Mlp(self.embed_dim, self.embed_dim * args.mlp_ratio, args.num_classes)
        self.dropout = nn.Dropout(p=args.dropout_rate)  # Add dropout with a specified rate
        
    def forward(self, x):
        # Initial transformation
        x = self.patch_embedding(x)  # Apply patch embedding
        x = x.flatten(2).transpose(1, 2)  # Flatten and transpose
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # Expand class token for the batch
        x = torch.cat((cls_token, x), dim=1)  # Concatenate class token with embedded patches
        x = x + self.pos_embed  # Add positional embedding

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Apply normalization and classification head
        x = self.norm(x)  # Normalize
        x = x[:, 0]  # Extract class token
        x = self.mlp_head(x)  # Pass class token through MLP head
        x = self.dropout(x)  # Apply dropout

        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, enable_sample, enable_hcs, channel_dropout_rate=0., mlp_ratio=4., qkv_bias=False, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, enable_sample, enable_hcs, qkv_bias, channel_dropout_rate)
        self.norm2 = norm_layer(embed_dim)
        self.mlp = Mlp(embed_dim, int(embed_dim * mlp_ratio), embed_dim)
        
    def forward(self, x):
        #print(f"Input shape to TransformerBlock: {x.shape}")
        x = x + self.attn(self.norm1(x))
        #print(f"Shape after attention: {x.shape}")
        x = x + self.mlp(self.norm2(x))
        return x

    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, enable_sample, enable_hcs, qkv_bias=False, channel_dropout_rate=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(channel_dropout_rate)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(channel_dropout_rate)
        self.enable_sample = enable_sample
        self.enable_hcs = enable_hcs  # 添加这一行
        
    def forward(self, x):
        #print(f"Input shape: {x.shape}")
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * (1.0 / self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        #print(f"Shape after attention: {x.shape}")

        cls_token, x = x[:, 0:1], x[:, 1:]  # 分离cls_token和x

        if self.enable_hcs:
            x = x.view(B, N-1, self.num_heads, self.head_dim).permute(0, 2, 1, 3).reshape(B, N-1, self.embed_dim)
            #print(f"Shape before proj: {x.shape}")
            x = self.proj(x)
            #print(f"Shape after proj: {x.shape}")
        else:
            x = self.proj(x)

        x = torch.cat((cls_token, x), dim=1)  # 重新拼接cls_token和x
        x = self.proj_drop(x)
        return x
