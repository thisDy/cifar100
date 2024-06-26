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

from einops import rearrange, repeat
class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

import torch
import torch.nn as nn
import random
from torch.nn.init import trunc_normal_

class PatchEmbedPerChannel(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        enable_sample: bool = True,
        channel_dropout_rate: float = 0.0,  # 假设有一个通道 dropout 参数
    ):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size) * in_chans
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv3d(
            1,
            embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
        )

        self.channel_embed = nn.Embedding(in_chans, embed_dim)
        self.enable_sample = enable_sample
        self.channel_dropout_rate = channel_dropout_rate

        trunc_normal_(self.channel_embed.weight, std=0.02)

    
    def forward(self, x, extra_tokens={}):
        B, Cin, H, W = x.shape
        # print('Shape of x before operations:', x.shape)
        cur_channel_embed = self.channel_embed(extra_tokens["channels"])
        # print('Shape of cur_channel_embed:', cur_channel_embed.shape)
        if self.training and self.enable_sample:
            Cin_new = random.randint(1, Cin)
            channels = random.sample(range(Cin), k=Cin_new)
            x = x[:, channels, :, :]
            # print('Shape of x after channel sampling:', x.shape)
            cur_channel_embed = cur_channel_embed[channels]
        if self.training and self.channel_dropout_rate > 0.0:
            dropout_mask = torch.rand(B, Cin, 1, 1, device=x.device) > self.channel_dropout_rate
            x = x * dropout_mask
            # print('Shape of x after dropout:', x.shape)
        x = self.proj(x.unsqueeze(1))  # B, Cout, Cin, H, W
        # print('Shape of x after projection:', x.shape)
        cur_channel_embed = rearrange(cur_channel_embed, 'a b -> b a')
        cur_channel_embed = repeat(cur_channel_embed, 'b a -> 1 b a 4 4')
        x += cur_channel_embed
        # print('Shape of x after adding embed:', x.shape)
        x = x.flatten(2)  # B, Cout, CinHW
        x = x.transpose(1, 2)  # B, CinHW, Cout
        # print('Shape of x final:', x.shape)
        return x, Cin

class ChannelVisionTransformer(nn.Module):
    """Channel Vision Transformer"""

    def __init__(
        self,
        img_size=[224],
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        enable_sample=True,
        **kwargs,
    ):
        super().__init__()
#         # print(
#             "Warning!!!\n"
#             "Samplev2 channel vit randomly sample channels for each batch.\n"
#             "It is only compatible with Supervised learning\n"
#             "Doesn't work with DINO or Linear Prob"
#         )

        self.num_features = self.embed_dim = self.out_dim = embed_dim
        self.in_chans = in_chans

#         self.patch_embed = PatchEmbedPerChannel(
#             img_size=img_size[0],
#             patch_size=patch_size,
#             in_chans=in_chans,
#             embed_dim=embed_dim,
#             enable_sample=enable_sample,
#         )
        self.patch_embed = PatchEmbedPerChannel(
                    img_size=img_size[0],
                    patch_size=patch_size,
                    in_chans=in_chans,
                    embed_dim=embed_dim,
                    enable_sample=enable_sample,
                )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.num_extra_tokens = 1  # cls token

        self.pos_embed = nn.Parameter(
            torch.zeros(
                1, num_patches // self.in_chans + self.num_extra_tokens, embed_dim
            )
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h, c):
        # number of auxilary dimensions before the patches
        if not hasattr(self, "num_extra_tokens"):
            # backward compatibility
            num_extra_tokens = 1
        else:
            num_extra_tokens = self.num_extra_tokens

        npatch = x.shape[1] - num_extra_tokens
        N = self.pos_embed.shape[1] - num_extra_tokens

        if npatch == N and w == h:
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, :num_extra_tokens]
        patch_pos_embed = self.pos_embed[:, num_extra_tokens:]
        
        # print('class_pos_embed pre', class_pos_embed.shape)
        # print('patch_pos_embed pre', patch_pos_embed.shape)
        
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(
                1, int(math.sqrt(N)), int(math.sqrt(N)), dim
            ).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        # print('patch_pos_embed inter', patch_pos_embed.shape)
        assert (
            int(w0) == patch_pos_embed.shape[-2]
            and int(h0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, 1, -1, dim)

        # create copies of the positional embeddings for each channel
        patch_pos_embed = patch_pos_embed.expand(1, c, -1, dim).reshape(1, -1, dim)
        
        # print('patch_pos_embed post', patch_pos_embed.shape)
        # print('concat', torch.cat((class_pos_embed, patch_pos_embed), dim=1).shape)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def prepare_tokens(self, x, extra_tokens):
        B, nc, w, h = x.shape
        # print('nc pre', nc)
        # print('input pre x', x.shape)
        x, nc = self.patch_embed(x, extra_tokens)  # patch linear embedding
        # print('nc post', nc)

        # add the [CLS] token to the embed patch tokens
        # print('input x', x.shape)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # print('Shape of x prepare:', x.shape)
        # add positional encoding to each token
        # print('w, h, n', w, h, nc)
        # print('self.interpolate_pos_encoding(x, w, h, nc)', self.interpolate_pos_encoding(x, w, h, nc).shape)
        x = x + self.interpolate_pos_encoding(x, w, h, nc)

        return self.pos_drop(x)

    def forward(self, x, extra_tokens={}):
        # print(x.shape)
        x = self.prepare_tokens(x, extra_tokens)
        # print('Shape of x after prepare_tokens:', x.shape)
        for blk in self.blocks:
            x = blk(x)
            # print('Shape of x in blocks loop:', x.shape)
        x = self.norm(x)
        # print('Shape of x after norm:', x.shape)
#         return x[:, 0]
        return self.head(x[:, 0])
    
    def get_last_selfattention(self, x, extra_tokens={}):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, extra_tokens, n=1):
        x = self.prepare_tokens(x, extra_tokens)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def hcs_channelvit_tiny(patch_size=16, **kwargs):
    model = ChannelVisionTransformer(
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def hcs_channelvit_small(patch_size=16, **kwargs):
    model = ChannelVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def hcs_channelvit_base(patch_size=16, **kwargs):
    model = ChannelVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model

def create_extra_tokens(x):
    channels = list(range(x.shape[1]))  # 假设 x 的形状是 [batch_size, channels,height, width]
    return {"channels": torch.tensor(channels).to(device)}
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()
        
        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type Tensor, float, float, float, float) conda env update -f environment.yml-> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def hierarchical_channel_sampling(x, level1_ratio=0.5, level2_ratio=0.5):
#     Perform two-level hierarchical channel sampling.
    
#     Parameters:
#     - x: input tensor of shape (batch_size, num_channels, height, width)
#     - level1_ratio: the ratio of channels to keep in the first level
#     - level2_ratio: the ratio of channels to keep in the second level after level1 sampling

#     Returns:
#     - A tensor with the same shape as x with some channels set to zero
    batch_size, num_channels, height, width = x.size()
    # First level sampling
    level1_sampled_channels = int(num_channels * level1_ratio)
    level1_channels = np.random.choice(num_channels, level1_sampled_channels, replace=False)

    # Second level sampling
    level2_sampled_channels = int(level1_sampled_channels * level2_ratio)
    level2_channels = np.random.choice(level1_channels, level2_sampled_channels, replace=False)

    # Create a mask for selected channels
    mask = torch.zeros_like(x)
    mask[:, level2_channels, :, :] = 1

    # Apply the mask to input
    x_hcs = x * mask
    return x_hcs

def _loop_with_hcs(dataloader, net, loss_fn, optimizer, device):
    """
    Perform a training loop with hierarchical channel sampling integrated.

    Parameters:
    - dataloader: DataLoader providing the training batches
    - net: the neural network model
    - loss_fn: loss function
    - optimizer: optimizer
    - device: the device to run the calculations on (e.g., "cuda" or "cpu")
    """
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Perform hierarchical channel sampling on the input
        X_hcs = hierarchical_channel_sampling(X)
        X_hcs, y = X_hcs.to(device), y.to(device)

        # Compute prediction and loss
        pred = net(X_hcs)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        # print('Shape of x after fc1:', x.shape)
        x = self.act(x)
        # print('Shape of x after activation:', x.shape)
        x = self.drop(x)
        # print('Shape of x after dropout:', x.shape)
        x = self.fc2(x)
        # print('Shape of x after fc2:', x.shape)
        x = self.drop(x)
        # print('Shape of x after final dropout:', x.shape)
        return x

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # print('Shape of input x:', x.shape)
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        # print('Shape of qkv:', qkv.shape)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # print('Shape of q:', q.shape)
        # print('Shape of k:', k.shape)
        # print('Shape of v:', v.shape)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # print('Shape of attn:', attn.shape)
        attn = attn.softmax(dim=-1)
        # print('Shape of attn after softmax:', attn.shape)
        attn = self.attn_drop(attn)
        # print('Shape of attn after dropout:', attn.shape)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # print('Shape of x after attention:', x.shape)
        x = self.proj(x)
        # print('Shape of x after projection:', x.shape)
        x = self.proj_drop(x)
        # print('Shape of x after projection dropout:', x.shape)

        return x, attn



class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
#         # print('Shape of y:', y.shape)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
#         # print('Shape of x after drop_path(y):', x.shape)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
#         # print('Shape of x after mlp:', x.shape)
        return x
    
class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model
def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

def build_client_net(num_client:int, device):
    client_net = {}
    files = os.listdir()
    for i in range(num_client):
        file = 'model {}.pth'.format(i)
        if file in files:
            client_net[i] = torch.load(file).to(device)
        else:
            client_net[i] = ChannelVisionTransformer(
                img_size=[32],
                num_classes=10,
                enable_sample=False,
                patch_size=8,
                embed_dim=192,
                depth=12,
                num_heads=3,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                enable_hcs=True,  # 启用 HCS
                channel_dropout_rate=0.01,  # 设置 Channel Dropout 概率
            ).to(device)
    return client_net

