#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")#global epoch
    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")#client number
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=20, help="the number of local epochs: E")#local epoch
    parser.add_argument('--local_bs', type=int, default=32, help="local batch size: B")#-------------------这个是啥
    parser.add_argument('--bs', type=int, default=64, help="test batch size")#batch size
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum (default: 0.5)")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="SGD weight decay (default: 1e-4)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments
    parser.add_argument('--model', type=str, default='vit_mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='1,16,16',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')
    

    #vit arguements
    parser.add_argument('--img_size', type=int, nargs='+', default=[32], help="input image size")
    parser.add_argument('--patch_size', type=int, default=8, help="patch size for ViT")
    parser.add_argument('--embed_dim', type=int, default=192, help="embedding dimension for ViT")
    parser.add_argument('--depth', type=int, default=12, help="number of layers in ViT")
    parser.add_argument('--num_heads', type=int, default=3, help="number of attention heads in ViT")
    parser.add_argument('--mlp_ratio', type=int, default=4, help="ratio of the hidden dimension in MLP blocks")
    parser.add_argument('--qkv_bias', type=eval, default=True, help="enable bias for QKV in ViT")
    parser.add_argument('--norm_layer', type=str, default='LayerNorm', help="normalization layer type")
            # Additional args for the unspecified parameters
    parser.add_argument('--enable_sample', type=eval, default=False, help="enable sampling in ViT")
    parser.add_argument('--enable_hcs', type=eval, default=True, help="enable Hybrid Channel-Spatial attention in ViT")
    parser.add_argument('--channel_dropout_rate', type=float, default=0.65, help="dropout rate for channels in ViT")
    parser.add_argument('--dropout_rate', type=float, default=0.3, help="dropout rate for channels in ViT")
    # Additional arguments for AdamW optimizer
    parser.add_argument('--betas', type=eval, default=(0.9, 0.999), help="betas for AdamW optimizer")
    parser.add_argument('--eps', type=float, default=1e-8, help="epsilon for AdamW optimizer")

    args = parser.parse_args()
    return args