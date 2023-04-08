#!/usr/bin/env python

"""

"""

__author__ = "Connor Jordan"

#==============================================================================#
#                                  DEPENDENCIES                                #
#==============================================================================#

import ipdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#==============================================================================#
#                              ACTIVATION FUNCTIONS                            #
#==============================================================================#

class Swish(nn.Module):
    
    def __init__(self, beta: float=1):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

#==============================================================================#
#                              FINGERPRINT NETWORKS                            #
#==============================================================================#

def bare_MLP(l, act=nn.LeakyReLU(), tail=[], with_bn=False):
    """
    makes an MLP with no top layer activation
    """
    
    net = nn.Sequential(*(sum(
        [[nn.Linear(i, o)] + (([nn.BatchNorm1d(o), act] if with_bn else [act]) if n < len(l) - 2 else [])
         for n, (i, o) in enumerate(zip(l, l[1:]))], []
    ) + tail))
    return net


def EBM_MLP(nin, nint=256, nout=1):
    return nn.Sequential(
        nn.Linear(nin, nint),
        Swish(),
        nn.Linear(nint, nint),
        Swish(),
        nn.Linear(nint, nint),
        Swish(),
        nn.Linear(nint, nout),
    )

#==============================================================================#
#                                STRING  NETWORKS                              #
#==============================================================================#





#==============================================================================#
#                                 GRAPH NETWORKS                               #
#==============================================================================#

class SpectralNorm:
    """
    
    """
    
    def __init__(self, name, bound=False):
        self.name = name
        self.bound = bound

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)

        with torch.no_grad():
            v = weight_mat.t() @ u
            v = v / v.norm()
            u = weight_mat @ v
            u = u / u.norm()

        sigma = u @ weight_mat @ v

        if self.bound:
            weight_sn = weight / (sigma + 1e-6) * torch.clamp(sigma, max=1)

        else:
            weight_sn = weight / sigma

        return weight_sn, u

    @staticmethod
    def apply(module, name, bound):
        fn = SpectralNorm(name, bound)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', weight)
        input_size = weight.size(0)
        u = weight.new_empty(input_size).normal_()
        module.register_buffer(name, weight)
        module.register_buffer(name + '_u', u)

        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight_sn, u = self.compute_weight(module)
        setattr(module, self.name, weight_sn)
        setattr(module, self.name + '_u', u)


def spectral_norm(module, init=True, std=1, bound=False):
    if init:
        nn.init.normal_(module.weight, 0, std)

    if hasattr(module, 'bias') and module.bias is not None:
        module.bias.data.zero_()

    SpectralNorm.apply(module, 'weight', bound=bound)

    return module


class GraphConv(nn.Module):
    """
    
    """

    def __init__(self, in_channels, out_channels, num_edge_type, std, bound=True, add_self=False):
        super(GraphConv, self).__init__()
        
        self.add_self = add_self
        if self.add_self:
            self.linear_node = spectral_norm(nn.Linear(in_channels, out_channels), std=std, bound=bound)
        self.linear_edge = spectral_norm(nn.Linear(in_channels, out_channels * num_edge_type), std=std, bound=bound)
        self.num_edge_type = num_edge_type
        self.in_ch = in_channels
        self.out_ch = out_channels

    def forward(self, adj, h):
        mb, node, _ = h.shape 
        if self.add_self:
            h_node = self.linear_node(h) 
        m = self.linear_edge(h)
        m = m.reshape(mb, node, self.out_ch, self.num_edge_type) 
        m = m.permute(0, 3, 1, 2) # m: (batchsize, edge_type, node, ch)
        hr = torch.matmul(adj, m)  # hr: (batchsize, edge_type, node, ch)
        hr = hr.sum(dim=1)   # hr: (batchsize, node, ch)
        if self.add_self:
            return hr+h_node  #
        else:
            return hr