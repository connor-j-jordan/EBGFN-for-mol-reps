#!/usr/bin/env python

"""

"""

__author__ = "Connor Jordan"

#==============================================================================#
#                                   DEPENDENCIES                               #
#==============================================================================#

import numpy as np
import torch
import torch.nn as nn
import os, sys
import copy
import time
import random
import ipdb
from tqdm import tqdm
import argparse
import network

from gfn import get_GFlowNet

#==============================================================================#
#                                MODEL DEFINITIONS                             #
#==============================================================================#
        
class FingerprintEBM(nn.Module):
    
    def __init__(self, net, mean=None):
        super().__init__()
        
        self.net = net
        if mean is None:
            self.mean = None
        else:
            self.mean = nn.Parameter(mean, requires_grad=False)
            self.base_dist = torch.distributions.Bernoulli(probs=self.mean)

    def forward(self, x):
        if self.mean is None:
            bd = 0.
        else:
            bd = self.base_dist.log_prob(x).sum(-1)

        logp = self.net(x).squeeze()
        return logp + bd
    
    
    
class StringEBM(nn.Module):
    
    
    
    
    
class GraphEBM(nn.Module):

#==============================================================================#
#                                 MAIN METHOD (gfn.py)                         #
#==============================================================================#

parser = argparse.ArgumentParser()


def main():
        

                
                
if __name__ == "__main__":
    main()