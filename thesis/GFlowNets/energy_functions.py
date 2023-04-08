#!/usr/bin/env python

"""


Adapted from GraphEBM (Liu et al., 2021)
"""

#==============================================================================#
#                                   DEPENDENCIES                               #
#==============================================================================#

import torch
from torch import nn
from torch.nn import functional as F

from networks import Swish




    
    
#==============================================================================#
#                                  ENERGY FUNCTIONS                            #
#==============================================================================#

class FPEnergyFunction(nn.Module):
    """
    
    """
    
    def __init__(self):
        super(FPEnergyFunction, self).__init__()
        
        
    def forward(self, ):
        
        
        
        
class StringEnergyFunction(nn.Module):
    """
    
    """
    
    def __init__(self):
        super(StringEnergyFunction, self).__init__()
        
        
    def forward(self, ):


class GraphEnergyFunction(nn.Module):
    """
    
    """

    def __init__(self, n_atom_type, hidden, num_edge_type=4, depth=2, add_self=False, dropout=0):
        super(GraphEnergyFunction, self).__init__()

        self.depth = depth
        self.graphconv1 = GraphConv(n_atom_type, hidden, num_edge_type, std=1, bound=False, add_self=add_self)
        self.graphconv = nn.ModuleList(GraphConv(hidden, hidden, num_edge_type, std=1e-10, add_self=add_self) for i in range(self.depth))
        self.dropout = dropout
        self.linear = nn.Linear(hidden, 1)
            
        
    def forward(self, adj, h):
        h = h.permute(0, 2, 1)
        out = self.graphconv1(adj, h)
            
        out = F.dropout(out, p=self.dropout, training=self.training)
        
        if self.swish:
            out = swish(out)
        else:
            out = F.leaky_relu(out, negative_slope=0.2)


        for i in range(self.depth):
            out = self.graphconv[i](adj, out)
                
            out = F.dropout(out, p=self.dropout, training=self.training)
            if self.swish:
                out = swish(out)
            else:
                out = F.relu(out)
        
        out = out.sum(1) # (batchsize, node, ch) --> (batchsize, ch)
        out = self.linear(out)
        
        return out # Energy value (batchsize, 1)