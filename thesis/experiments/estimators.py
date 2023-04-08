#!/usr/bin/env python

"""

"""

__author__ = "Connor Jordan"

#==============================================================================#
#                                   DEPENDENCIES                               #
#==============================================================================#

import copy
import gzip
import argparse
from tqdm import tqdm
from typing import List, Generator, Iterable

import numpy   as np
import pandas  as pd
import selfies as slf

from scipy.special import digamma

#==============================================================================#
#                               ENTROPY ESTIMATORS                             #
#==============================================================================#

def plugin_estimator(probs: Iterable) -> float:
    return -sum(probs * np.log2(probs))

def grassberger(probs: Iterable) -> float:
    """
    
    """
    
    # Define Grassberger function G
    def G(h: float) -> float:
        diff = digamma((h+1)/2) - digamma(h/2)
        return digamma(h) + 0.5 * ((-1)**h) * diff
    
    n = 0
    total = 0
    for prob in probs:
        
        # Count total number of observations
        n += 1
        
        # Sum up expected value
        total += prob * G(prob)
        
    return np.log2(n) - (total / n)

#==============================================================================#
#                             GENERATIVE ESTIMATORS                            #
#==============================================================================#
    
def GFN_estimator(GFN, entropic_GFN) -> float:
    """
    
    """
    
    