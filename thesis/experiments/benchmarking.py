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
from typing import List, Generator, Iterable, Callable

import numpy  as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

from rdkit.Chem import AllChem as Chem

#==============================================================================#
#                              BIT VECTOR BENCHMARKS                           #
#==============================================================================#

def sample_vecs(num_vecs: int, dim: int, dist: Callable) -> np.ndarray:
    """
    
    """



