#!/usr/bin/env python

"""

"""

__author__ = "Connor Jordan"

#==============================================================================#
#                                   DEPENDENCIES                               #
#==============================================================================#

import numpy   as np
import pandas  as pd
import selfies as slf

from deepchem.feat.smiles_tokenizer import SmilesTokenizer

#==============================================================================#
#                                   TOKENIZATION                               #
#==============================================================================#

def tokenize_smiles(smi_str: str) -> list:
    
    tokenizer = SmilesTokenizer("smiles_vocab.txt")
    return list(tokenizer.encode(smi_str))


def tokenize_selfies(slf_str: str) -> list:
    
    return list(slf.split_selfies(slf_str))


def max_length(file_path: str) -> int:
    """
    Given the path to a .smiles file containing SMILES strings,
    or a .selfies file containing SELFIES strings, tokenizes 
    them and returns the number of tokens in the longest string.
    """
    
    # Check file extension for formatting
    if file_path.endswith('.smi'):
        tokenize = tokenize_smiles
    elif file_path.endswith('.selfies'):
        tokenize = tokenize_selfies
    else:
        raise
    
    max_len = 0
    with open(file_path, 'r') as file:

        # Tokenize each SELFIES string
        tokenized = [tokenize(line.strip('\n')) for line in file.readlines()]
        lengths   = [len(list(tl)) for tl in tokenized]

        return max(lengths)
    
#==============================================================================#
#                                 ONE-HOT ENCODING                             #
#==============================================================================#
    
def encode_string() -> np.ndarray:
    """
    
    """
    
    