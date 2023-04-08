#!/usr/bin/env python

"""
preprocess.py: Handles preprocessing of molecular data in SDF format and
               conversion into fingerprints, strings, and graphs.
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

from rdkit import RDLogger
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.rdmolfiles  import SDMolSupplier
from rdkit.Chem.rdmolfiles  import ForwardSDMolSupplier
from rdkit.Chem.rdmolfiles  import MolToSmiles
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.AllChem     import GetMorganFingerprintAsBitVect as bin_ECFP

#==============================================================================#
#                                  DATA  LOADING                               #
#==============================================================================#

def mols_from_sdf(file_path: str) -> Generator:
    """
    Takes .sdf files (compressed or otherwise) and returns a 
    generator over the molecules, encoded as RDKit 'Mol' objects.
    """
    
    # Handle zipped files using ForwardSDMolSupplier
    if file_path.endswith('.gz'):
        file  = gzip.open(file_path)
        suppl = ForwardSDMolSupplier(file)
       
    # Otherwise, just use normal Supplier
    else:
        file  = open(file_path)
        suppl = SDMolSupplier(file)
    
    # Return Generator of Mol objects
    return suppl


def mols_from_str(file_path: str) -> Generator:
    """
    
    """
    
    # Check file extension for formatting
    if file_path.endswith('.smi'):
        decode = lambda x: x
    elif file_path.endswith('.selfies'):
        decode = lambda x: slf.decode(x)
    else:
        raise
    
    # Instantiate Mol objects from SMILES strings
    with open(smi_path, 'r') as file:
        lines = [line.strip('\n') for line in file.readlines()]
        return (MolFromSmiles(decode(line)) for line in lines)
    

#==============================================================================#
#                            CONVERT TO REPRESENTATIONS                        #
#==============================================================================#

def mols_to_fps(mols: Iterable[Chem.rdchem.Mol], 
                save_path: str, dim: int=1024):
    """
    
    """
    
    # Iterate over all molecules and compute fingerprints
    fp_list = []
    for mol in tqdm(mols):
        fp_list.append(bin_ECFP(mol, radius=2, nBits=dim, useFeatures=True))
        del mol
    print("Finished iterating!")
    
    # Concatenate fingerprints into array
    fp_array = np.fromiter(fp_list, dtype=np.dtype((bool, dim)))
    print("Finished array conversion!")
    
    # Write array to disk
    np.save(save_path, fp_array)
    print("Finished saving to disk!")
    

def mols_to_strings(mols      : Iterable[Chem.rdchem.Mol], 
                    save_path : str):
    """
    
    """
    
    # Open SMILES/SELFIES output files
    with open(save_path + '.smi', 'w') as smi_file:
        with open(save_path + '.selfies', 'w') as slf_file:
    
            for mol in tqdm(mols):
                
                # Convert to SMILES/SELFIES format
                smi_str = MolToSmiles(mol)
                try:
                    slf_str = slf.encoder(smi_str.replace('*', ''))
                except slf.EncoderError:
                    pass
                
                # Write to output file
                smi_file.write(smi_str + '\n')
                slf_file.write(slf_str + '\n')
                
                del mol
                
    print("Finished iterating!")


def mols_to_graphs(mols     : Iterable[Chem.rdchem.Mol],
                   save_path: str=None,
                   featurizer=None,
                  ) -> None:
    """
    
    """
    
    pass


#==============================================================================#
#                           MAIN METHOD (preprocess.py)                        #
#==============================================================================#

RDLogger.DisableLog('rdApp.warning')

parser = argparse.ArgumentParser()
parser.add_argument("--mol_path", help=".sdf file path with input molecules")

template = "file path (no file extension) to write {} to"
parser.add_argument("--fp_path",    help=template.format('fingerprints'))
parser.add_argument("--str_path",   help=template.format('strings'))
parser.add_argument("--graph_path", help=template.format('graphs'))

parser.add_argument("--fp_dim",     help="dimension of ECFP vectors", type=int)
parser.add_argument("--small_only", help="only consider small molecules",
                    action="store_true")

def main():
    
    # Parse arguments from CLI
    args = parser.parse_args()
    
    # Read molecules from input file
    mols = mols_from_sdf(args.mol_path)
    
    # Filter for sufficiently small molecules
    if args.small_only:
        mols = (m for m in mols if ExactMolWt(m) <= 1000)
    
    # Encode and store molecules in each representation
    if args.fp_path:
        mols_to_fps(mols, args.fp_path, args.fp_dim)
    if args.str_path:
        mols_to_strings(mols, args.str_path)
    if args.graph_path:
        mols_to_graphs(mols, args.graph_path)
    
    
if __name__ == "__main__":
    main()
