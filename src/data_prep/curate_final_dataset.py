"""
curate_final_dataset.py — Final curation of the stability constant dataset.

Curation steps:
  1. Fix metal_type extraction (previous version misidentified N, Se, Cl etc as metals)
  2. Remove logK outliers (>30 likely multi-step beta, <-3 likely erroneous)
  3. Deduplicate: same SMILES → median logK
  4. Remove very small molecules (<3 atoms)
  5. Remove metals with <5 entries
  6. RDKit re-validation, donor extraction

Output: stability_constants_dative_clean.csv
"""

import os
import re
import logging
from collections import Counter

import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

# Complete set of metals that can appear as the central coordination atom.
# Includes transition metals, lanthanides, actinides, and main-group metals
# that form coordination complexes.
COORDINATION_METALS = {
    # 3d transition metals
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    # 4d
    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    # 5d
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    # Lanthanides (4f)
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    # Actinides (5f)
    'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf',
    'Es', 'Fm', 'Md', 'No',
    # Main-group metals that form coordination complexes
    'Al', 'Ga', 'In', 'Tl', 'Sn', 'Pb', 'Bi',
    # Alkaline earth (form complexes with chelators)
    'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra',
    # Alkali (weak complexes but measured)
    'Li', 'Na', 'K', 'Rb', 'Cs',
}

# These are NOT metals — they are non-metal elements that appear in
# SMILES brackets (e.g. [N+], [O-], [Se], [S]) and can be misidentified
NOT_METALS = {
    'H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I',
    'Se', 'Te', 'As', 'Si', 'Ge', 'B', 'At',
}

DONOR_ELEMENTS = {'S', 'N', 'O', 'P', 'Cl', 'Br', 'I'}


def extract_metal_from_mol(mol):
    """Extract the coordination metal from an RDKit mol object.

    Finds atoms that are metals AND have dative bonds.
    Falls back to any metal atom if no dative bonds found.
    """
    if mol is None:
        return None

    # First: find atoms with dative bonds that are metals
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DATIVE:
            begin = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
            end = mol.GetAtomWithIdx(bond.GetEndAtomIdx())
            # Metal is typically the acceptor of the dative bond
            if end.GetSymbol() in COORDINATION_METALS:
                return end.GetSymbol()
            if begin.GetSymbol() in COORDINATION_METALS:
                return begin.GetSymbol()

    # Fallback: any metal atom in the molecule
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in COORDINATION_METALS:
            return atom.GetSymbol()

    return None


def analyze_smiles(smi):
    """Extract metadata from a dative SMILES."""
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None

    n_atoms = mol.GetNumAtoms()
    metal = extract_metal_from_mol(mol)
    n_dative = sum(1 for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DATIVE)

    # Identify donor elements (non-metal atoms with dative bond to metal)
    donor_atoms = set()
    for b in mol.GetBonds():
        if b.GetBondType() == Chem.BondType.DATIVE:
            begin_sym = mol.GetAtomWithIdx(b.GetBeginAtomIdx()).GetSymbol()
            end_sym = mol.GetAtomWithIdx(b.GetEndAtomIdx()).GetSymbol()
            if begin_sym not in COORDINATION_METALS:
                donor_atoms.add(begin_sym)
            if end_sym not in COORDINATION_METALS:
                donor_atoms.add(end_sym)

    donor_str = '+'.join(sorted(donor_atoms)) if donor_atoms else ''

    return {
        'n_atoms': n_atoms,
        'metal_type': metal,
        'n_dative_bonds': n_dative,
        'donor_elements': donor_str,
    }


def main():
    input_path = os.path.join(DATA_DIR, 'stability_constants_dative_master.csv')
    df = pd.read_csv(input_path)
    n_start = len(df)
    logger.info(f"Input: {n_start} entries")

    # ================================================================
    # Step 1: Remove logK outliers
    # ================================================================
    before = len(df)
    df = df[(df['logK1'] >= -3) & (df['logK1'] <= 30)].reset_index(drop=True)
    logger.info(f"Step 1 — logK filter [-3, 30]: {before} -> {len(df)} (removed {before-len(df)})")

    # ================================================================
    # Step 2: Deduplicate — median logK for same SMILES
    # ================================================================
    before = len(df)
    df = df.groupby('smiles').agg(
        logK1=('logK1', 'median'),
        n_measurements=('logK1', 'count'),
    ).reset_index()
    logger.info(f"Step 2 — Deduplicate: {before} -> {len(df)} unique SMILES")

    # ================================================================
    # Step 3: Re-analyze ALL SMILES with RDKit (fixes metal misidentification)
    # ================================================================
    logger.info("Step 3 — Re-analyzing with RDKit (fixing metal extraction)...")
    meta = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing"):
        result = analyze_smiles(row['smiles'])
        meta.append(result if result else {
            'n_atoms': 0, 'metal_type': None, 'n_dative_bonds': 0, 'donor_elements': ''
        })

    meta_df = pd.DataFrame(meta)
    df['metal_type'] = meta_df['metal_type']
    df['n_atoms'] = meta_df['n_atoms']
    df['n_dative_bonds'] = meta_df['n_dative_bonds']
    df['donor_elements'] = meta_df['donor_elements']

    # Remove entries where no metal was found
    before = len(df)
    df = df[df['metal_type'].notna()].reset_index(drop=True)
    logger.info(f"Step 3 — Remove no-metal: {before} -> {len(df)}")

    # ================================================================
    # Step 4: Remove very small molecules
    # ================================================================
    before = len(df)
    df = df[df['n_atoms'] >= 3].reset_index(drop=True)
    logger.info(f"Step 4 — Remove <3 atoms: {before} -> {len(df)}")

    # ================================================================
    # Step 5: Remove metals with <5 entries
    # ================================================================
    metal_counts = df['metal_type'].value_counts()
    rare = metal_counts[metal_counts < 5].index.tolist()
    before = len(df)
    df = df[~df['metal_type'].isin(rare)].reset_index(drop=True)
    logger.info(f"Step 5 — Remove rare metals (<5): {before} -> {len(df)}")
    if rare:
        logger.info(f"  Removed: {rare}")

    # ================================================================
    # Step 6: Final cleanup
    # ================================================================
    df['logK1'] = df['logK1'].round(3)
    df = df[['smiles', 'logK1', 'metal_type', 'n_atoms', 'n_dative_bonds',
             'donor_elements', 'n_measurements']].copy()

    # Save
    output_path = os.path.join(DATA_DIR, 'stability_constants_dative_clean.csv')
    df.to_csv(output_path, index=False)

    # ================================================================
    # Summary
    # ================================================================
    logger.info(f"\n{'='*60}")
    logger.info("FINAL CURATED DATASET")
    logger.info(f"{'='*60}")
    logger.info(f"Entries:       {len(df)} (from {n_start})")
    logger.info(f"Unique metals: {df['metal_type'].nunique()}")
    logger.info(f"logK1:         {df['logK1'].min():.2f} to {df['logK1'].max():.2f} (mean {df['logK1'].mean():.2f})")
    logger.info(f"Atoms/complex: {df['n_atoms'].min()}-{df['n_atoms'].max()} (mean {df['n_atoms'].mean():.1f})")

    logger.info(f"\nTop 20 metals:")
    for m, c in df['metal_type'].value_counts().head(20).items():
        logger.info(f"  {m:4s}: {c:5d}")

    logger.info(f"\nDonor patterns:")
    for d, c in df['donor_elements'].value_counts().head(10).items():
        logger.info(f"  {d:10s}: {c}")

    flotation = {'Cu', 'Fe', 'Zn', 'Pb', 'Ni', 'Co', 'Mn'}
    flot = df[df['metal_type'].isin(flotation)]
    logger.info(f"\nFlotation metals: {len(flot)}")
    for m in sorted(flotation):
        n = (flot['metal_type'] == m).sum()
        if n > 0:
            logger.info(f"  {m}: {n}")

    lanthanides = {'La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu'}
    lant = df[df['metal_type'].isin(lanthanides)]
    logger.info(f"\nLanthanides (REE): {len(lant)}")

    actinides_set = {'Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm'}
    act = df[df['metal_type'].isin(actinides_set)]
    logger.info(f"Actinides: {len(act)}")

    logger.info(f"\nSaved: {output_path}")


if __name__ == '__main__':
    main()
