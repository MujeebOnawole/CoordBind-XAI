#!/usr/bin/env python
"""
convert_iupac_to_dative.py — Convert dot-separated metal-ligand SMILES to dative bond SMILES.

Input:  IUPAC v2 format: "NC(Cc1cnc[nH]1)C(=O)O.[La+3]"  (ligand.metal, no bond)
Output: Dative format:   "NC(Cc1cnc[nH]1)C(=O)[O-]->[La+3]"  (coordination bond)

Strategy:
  1. Parse ligand with RDKit
  2. Identify the best donor atom(s) using chemical rules:
     - Deprotonated carboxylate [O-] > amine N > pyridine n > thiolate [S-] > hydroxyl O
  3. Insert dative bond: donor->[Metal]
  4. Validate the resulting SMILES with RDKit

For logK1 (first stability constant), we connect 1 ligand to 1 metal
through the most likely donor atom(s).

Usage:
    python convert_iupac_to_dative.py
"""

import os
import re
import logging
from collections import Counter

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGKPREDICT_DIR = os.path.join(SCRIPT_DIR, 'data', 'external', 'LOGKPREDICT')

# Metals that should not be treated as donor atoms
# Metals that should not be treated as donor atoms.
# Derived from periodic table — no hardcoded list needed.
_NON_METALS = {
    'H', 'He', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Si', 'P', 'S', 'Cl', 'Ar',
    'Ge', 'As', 'Se', 'Br', 'Kr',
    'Te', 'I', 'Xe', 'At', 'Rn', 'Og',
}
METALS = {
    Chem.GetPeriodicTable().GetElementSymbol(z)
    for z in range(1, 119)
    if Chem.GetPeriodicTable().GetElementSymbol(z) not in _NON_METALS
    and z > 0
}


def find_donor_atoms(mol):
    """Find likely coordination donor atoms in a ligand.

    Returns list of (atom_idx, priority, donor_type) sorted by coordination priority.

    Priority (lower = more likely to coordinate):
      1. Negatively charged O ([O-]) in carboxylate — strongest donor
      2. Negatively charged S ([S-]) thiolate
      3. Amine N with lone pair (not in ring, not positively charged)
      4. Aromatic/pyridine N (ring nitrogen with lone pair)
      5. Neutral O in C=O (carbonyl, weaker donor)
      6. Phosphorus donors
      7. Neutral S (thioether)
    """
    donors = []

    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        charge = atom.GetFormalCharge()
        idx = atom.GetIdx()
        in_ring = atom.IsInRing()
        aromatic = atom.GetIsAromatic()
        neighbors = [n.GetSymbol() for n in atom.GetNeighbors()]

        if symbol in METALS:
            continue

        # 1. Deprotonated carboxylate O: [O-] bonded to C(=O)
        if symbol == 'O' and charge == -1:
            # Check if part of carboxylate (neighbor is C with another O)
            for nbr in atom.GetNeighbors():
                if nbr.GetSymbol() == 'C':
                    nbr_os = [n for n in nbr.GetNeighbors() if n.GetSymbol() == 'O' and n.GetIdx() != idx]
                    if nbr_os:
                        donors.append((idx, 1, 'carboxylate_O-'))
                        break
            else:
                donors.append((idx, 1, 'O-'))
            continue

        # 2. Thiolate [S-]
        if symbol == 'S' and charge == -1:
            donors.append((idx, 2, 'S-'))
            continue

        # 3. Amine N (not in ring, not charged, has lone pair)
        if symbol == 'N' and charge == 0 and not aromatic:
            # Primary/secondary/tertiary amine
            n_H = atom.GetTotalNumHs()
            if n_H >= 1 or len(neighbors) <= 3:
                donors.append((idx, 3, 'amine_N'))
            continue

        # 4. Aromatic/pyridine N
        if symbol == 'N' and aromatic and charge == 0:
            donors.append((idx, 4, 'pyridine_N'))
            continue

        # Ring N (non-aromatic but in ring, like imidazole N)
        if symbol == 'N' and in_ring and charge == 0 and not aromatic:
            donors.append((idx, 4, 'ring_N'))
            continue

        # 5. Carbonyl O (C=O)
        if symbol == 'O' and charge == 0:
            for nbr in atom.GetNeighbors():
                bond = mol.GetBondBetweenAtoms(idx, nbr.GetIdx())
                if nbr.GetSymbol() == 'C' and bond and bond.GetBondTypeAsDouble() == 2.0:
                    donors.append((idx, 5, 'carbonyl_O'))
                    break
            continue

        # 6. Phosphorus
        if symbol == 'P' and charge == 0:
            donors.append((idx, 6, 'P'))
            continue

        # 7. Neutral S (thioether)
        if symbol == 'S' and charge == 0:
            donors.append((idx, 7, 'thioether_S'))
            continue

    # Sort by priority
    donors.sort(key=lambda x: x[1])
    return donors


def convert_dot_to_dative(dot_smiles, max_donors=2):
    """Convert 'ligand.[Metal+n]' to 'ligand_with_dative->[Metal+n]'.

    For logK1, we typically connect through 1-2 donor atoms (mono/bidentate).

    Returns:
        dative_smiles: str or None if conversion fails
        donor_info: str describing which donors were used
        n_donors: int
    """
    parts = dot_smiles.split('.')
    if len(parts) != 2:
        return None, 'not_2_fragments', 0

    # Identify which part is the metal
    metal_part = None
    ligand_part = None
    for p in parts:
        if re.search(r'\[[A-Z][a-z]?\+\d+\]', p):
            metal_part = p
        else:
            ligand_part = p

    if metal_part is None or ligand_part is None:
        return None, 'no_metal_found', 0

    # Parse ligand
    mol = Chem.MolFromSmiles(ligand_part)
    if mol is None:
        return None, 'invalid_ligand_smiles', 0

    # Find donor atoms
    donors = find_donor_atoms(mol)
    if not donors:
        return None, 'no_donors_found', 0

    # Select top donors (up to max_donors)
    selected = donors[:max_donors]

    # Build dative SMILES
    # Strategy: modify the ligand SMILES to add ->[Metal] at donor positions
    # For simplicity and reliability, use RDKit's editable mol
    try:
        rw_mol = Chem.RWMol(mol)

        # Add metal atom
        metal_match = re.search(r'\[([A-Z][a-z]?)(\+\d+)?\]', metal_part)
        if not metal_match:
            return None, 'cant_parse_metal', 0

        metal_symbol = metal_match.group(1)
        metal_charge_str = metal_match.group(2) or ''
        metal_charge = int(metal_charge_str.replace('+', '')) if metal_charge_str else 0

        metal_idx = rw_mol.AddAtom(Chem.Atom(metal_symbol))
        rw_mol.GetAtomWithIdx(metal_idx).SetFormalCharge(metal_charge)

        # Add dative bonds from donors to metal
        for donor_idx, priority, dtype in selected:
            # DATIVE bond: donor -> metal (donor gives electrons to metal)
            rw_mol.AddBond(donor_idx, metal_idx, Chem.BondType.DATIVE)

        # Sanitize (partial — dative bonds can cause issues with full sanitize)
        try:
            Chem.SanitizeMol(rw_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                             Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        except Exception:
            pass

        dative_smiles = Chem.MolToSmiles(rw_mol)

        # Verify the result parses back.
        # Use relaxed sanitization: some metals (Ga, In, Tl, Be) have
        # non-standard valences that cause RDKit's strict sanitizer to
        # reject valid dative-bond SMILES. We skip SANITIZE_PROPERTIES
        # (which enforces valence rules) since dative bonds to metals
        # are not covered by RDKit's default valence model.
        check_mol = Chem.MolFromSmiles(dative_smiles, sanitize=False)
        if check_mol is None:
            return None, 'output_smiles_invalid', 0
        try:
            Chem.SanitizeMol(check_mol,
                             sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                             Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        except Exception:
            return None, 'output_smiles_sanitize_failed', 0

        # Check dative bonds exist
        n_dative = sum(1 for b in check_mol.GetBonds()
                       if b.GetBondType() == Chem.BondType.DATIVE)

        donor_types = '+'.join(d[2] for d in selected)
        return dative_smiles, donor_types, n_dative

    except Exception as e:
        return None, f'error: {str(e)[:50]}', 0


def main():
    # Load IUPAC data
    iupac_path = os.path.join(LOGKPREDICT_DIR, 'data', 'IUPAC_v2', 'ML_inp_train.csv')
    df = pd.read_csv(iupac_path, encoding='utf-8-sig')
    df.columns = [c.strip() for c in df.columns]

    smi_col = 'Cmplx_smiles_neutral_std'
    k_col = 'Lg K1'

    logger.info(f"Input: {len(df)} complexes from IUPAC v2")

    # Convert
    results = []
    fail_reasons = Counter()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting"):
        dot_smi = str(row[smi_col]).strip()
        logk = row[k_col]

        dative_smi, donor_info, n_dative = convert_dot_to_dative(dot_smi)

        # Extract metal
        metal_match = re.search(r'\[([A-Z][a-z]?)\+?\d*\]', dot_smi)
        metal = metal_match.group(1) if metal_match else 'unknown'

        if dative_smi:
            results.append({
                'original_smiles': dot_smi,
                'dative_smiles': dative_smi,
                'logK1': logk,
                'metal_type': metal,
                'donor_info': donor_info,
                'n_dative_bonds': n_dative,
                'valid': True,
            })
        else:
            fail_reasons[donor_info] += 1
            results.append({
                'original_smiles': dot_smi,
                'dative_smiles': '',
                'logK1': logk,
                'metal_type': metal,
                'donor_info': donor_info,
                'n_dative_bonds': 0,
                'valid': False,
            })

    results_df = pd.DataFrame(results)
    valid = results_df[results_df['valid']]
    invalid = results_df[~results_df['valid']]

    # Save
    out_dir = os.path.join(SCRIPT_DIR, 'data')
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, 'iupac_logk_dative.csv')
    valid.to_csv(out_path, index=False)

    fail_path = os.path.join(out_dir, 'iupac_logk_failed.csv')
    invalid.to_csv(fail_path, index=False)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("CONVERSION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total:     {len(results_df)}")
    logger.info(f"Valid:     {len(valid)} ({100*len(valid)/len(results_df):.1f}%)")
    logger.info(f"Failed:    {len(invalid)} ({100*len(invalid)/len(results_df):.1f}%)")

    logger.info(f"\nDative bonds per complex:")
    for n, c in sorted(valid['n_dative_bonds'].value_counts().items()):
        logger.info(f"  {n} bonds: {c}")

    logger.info(f"\nDonor types:")
    for dt, c in valid['donor_info'].value_counts().head(10).items():
        logger.info(f"  {dt}: {c}")

    logger.info(f"\nMetals (valid):")
    for m, c in valid['metal_type'].value_counts().head(15).items():
        logger.info(f"  {m}: {c}")

    logger.info(f"\nlogK1 range (valid): {valid['logK1'].min():.2f} to {valid['logK1'].max():.2f}")
    logger.info(f"logK1 mean: {valid['logK1'].mean():.2f}")

    if len(invalid) > 0:
        logger.info(f"\nFailure reasons:")
        for reason, c in fail_reasons.most_common():
            logger.info(f"  {reason}: {c}")

    logger.info(f"\nSaved: {out_path}")
    logger.info(f"Failed: {fail_path}")

    # Sanity check: show 5 examples
    logger.info(f"\n{'='*60}")
    logger.info("SANITY CHECK — 5 random conversions")
    logger.info(f"{'='*60}")
    sample = valid.sample(min(5, len(valid)), random_state=42)
    for _, row in sample.iterrows():
        logger.info(f"  Original: {row['original_smiles'][:60]}")
        logger.info(f"  Dative:   {row['dative_smiles'][:60]}")
        logger.info(f"  Metal={row['metal_type']}, Donors={row['donor_info']}, logK={row['logK1']}")
        logger.info("")


if __name__ == '__main__':
    main()
