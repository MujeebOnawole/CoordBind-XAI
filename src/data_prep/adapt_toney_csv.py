"""
adapt_toney_csv.py - Convert Toney et al. CSV to our evaluation format

Toney's CSV has its own column naming. This script reads it, identifies
the relevant columns (SMILES, coordinating atom info), and outputs a CSV
in the format expected by eval_coordination_binding.py:

    smiles, denticities, coordinating_atom_indices, coordinating_atom_symbols

Usage:
    python adapt_toney_csv.py --input data/toney/ligand_data_final_test.csv \
                              --output data/toney/toney_test_adapted.csv
"""

import os
import re
import ast
import csv
import argparse
import logging
import numpy as np
import pandas as pd
from rdkit import Chem

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_column(df, candidates, required=True):
    """Find a column by trying multiple possible names (case-insensitive)."""
    for cand in candidates:
        for col in df.columns:
            if col.lower().strip() == cand.lower().strip():
                return col
    if required:
        logger.error(f"Could not find column. Tried: {candidates}")
        logger.error(f"Available columns: {list(df.columns)}")
        raise ValueError(f"Missing required column: {candidates}")
    return None


def parse_catom_indices(value):
    """Parse coordinating atom indices from various formats."""
    if pd.isna(value):
        return []

    value = str(value).strip()

    # Try direct list parsing: [0, 3, 5] or (0, 3, 5)
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (list, tuple)):
            return [int(x) for x in parsed]
        if isinstance(parsed, int):
            return [parsed]
    except (ValueError, SyntaxError):
        pass

    # Try comma-separated: "0,3,5"
    if ',' in value:
        try:
            return [int(x.strip()) for x in value.split(',')]
        except ValueError:
            pass

    # Try space-separated: "0 3 5"
    parts = value.split()
    if len(parts) > 1:
        try:
            return [int(x) for x in parts]
        except ValueError:
            pass

    # Single integer
    try:
        return [int(value)]
    except ValueError:
        pass

    return []


def get_atom_symbols(smiles, indices):
    """Get element symbols for atom indices from SMILES."""
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return []
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                         Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    except Exception:
        pass

    symbols = []
    for idx in indices:
        if idx < mol.GetNumAtoms():
            symbols.append(mol.GetAtomWithIdx(idx).GetSymbol())
        else:
            symbols.append('?')
    return symbols


def adapt_csv(input_path: str, output_path: str, max_rows: int = None):
    """Convert Toney CSV to our evaluation format."""

    logger.info(f"Reading {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"First row sample:")
    for col in df.columns:
        logger.info(f"  {col}: {df[col].iloc[0]}")

    # Find SMILES column
    smiles_col = find_column(df, [
        'SMILES_without_X_rdkit', 'SMILES_without_X_molsimplify',
        'smiles', 'SMILES', 'smi', 'canonical_smiles', 'ligand_smiles',
        'smiles_rdkit', 'rdkit_smiles',
    ])
    logger.info(f"SMILES column: '{smiles_col}'")

    # Find coordinating atom indices column
    # Toney uses "Connecting_Atom_Indices_rdkit_smiles" for indices matching RDKit SMILES
    catom_col = find_column(df, [
        'Connecting_Atom_Indices_rdkit_smiles',
        'Connecting_Atom_Indices_without_XH',
        'Connecting_Atom_Indices_without_X',
        'Connecting_Atom_Indices',
        'coordinating_atom_indices', 'catom_indices', 'catoms', 'catom_idx',
        'coordinating_atoms', 'coord_atoms', 'coordinating_atom_idx',
        'connecting_atoms', 'connecting_atom_indices',
    ], required=False)

    # Find denticity column
    dent_col = find_column(df, [
        'Ligand_Denticities', 'denticities_zero_index',
        'denticities', 'denticity', 'dent', 'n_catoms', 'num_catoms',
        'n_coordinating_atoms', 'coordination_number', 'num_coordinating_atoms',
    ], required=False)

    # Find symbols column
    symbols_col = find_column(df, [
        'Connecting_Atom_Symbols', 'Test_catoms_read_from_smiles',
        'coordinating_atom_symbols',
    ], required=False)

    if catom_col is None and dent_col is None:
        logger.info("Searching for columns with list-like values...")
        for col in df.columns:
            sample = str(df[col].iloc[0])
            if '[' in sample and ']' in sample:
                logger.info(f"  Potential list column: '{col}' = {sample[:100]}")

    if catom_col:
        logger.info(f"Coordinating atom column: '{catom_col}'")
        logger.info(f"  Sample values: {df[catom_col].iloc[:3].tolist()}")
    if dent_col:
        logger.info(f"Denticity column: '{dent_col}'")
        logger.info(f"  Distribution: {df[dent_col].value_counts().sort_index().to_dict()}")
    if symbols_col:
        logger.info(f"Symbols column: '{symbols_col}'")

    # Build output
    rows = []
    parse_failures = 0
    smiles_failures = 0

    if max_rows:
        df = df.head(max_rows)

    for idx, row in df.iterrows():
        smiles = str(row[smiles_col]).strip()

        # Skip empty SMILES
        if not smiles or smiles == 'nan':
            smiles_failures += 1
            continue

        # Validate SMILES
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            smiles_failures += 1
            continue

        # Parse coordinating atom indices
        if catom_col:
            catom_indices = parse_catom_indices(row[catom_col])
        else:
            catom_indices = []

        # Get denticity
        if dent_col and not pd.isna(row[dent_col]):
            dent = int(row[dent_col])
        else:
            dent = len(catom_indices)

        if dent == 0 and len(catom_indices) == 0:
            parse_failures += 1
            continue

        # Get atom symbols (prefer Toney's column, fallback to RDKit)
        if symbols_col and not pd.isna(row[symbols_col]):
            raw_symbols = parse_catom_indices(row[symbols_col])  # handles list parsing
            # If it parsed as strings not ints, use directly
            raw_str = str(row[symbols_col]).strip()
            try:
                symbols = ast.literal_eval(raw_str)
                if isinstance(symbols, (list, tuple)):
                    symbols = [str(s) for s in symbols]
                else:
                    symbols = get_atom_symbols(smiles, catom_indices)
            except (ValueError, SyntaxError):
                symbols = get_atom_symbols(smiles, catom_indices)
        else:
            symbols = get_atom_symbols(smiles, catom_indices)

        rows.append({
            'smiles': smiles,
            'denticities': dent,
            'coordinating_atom_indices': str(catom_indices),
            'coordinating_atom_symbols': str(symbols),
        })

        if (idx + 1) % 1000 == 0:
            logger.info(f"  Processed {idx+1}/{len(df)} rows")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False)

    logger.info(f"\n{'='*60}")
    logger.info(f"ADAPTATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Input:  {len(df)} rows")
    logger.info(f"Output: {len(out_df)} rows")
    logger.info(f"SMILES failures: {smiles_failures}")
    logger.info(f"Parse failures:  {parse_failures}")
    logger.info(f"Denticity distribution: {out_df['denticities'].value_counts().sort_index().to_dict()}")
    logger.info(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Adapt Toney CSV to our format')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--max_rows', type=int, default=None,
                        help='Limit rows (for testing)')
    args = parser.parse_args()

    adapt_csv(args.input, args.output, args.max_rows)


if __name__ == '__main__':
    main()
