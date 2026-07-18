"""
extract_logbeta.py — Experiment 1: K1 vs log beta selectivity proxy test

Re-parses NIST SRD 46 to extract both K1 (first stability constant) and
overall beta_n (cumulative stability constants with n ligands), then pairs
them by ligand+metal and computes Spearman rank correlation analysis to
quantify how well K1 rankings proxy for overall stability rankings.

Steps:
  1. Parse NIST SRD 46 tables for K1 and beta_n records at 25C
  2. Resolve ligand names to SMILES (uses existing PubChem cache)
  3. Pair K1 with beta_n for same ligand+metal combinations
  4. Compute per-ligand Spearman rho across metals
  5. Save summary JSON + all intermediate CSVs

Schema reference: data/experiment1_schema_notes.md

Usage:
    python stability_gnn/extract_logbeta.py
"""

import os
import re
import json
import time
import logging
from collections import defaultdict
from urllib.parse import quote
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# Paths
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # TMC_binding/
SRD46_DIR = os.path.join(BASE_DIR, 'data', 'external', 'NIST_SRD46')
DATA_DIR = os.path.join(BASE_DIR, 'data')
EXTERNAL_DIR = os.path.join(DATA_DIR, 'external')
CACHE_FILE = os.path.join(DATA_DIR, 'pubchem_smiles_cache.json')

# Beta definition IDs — from schema investigation (see experiment1_schema_notes.md)
# K1 variants: simple [ML]/[M][L] and stereochemical forms
K1_BETA_IDS = {812, 814, 817}  # [ML]/[M][L], [ML(oct.)]/[M][L], [ML(red)]/[M][L]

# Overall beta_n: [ML_n]/[M][L]^n for n >= 2
BETA_N_TO_N = {
    840: 2,   # [ML2]/[M][L]^2
    872: 3,   # [ML3]/[M][L]^3
    894: 4,   # [ML4]/[M][L]^4
    903: 5,   # [ML5]/[M][L]^5
    907: 6,   # [ML6]/[M][L]^6
    910: 8,   # [ML8]/[M][L]^8
}

ALL_TARGET_BETA_IDS = K1_BETA_IDS | set(BETA_N_TO_N.keys())


# ============================================================================
# Step 1: Parse NIST SRD 46 tables
# ============================================================================

def parse_ligands(path):
    """Parse liganden.txt -> {ligand_id: ligand_name}"""
    ligands = {}
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                lig_id = int(parts[0])
                name = re.sub(r'<[^>]+>', '', parts[1].strip())
                ligands[lig_id] = name
    return ligands


def parse_metals(path):
    """Parse metal.txt -> {metal_id: (symbol, charge, name)}"""
    metals = {}
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 5:
                met_id = int(parts[0])
                name = re.sub(r'<[^>]+>', '', parts[1].strip())
                symbol = parts[2].strip()
                charge_match = re.search(r'(\d+)\+', name)
                charge = int(charge_match.group(1)) if charge_match else 0
                metals[met_id] = (symbol, charge, name)
    return metals


def parse_beta_definitions(path):
    """Parse beta_definition.txt -> {beta_id: equation_clean}"""
    betas = {}
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                try:
                    beta_id = int(parts[0])
                    equation = re.sub(r'<[^>]+>', '', parts[1].strip())
                    betas[beta_id] = equation
                except (ValueError, IndexError):
                    continue
    return betas


def parse_stability_constants_filtered(path):
    """
    Parse verkn_ligand_metal.txt and keep only records relevant to this experiment.

    Columns (0-indexed, tab-separated):
      0: record_id
      1: ligand_id
      2: metal_id
      3: beta_def_id
      4: const_type_id  (3 = K, equilibrium constant)
      5: temperature (C)
      6: ionic_strength
      7: log_value
      8: log_value2
      9: error
     10: NULL
     11: solvent_flag (1 = aqueous)
     ...
    """
    records = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 8:
                continue
            try:
                const_type_id = int(parts[4])
                if const_type_id != 3:
                    continue  # Only K-type (equilibrium constants)

                beta_def_id = int(parts[3])
                if beta_def_id not in ALL_TARGET_BETA_IDS:
                    continue  # Only K1 and beta_n we care about

                temperature = float(parts[5]) if parts[5] != '\\N' else None
                log_value = float(parts[7]) if parts[7] != '\\N' else None

                if log_value is None:
                    continue

                rec = {
                    'record_id': int(parts[0]),
                    'ligand_id': int(parts[1]),
                    'metal_id': int(parts[2]),
                    'beta_def_id': beta_def_id,
                    'const_type_id': const_type_id,
                    'temperature': temperature,
                    'ionic_strength': float(parts[6]) if parts[6] != '\\N' else None,
                    'log_value': log_value,
                }
                records.append(rec)
            except (ValueError, IndexError):
                continue
    return records


# ============================================================================
# Step 2: PubChem name -> SMILES (uses existing cache, minimal API calls)
# ============================================================================

def load_smiles_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_smiles_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)


def query_pubchem_smiles(name, cache):
    """Query PubChem for SMILES from compound name. Returns SMILES or None."""
    if name in cache:
        return cache[name]

    clean_name = name.split('(')[0].strip()
    if not clean_name:
        clean_name = name

    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(clean_name)}/property/CanonicalSMILES/JSON"

    try:
        req = Request(url, headers={'User-Agent': 'NIST_SRD46_Experiment1/1.0'})
        with urlopen(req, timeout=10) as response:
            data = json.loads(response.read())
            smiles = data['PropertyTable']['Properties'][0]['CanonicalSMILES']
            cache[name] = smiles
            return smiles
    except (HTTPError, URLError, KeyError, IndexError, json.JSONDecodeError):
        pass

    # Fallback: NCI CIR
    try:
        url2 = f"https://cactus.nci.nih.gov/chemical/structure/{quote(clean_name)}/smiles"
        req2 = Request(url2, headers={'User-Agent': 'NIST_SRD46_Experiment1/1.0'})
        with urlopen(req2, timeout=10) as response:
            smiles = response.read().decode('utf-8').strip()
            if smiles and len(smiles) < 500 and '.' not in smiles[:5]:
                cache[name] = smiles
                return smiles
    except (HTTPError, URLError):
        pass

    cache[name] = None
    return None


def batch_resolve_smiles(ligand_names, batch_size=5, delay=0.25):
    """Resolve ligand names to SMILES; uses cache, queries PubChem only for new names."""
    cache = load_smiles_cache()

    unique_names = list(set(ligand_names))
    to_query = [n for n in unique_names if n not in cache]

    logger.info(f"Unique ligands: {len(unique_names)}")
    logger.info(f"Already cached: {len(unique_names) - len(to_query)}")
    logger.info(f"New queries needed: {len(to_query)}")

    resolved_new = 0
    failed_new = 0

    for i, name in enumerate(tqdm(to_query, desc="PubChem lookup")):
        smiles = query_pubchem_smiles(name, cache)
        if smiles:
            resolved_new += 1
        else:
            failed_new += 1

        # Rate limiting: <= 5 requests/second
        if (i + 1) % batch_size == 0:
            time.sleep(delay)

        # Save cache periodically
        if (i + 1) % 100 == 0:
            save_smiles_cache(cache)

    save_smiles_cache(cache)
    logger.info(f"New: resolved={resolved_new}, failed={failed_new}")

    return cache


# ============================================================================
# Step 3: Denticity estimation from SMILES
# ============================================================================

def estimate_denticity(smiles):
    """
    Estimate denticity (number of coordinating atoms) from SMILES.

    Rules (replicating logic from coord_only_inference.py::_is_coordination_site):
      - N: if not positively charged (formal charge < 1)
      - O, S, Cl, Br, I: always
      - P: if formal charge <= 0
    Counts these atoms in the SMILES string. Capped at 6.
    This is a lightweight approximation without RDKit parsing.
    """
    if not smiles or pd.isna(smiles):
        return None

    try:
        # Use RDKit for accurate count
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return _estimate_denticity_regex(smiles)

        count = 0
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            formal_charge = atom.GetFormalCharge()
            if symbol == 'N' and formal_charge < 1:
                count += 1
            elif symbol in ('O', 'S', 'Cl', 'Br', 'I'):
                count += 1
            elif symbol == 'P' and formal_charge <= 0:
                count += 1
        return min(count, 6)
    except ImportError:
        return _estimate_denticity_regex(smiles)


def _estimate_denticity_regex(smiles):
    """Fallback denticity estimation without RDKit using regex."""
    if not smiles:
        return None
    count = 0
    # Count N not followed by +, O, S, Cl, Br, I, P
    # Simple approximation: count heteroatoms
    count += len(re.findall(r'(?<![A-Z])N(?!\+)', smiles))
    count += len(re.findall(r'O', smiles))
    count += len(re.findall(r'S(?!i)', smiles))
    count += len(re.findall(r'P(?!b)', smiles))
    return min(count, 6)


# ============================================================================
# Step 4: Build K1 vs beta_n paired table
# ============================================================================

def build_paired_table(df_25c):
    """
    For each ligand+metal pair, pair K1 records with beta_n records.

    Returns a DataFrame with columns:
      smiles, ligand_id, ligand_name, metal_symbol, metal_charge,
      n_ligands_in_beta, logK1, logBeta, denticity_estimate
    """
    # Separate K1 and beta_n records
    df_k1 = df_25c[df_25c['beta_def_id'].isin(K1_BETA_IDS)].copy()
    df_beta = df_25c[df_25c['beta_def_id'].isin(BETA_N_TO_N.keys())].copy()

    # Add n_ligands_in_beta column
    df_beta = df_beta.copy()
    df_beta['n_ligands_in_beta'] = df_beta['beta_def_id'].map(BETA_N_TO_N)

    logger.info(f"K1 records at 25C: {len(df_k1)}")
    logger.info(f"Beta_n records at 25C: {len(df_beta)}")

    # Aggregate K1: median per (ligand_id, metal_id)
    # (Multiple measurements exist at different ionic strengths — use median)
    k1_agg = (
        df_k1.groupby(['ligand_id', 'metal_id'])['log_value']
        .median()
        .reset_index()
        .rename(columns={'log_value': 'logK1'})
    )

    # Aggregate beta_n: median per (ligand_id, metal_id, n_ligands_in_beta)
    beta_agg = (
        df_beta.groupby(['ligand_id', 'metal_id', 'n_ligands_in_beta'])['log_value']
        .median()
        .reset_index()
        .rename(columns={'log_value': 'logBeta'})
    )

    # Merge: inner join on ligand_id + metal_id
    paired = pd.merge(k1_agg, beta_agg, on=['ligand_id', 'metal_id'], how='inner')

    logger.info(f"Paired records (K1 + at least one beta_n): {len(paired)}")
    logger.info(f"Unique ligand+metal pairs: {paired[['ligand_id','metal_id']].drop_duplicates().shape[0]}")

    # Add metadata from the full df
    meta_cols = ['ligand_id', 'metal_id', 'ligand_name', 'metal_symbol',
                 'metal_charge', 'ligand_smiles', 'beta_def_equation']
    # Build metadata lookup from df_25c
    meta = (
        df_25c[['ligand_id', 'metal_id', 'ligand_name', 'metal_symbol',
                'metal_charge', 'ligand_smiles', 'beta_def_equation']]
        .drop_duplicates(subset=['ligand_id', 'metal_id'])
    )

    paired = pd.merge(paired, meta, on=['ligand_id', 'metal_id'], how='left')

    # Denticity estimate
    smiles_to_dent = {}
    for smi in paired['ligand_smiles'].dropna().unique():
        smiles_to_dent[smi] = estimate_denticity(smi)

    paired['denticity_estimate'] = paired['ligand_smiles'].map(smiles_to_dent)

    # Compute offset
    paired['logBeta_minus_logK1'] = paired['logBeta'] - paired['logK1']

    # Final column selection and rename
    out = paired[[
        'ligand_smiles', 'ligand_id', 'ligand_name', 'metal_symbol', 'metal_charge',
        'n_ligands_in_beta', 'logK1', 'logBeta', 'logBeta_minus_logK1',
        'denticity_estimate'
    ]].rename(columns={'ligand_smiles': 'smiles'})

    return out


# ============================================================================
# Step 5: Spearman rank correlation analysis
# ============================================================================

def compute_selectivity_analysis(paired_df):
    """
    For each ligand SMILES, compute Spearman rho between K1 and log beta
    across different metals.

    Groups by denticity for reporting.
    """
    results = {
        'n_paired_records': int(len(paired_df)),
        'n_unique_ligands': int(paired_df['smiles'].nunique()),
        'n_ligands_with_multi_metal_data': 0,
        'per_denticity_agreement': {},
        'chelate_effect_by_denticity': {},
        'per_ligand_rho': []
    }

    # For each ligand+n_ligands_in_beta combination, compute rho across metals
    # Use n_ligands_in_beta=2 as the representative "overall stability" for cross-metal comparison
    # Also compute for maximum n (most informative of full chelation)
    # Strategy: for each ligand SMILES, pick the highest n beta available,
    # then compute Spearman rho of K1 vs that beta across metals

    per_ligand_rows = []

    for smiles, grp in paired_df.groupby('smiles'):
        if smiles is None or pd.isna(smiles):
            continue

        # Get denticity (should be same for all rows of this ligand)
        dent = grp['denticity_estimate'].iloc[0]

        # For each n_ligands_in_beta separately
        for n_beta, grp_n in grp.groupby('n_ligands_in_beta'):
            # Need K1 and beta for same (ligand, metal) pair — already ensured by merge
            # Need >= 3 unique metals for rank correlation
            metals = grp_n['metal_symbol'].unique()
            if len(metals) < 3:
                continue

            # Aggregate by metal: median K1, median logBeta
            metal_agg = (
                grp_n.groupby('metal_symbol')
                .agg(logK1_med=('logK1', 'median'), logBeta_med=('logBeta', 'median'))
                .reset_index()
            )

            if len(metal_agg) < 3:
                continue

            rho, pval = spearmanr(metal_agg['logK1_med'], metal_agg['logBeta_med'])

            per_ligand_rows.append({
                'smiles': smiles,
                'n_ligands_in_beta': n_beta,
                'denticity_estimate': dent,
                'n_metals': len(metal_agg),
                'spearman_rho': rho,
                'spearman_pval': pval,
            })

    per_ligand_df = pd.DataFrame(per_ligand_rows)
    results['n_ligands_with_multi_metal_data'] = int(per_ligand_df['smiles'].nunique())
    results['per_ligand_rho'] = per_ligand_df.to_dict(orient='records')

    logger.info(f"Ligands with multi-metal data: {results['n_ligands_with_multi_metal_data']}")
    logger.info(f"Per-ligand Spearman computations: {len(per_ligand_df)}")

    # Per-denticity agreement statistics
    # Group denticity 6+ together
    def dent_label(d):
        if d is None or pd.isna(d):
            return 'unknown'
        d = int(d)
        return str(min(d, 6))

    per_ligand_df['dent_label'] = per_ligand_df['denticity_estimate'].apply(dent_label)

    for dent_str, dg in per_ligand_df.groupby('dent_label'):
        rhos = dg['spearman_rho'].dropna().values
        if len(rhos) == 0:
            continue
        results['per_denticity_agreement'][dent_str] = {
            'n_observations': int(len(dg)),
            'n_ligands': int(dg['smiles'].nunique()),
            'mean_rho': float(np.mean(rhos)),
            'median_rho': float(np.median(rhos)),
            'std_rho': float(np.std(rhos)),
            'frac_above_0.8': float(np.mean(rhos > 0.8)),
            'frac_above_0.5': float(np.mean(rhos > 0.5)),
            'frac_below_0.5': float(np.mean(rhos < 0.5)),
        }

    # Chelate effect offset by denticity
    paired_df['dent_label'] = paired_df['denticity_estimate'].apply(dent_label)
    for dent_str, dg in paired_df.groupby('dent_label'):
        offsets = dg['logBeta_minus_logK1'].dropna().values
        if len(offsets) == 0:
            continue
        results['chelate_effect_by_denticity'][dent_str] = {
            'n_records': int(len(offsets)),
            'mean_offset': float(np.mean(offsets)),
            'median_offset': float(np.median(offsets)),
            'std_offset': float(np.std(offsets)),
            'q25_offset': float(np.percentile(offsets, 25)),
            'q75_offset': float(np.percentile(offsets, 75)),
        }

    return results, per_ligand_df


# ============================================================================
# Main
# ============================================================================

def main():
    logger.info("=" * 60)
    logger.info("Experiment 1: K1 vs log beta selectivity proxy analysis")
    logger.info("=" * 60)

    os.makedirs(EXTERNAL_DIR, exist_ok=True)

    # --- Parse tables ---
    logger.info("Parsing NIST SRD 46 tables...")
    ligands = parse_ligands(os.path.join(SRD46_DIR, 'liganden.txt'))
    metals = parse_metals(os.path.join(SRD46_DIR, 'metal.txt'))
    beta_defs = parse_beta_definitions(os.path.join(SRD46_DIR, 'beta_definition.txt'))

    logger.info(f"Ligands: {len(ligands)}")
    logger.info(f"Metals: {len(metals)}")
    logger.info(f"Beta definitions: {len(beta_defs)}")

    # Verify our target beta IDs exist in the file
    for bid in sorted(ALL_TARGET_BETA_IDS):
        eq = beta_defs.get(bid, 'NOT FOUND')
        logger.info(f"  beta_id {bid}: {eq}")

    # --- Parse stability constants (filtered to target beta IDs) ---
    logger.info("\nParsing stability constants (K1 and beta_n only)...")
    records = parse_stability_constants_filtered(
        os.path.join(SRD46_DIR, 'verkn_ligand_metal.txt')
    )
    logger.info(f"Filtered records (K1 + beta_n): {len(records)}")

    # --- Build dataframe ---
    rows = []
    for rec in records:
        lig_id = rec['ligand_id']
        met_id = rec['metal_id']
        if lig_id not in ligands or met_id not in metals:
            continue
        lig_name = ligands[lig_id]
        met_symbol, met_charge, met_name = metals[met_id]
        bid = rec['beta_def_id']
        rows.append({
            'ligand_id': lig_id,
            'ligand_name': lig_name,
            'metal_id': met_id,
            'metal_symbol': met_symbol,
            'metal_charge': met_charge,
            'beta_def_id': bid,
            'beta_def_equation': beta_defs.get(bid, ''),
            'temperature': rec['temperature'],
            'ionic_strength': rec['ionic_strength'],
            'log_value': rec['log_value'],
        })

    df = pd.DataFrame(rows)
    logger.info(f"Records with ligand+metal metadata: {len(df)}")

    # --- Filter to 25C ---
    df_25 = df[df['temperature'] == 25.0].copy()
    logger.info(f"At 25C: {len(df_25)}")

    k1_count = df_25[df_25['beta_def_id'].isin(K1_BETA_IDS)].shape[0]
    beta_count = df_25[df_25['beta_def_id'].isin(BETA_N_TO_N.keys())].shape[0]
    logger.info(f"  K1 records: {k1_count}")
    logger.info(f"  Beta_n records: {beta_count}")

    # --- Resolve SMILES ---
    logger.info("\nResolving ligand names to SMILES...")
    unique_ligands = df_25['ligand_name'].unique().tolist()
    cache = batch_resolve_smiles(unique_ligands)
    df_25['ligand_smiles'] = df_25['ligand_name'].map(cache)

    has_smiles = df_25['ligand_smiles'].notna()
    logger.info(f"Records with SMILES: {has_smiles.sum()} / {len(df_25)} ({100*has_smiles.mean():.1f}%)")

    # --- Save Step 2 output: K1 and beta_n at 25C ---
    out1_path = os.path.join(EXTERNAL_DIR, 'nist_srd46_K1_and_beta_25C.csv')
    df_25.to_csv(out1_path, index=False)
    logger.info(f"\nSaved: {out1_path} ({len(df_25)} rows)")

    # --- Save Step 3 output: with SMILES ---
    df_with_smiles = df_25[has_smiles].copy()
    out2_path = os.path.join(EXTERNAL_DIR, 'nist_srd46_K1_and_beta_with_smiles.csv')
    df_with_smiles.to_csv(out2_path, index=False)
    logger.info(f"Saved: {out2_path} ({len(df_with_smiles)} rows)")

    # --- Step 4: Build paired table ---
    logger.info("\nBuilding K1 vs beta_n paired table...")
    paired_df = build_paired_table(df_with_smiles)

    out3_path = os.path.join(EXTERNAL_DIR, 'nist_srd46_K1_vs_logbeta_paired.csv')
    paired_df.to_csv(out3_path, index=False)
    logger.info(f"Saved: {out3_path} ({len(paired_df)} rows)")

    # Distribution by n_ligands_in_beta
    logger.info("\nPaired records by n_ligands_in_beta:")
    for n, cnt in paired_df['n_ligands_in_beta'].value_counts().sort_index().items():
        logger.info(f"  beta_{n}: {cnt} records")

    logger.info("\nPaired records by denticity:")
    for d, cnt in paired_df['denticity_estimate'].value_counts().sort_index().items():
        logger.info(f"  denticity {d}: {cnt} records")

    # --- Step 5: Spearman analysis ---
    logger.info("\nComputing Spearman rank correlation analysis...")
    results, per_ligand_df = compute_selectivity_analysis(paired_df)

    # Save per-ligand rho to separate CSV
    per_ligand_csv = os.path.join(EXTERNAL_DIR, 'experiment1_per_ligand_rho.csv')
    per_ligand_df.to_csv(per_ligand_csv, index=False)
    logger.info(f"Saved: {per_ligand_csv}")

    # Remove per_ligand_rho from results dict before saving JSON
    # (keep it in CSV, it's large)
    results_json = {k: v for k, v in results.items() if k != 'per_ligand_rho'}

    out4_path = os.path.join(EXTERNAL_DIR, 'experiment1_selectivity_results.json')
    with open(out4_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    logger.info(f"Saved: {out4_path}")

    # --- Summary ---
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 1 SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total K1 records at 25C: {k1_count}")
    logger.info(f"Total beta_n records at 25C: {beta_count}")
    logger.info(f"Paired records (K1 + beta_n): {results['n_paired_records']}")
    logger.info(f"Ligands with multi-metal data: {results['n_ligands_with_multi_metal_data']}")
    logger.info("\nSpearman rho by denticity:")
    for dent in sorted(results['per_denticity_agreement'].keys()):
        d = results['per_denticity_agreement'][dent]
        logger.info(
            f"  denticity {dent}: n={d['n_ligands']}, "
            f"median_rho={d['median_rho']:.3f}, "
            f"frac>0.8={d['frac_above_0.8']:.2f}, "
            f"frac>0.5={d['frac_above_0.5']:.2f}"
        )
    logger.info("\nChelate effect offset (logBeta - logK1) by denticity:")
    for dent in sorted(results['chelate_effect_by_denticity'].keys()):
        d = results['chelate_effect_by_denticity'][dent]
        logger.info(
            f"  denticity {dent}: mean={d['mean_offset']:.2f}, "
            f"median={d['median_offset']:.2f}, "
            f"std={d['std_offset']:.2f}"
        )


if __name__ == '__main__':
    main()
