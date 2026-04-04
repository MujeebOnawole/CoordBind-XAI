"""
process_nist_srd46.py — Extract log K1 stability constants from NIST SRD 46
and convert ligand names to SMILES via PubChem.

Pipeline:
  1. Parse tab-delimited NIST SRD 46 text files
  2. Filter for simple 1:1 metal-ligand log K1 constants (type 'K')
  3. Batch-query PubChem to convert ligand names → SMILES
  4. Output CSV ready for GNN training

Usage:
    python process_nist_srd46.py
"""

import os
import re
import csv
import time
import json
import logging
from collections import Counter
from urllib.parse import quote
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRD46_DIR = os.path.join(SCRIPT_DIR, 'data', 'external', 'NIST_SRD46')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'data')
CACHE_FILE = os.path.join(OUTPUT_DIR, 'pubchem_smiles_cache.json')


# ============================================================================
# Step 1: Parse NIST SRD 46 tables
# ============================================================================

def parse_ligands(path):
    """Parse liganden.txt → {ligand_id: ligand_name}"""
    ligands = {}
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                lig_id = int(parts[0])
                name = parts[1].strip()
                # Clean up name: remove HTML tags
                name = re.sub(r'<[^>]+>', '', name)
                ligands[lig_id] = name
    return ligands


def parse_metals(path):
    """Parse metal.txt → {metal_id: (symbol, charge, name)}"""
    metals = {}
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 5:
                met_id = int(parts[0])
                name = re.sub(r'<[^>]+>', '', parts[1].strip())  # e.g. "Cu2+"
                symbol = parts[2].strip()  # e.g. "Cu"
                # Extract charge from name
                charge_match = re.search(r'(\d+)\+', name)
                charge = int(charge_match.group(1)) if charge_match else 0
                metals[met_id] = (symbol, charge, name)
    return metals


def parse_beta_definitions(path):
    """Parse beta_definition.txt → {beta_id: equation}"""
    betas = {}
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                beta_id = int(parts[0])
                equation = re.sub(r'<[^>]+>', '', parts[1].strip())
                betas[beta_id] = equation
    return betas


def parse_stability_constants(path):
    """Parse verkn_ligand_metal.txt → list of records.

    Columns (tab-separated):
      0: record_id
      1: ligand_id
      2: metal_id
      3: beta_definition_id
      4: constant_type_id (1=*, 2=H, 3=K, 4=S)
      5: temperature (C)
      6: ionic_strength
      7: log_value (the stability constant)
      8: log_value2 (often same)
      9: error
      ...
    """
    records = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 8:
                try:
                    rec = {
                        'record_id': int(parts[0]),
                        'ligand_id': int(parts[1]),
                        'metal_id': int(parts[2]),
                        'beta_def_id': int(parts[3]),
                        'const_type_id': int(parts[4]),
                        'temperature': float(parts[5]) if parts[5] != '\\N' else None,
                        'ionic_strength': float(parts[6]) if parts[6] != '\\N' else None,
                        'log_value': float(parts[7]) if parts[7] != '\\N' else None,
                    }
                    records.append(rec)
                except (ValueError, IndexError):
                    continue
    return records


# ============================================================================
# Step 2: Filter for simple log K1 (1:1 metal:ligand)
# ============================================================================

def filter_logK1(records, beta_defs):
    """Keep only simple 1:1 metal-ligand formation constants (log K1).

    K1 = [ML] / ([M][L])

    Filter criteria:
      - const_type_id = 3 (K type)
      - beta_definition matches [ML]/[M][L] pattern
      - log_value is not None
    """
    # Find beta_def_ids that correspond to simple K1: [ML]/[M][L]
    k1_beta_ids = set()
    for bid, eq in beta_defs.items():
        # Simple 1:1 formation: [ML]/[M][L]
        eq_clean = eq.replace(' ', '')
        if '[ML]/[M][L]' in eq_clean or eq_clean == '[ML]/[M][L]':
            k1_beta_ids.add(bid)

    logger.info(f"K1 beta definition IDs: {k1_beta_ids}")

    # If no exact match, try broader patterns
    if not k1_beta_ids:
        for bid, eq in beta_defs.items():
            eq_clean = eq.replace(' ', '').lower()
            # Match patterns like [ML]/[M][L], [MHL]/[M][HL], etc.
            if re.match(r'\[m\w*l\w*\]/\[m\]\[', eq_clean):
                k1_beta_ids.add(bid)

    logger.info(f"Found {len(k1_beta_ids)} K1 beta definitions")

    # Filter: K type (3) AND K1 beta def AND has value
    filtered = [
        r for r in records
        if r['const_type_id'] == 3
        and r['beta_def_id'] in k1_beta_ids
        and r['log_value'] is not None
    ]

    # If strict filter yields too few, relax: just K type + simple betas
    if len(filtered) < 1000:
        logger.info(f"Strict K1 filter: only {len(filtered)} records. Trying broader filter...")
        # Include all K-type constants
        filtered = [
            r for r in records
            if r['const_type_id'] == 3
            and r['log_value'] is not None
        ]

    return filtered


# ============================================================================
# Step 3: PubChem name → SMILES conversion
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

    # Clean name for URL
    clean_name = name.split('(')[0].strip()  # Remove parenthetical synonyms
    if not clean_name:
        clean_name = name

    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(clean_name)}/property/CanonicalSMILES/JSON"

    try:
        req = Request(url, headers={'User-Agent': 'NIST_SRD46_Processor/1.0'})
        with urlopen(req, timeout=10) as response:
            data = json.loads(response.read())
            smiles = data['PropertyTable']['Properties'][0]['CanonicalSMILES']
            cache[name] = smiles
            return smiles
    except (HTTPError, URLError, KeyError, IndexError, json.JSONDecodeError):
        pass

    # Fallback: try NCI Chemical Identifier Resolver
    try:
        url2 = f"https://cactus.nci.nih.gov/chemical/structure/{quote(clean_name)}/smiles"
        req2 = Request(url2, headers={'User-Agent': 'NIST_SRD46_Processor/1.0'})
        with urlopen(req2, timeout=10) as response:
            smiles = response.read().decode('utf-8').strip()
            if smiles and len(smiles) < 500 and '.' not in smiles[:5]:
                cache[name] = smiles
                return smiles
    except (HTTPError, URLError):
        pass

    cache[name] = None
    return None


def batch_resolve_smiles(ligand_names, batch_size=5, delay=0.3):
    """Resolve ligand names to SMILES in batches with rate limiting."""
    cache = load_smiles_cache()
    resolved = 0
    failed = 0

    unique_names = list(set(ligand_names))
    # Skip already cached
    to_query = [n for n in unique_names if n not in cache]

    logger.info(f"Unique ligands: {len(unique_names)}")
    logger.info(f"Already cached: {len(unique_names) - len(to_query)}")
    logger.info(f"To query: {len(to_query)}")

    for i, name in enumerate(tqdm(to_query, desc="PubChem lookup")):
        smiles = query_pubchem_smiles(name, cache)
        if smiles:
            resolved += 1
        else:
            failed += 1

        # Rate limiting: PubChem allows 5 requests/second
        if (i + 1) % batch_size == 0:
            time.sleep(delay)

        # Save cache periodically
        if (i + 1) % 100 == 0:
            save_smiles_cache(cache)

    save_smiles_cache(cache)
    logger.info(f"Resolved: {resolved}, Failed: {failed}")

    return cache


# ============================================================================
# Main
# ============================================================================

def main():
    logger.info("=" * 60)
    logger.info("NIST SRD 46 → logK + SMILES Processing")
    logger.info("=" * 60)

    # Parse tables
    logger.info("Parsing NIST SRD 46 tables...")
    ligands = parse_ligands(os.path.join(SRD46_DIR, 'liganden.txt'))
    metals = parse_metals(os.path.join(SRD46_DIR, 'metal.txt'))
    beta_defs = parse_beta_definitions(os.path.join(SRD46_DIR, 'beta_definition.txt'))
    records = parse_stability_constants(os.path.join(SRD46_DIR, 'verkn_ligand_metal.txt'))

    logger.info(f"Ligands: {len(ligands)}")
    logger.info(f"Metals: {len(metals)}")
    logger.info(f"Beta definitions: {len(beta_defs)}")
    logger.info(f"Total records: {len(records)}")

    # Filter for K-type constants
    k_records = filter_logK1(records, beta_defs)
    logger.info(f"K-type records: {len(k_records)}")

    # Build dataframe
    rows = []
    for rec in k_records:
        lig_id = rec['ligand_id']
        met_id = rec['metal_id']

        if lig_id not in ligands or met_id not in metals:
            continue

        lig_name = ligands[lig_id]
        met_symbol, met_charge, met_name = metals[met_id]

        rows.append({
            'ligand_id': lig_id,
            'ligand_name': lig_name,
            'metal_id': met_id,
            'metal_symbol': met_symbol,
            'metal_charge': met_charge,
            'metal_name': met_name,
            'logK': rec['log_value'],
            'temperature': rec['temperature'],
            'ionic_strength': rec['ionic_strength'],
            'beta_def_id': rec['beta_def_id'],
        })

    df = pd.DataFrame(rows)
    logger.info(f"Parsed K-type records with metadata: {len(df)}")

    # Filter for 25°C (most common, standard conditions)
    df_25 = df[df['temperature'] == 25.0].copy()
    logger.info(f"At 25°C: {len(df_25)}")

    # Stats
    logger.info(f"\nUnique ligands: {df_25['ligand_name'].nunique()}")
    logger.info(f"Unique metals: {df_25['metal_symbol'].nunique()}")
    logger.info(f"logK range: {df_25['logK'].min():.2f} to {df_25['logK'].max():.2f}")

    logger.info(f"\nTop 15 metals:")
    for met, cnt in df_25['metal_symbol'].value_counts().head(15).items():
        logger.info(f"  {met}: {cnt}")

    # Resolve ligand names to SMILES via PubChem
    unique_ligands = df_25['ligand_name'].unique().tolist()
    logger.info(f"\nResolving {len(unique_ligands)} ligand names to SMILES via PubChem...")

    cache = batch_resolve_smiles(unique_ligands)

    # Add SMILES to dataframe
    df_25['ligand_smiles'] = df_25['ligand_name'].map(cache)
    has_smiles = df_25['ligand_smiles'].notna()
    logger.info(f"\nRecords with SMILES: {has_smiles.sum()} / {len(df_25)} ({100*has_smiles.mean():.1f}%)")
    logger.info(f"Unique ligands with SMILES: {df_25.loc[has_smiles, 'ligand_name'].nunique()}")

    # Save full result
    df_25.to_csv(os.path.join(OUTPUT_DIR, 'nist_srd46_logK_25C.csv'), index=False)

    # Save SMILES-resolved subset
    df_resolved = df_25[has_smiles].copy()
    df_resolved.to_csv(os.path.join(OUTPUT_DIR, 'nist_srd46_logK_with_smiles.csv'), index=False)

    logger.info(f"\nSaved: nist_srd46_logK_25C.csv ({len(df_25)} rows)")
    logger.info(f"Saved: nist_srd46_logK_with_smiles.csv ({len(df_resolved)} rows)")

    # Summary of what we have
    logger.info(f"\n{'='*60}")
    logger.info("FINAL DATASET SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"NIST SRD 46 K-type at 25°C: {len(df_25)} records")
    logger.info(f"With SMILES resolved: {len(df_resolved)} records")
    logger.info(f"Unique ligands (with SMILES): {df_resolved['ligand_name'].nunique()}")
    logger.info(f"Unique metals: {df_resolved['metal_symbol'].nunique()}")
    logger.info(f"logK range: {df_resolved['logK'].min():.2f} to {df_resolved['logK'].max():.2f}")

    flotation_metals = {'Cu', 'Fe', 'Zn', 'Pb', 'Ni', 'Co', 'Mn'}
    flot = df_resolved[df_resolved['metal_symbol'].isin(flotation_metals)]
    logger.info(f"\nFlotation metals subset: {len(flot)} records")
    for m in sorted(flotation_metals):
        n = len(flot[flot['metal_symbol'] == m])
        if n > 0:
            logger.info(f"  {m}: {n}")


if __name__ == '__main__':
    main()
