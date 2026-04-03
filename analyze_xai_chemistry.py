"""
analyze_xai_chemistry.py - Post-XAI Chemical Validation

Reads XAI output files (xai_results_logK1.csv, xai_attributions_logK1.json)
and the dataset CSV, then validates predictions and attributions against
established coordination chemistry principles.

Analyses performed:
  1. Donor attribution by metal type (which donor atoms drive binding)
  2. Irving-Williams series validation
  3. HSAB (Hard-Soft Acid-Base) pattern validation
  4. Lanthanide contraction check
  5. Metal selectivity for shared ligands (flotation-relevant pairs)

No model loading required — reads only CSV/JSON output files.

Usage:
    python analyze_xai_chemistry.py --results_dir output/xai_results
    python analyze_xai_chemistry.py --results_dir ../Results/output/xai_results
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem
import logging
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    logger.warning("matplotlib not available — skipping figure generation")


# =============================================================================
# CHEMISTRY CONSTANTS
# =============================================================================

# Irving-Williams order for divalent first-row transition metals
IRVING_WILLIAMS_ORDER = ['Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
IRVING_WILLIAMS_RANK = {m: i for i, m in enumerate(IRVING_WILLIAMS_ORDER)}

# HSAB classification
HARD_METALS = {'Li', 'Na', 'Mg', 'Ca', 'Sr', 'Ba', 'Al', 'Sc', 'Ti', 'V',
               'Cr', 'Mn', 'Fe', 'Co', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu',
               'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Th', 'U',
               'Am', 'Cm', 'Zr', 'Hf', 'Y'}
BORDERLINE_METALS = {'Ni', 'Cu', 'Zn', 'Cd', 'Pb', 'Sn', 'Co', 'Fe'}
SOFT_METALS = {'Pd', 'Pt', 'Ag', 'Au', 'Hg', 'Bi'}
# Note: Fe, Co appear in both hard (Fe3+, Co3+) and borderline (Fe2+, Co2+).
# We classify them as borderline since most dataset entries are +2.

HARD_DONORS = {'O'}
BORDERLINE_DONORS = {'N'}
SOFT_DONORS = {'S', 'P'}

# Lanthanide series (in atomic number order)
LANTHANIDES = ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho',
               'Er', 'Tm', 'Yb', 'Lu']
LANTHANIDE_SET = set(LANTHANIDES)
LANTHANIDE_RANK = {m: i for i, m in enumerate(LANTHANIDES)}

# Flotation-relevant metal pairs (target / gangue)
SELECTIVITY_PAIRS = [
    ('Cu', 'Fe', 'Chalcopyrite vs pyrite'),
    ('Cu', 'Zn', 'Copper vs zinc'),
    ('Pb', 'Zn', 'Galena vs sphalerite'),
    ('Pb', 'Fe', 'Galena vs pyrite'),
    ('Ni', 'Fe', 'Pentlandite vs pyrite'),
    ('Co', 'Ni', 'Cobalt vs nickel separation'),
    ('Cu', 'Ni', 'Copper vs nickel'),
]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(results_dir, dataset_csv):
    """Load XAI results and dataset."""
    xai_csv = os.path.join(results_dir, 'xai_results_logK1.csv')
    xai_json = os.path.join(results_dir, 'xai_attributions_logK1.json')

    if not os.path.exists(xai_csv):
        raise FileNotFoundError(f"XAI results CSV not found: {xai_csv}")
    if not os.path.exists(xai_json):
        raise FileNotFoundError(f"XAI attributions JSON not found: {xai_json}")

    logger.info(f"Loading XAI results from {results_dir}")
    xai_df = pd.read_csv(xai_csv)
    with open(xai_json) as f:
        xai_attrs = json.load(f)
    logger.info(f"  {len(xai_df)} XAI results, {len(xai_attrs)} attribution records")

    logger.info(f"Loading dataset from {dataset_csv}")
    dataset_df = pd.read_csv(dataset_csv)
    logger.info(f"  {len(dataset_df)} dataset entries")

    return xai_df, xai_attrs, dataset_df


# =============================================================================
# SMILES PARSING: EXTRACT ATOM ROLES
# =============================================================================

def parse_atom_roles(smiles):
    """
    Parse dative-bond SMILES and return atom-level information.

    Returns list of dicts, one per atom (in RDKit atom index order = PyG node order):
        {'element': str, 'is_metal': bool, 'is_donor': bool, 'atom_idx': int}
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Known metals (same set as config.py)
    metal_set = {
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
        'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
        'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho',
        'Er', 'Tm', 'Yb', 'Lu', 'Th', 'U', 'Np', 'Pu', 'Am', 'Cm',
        'Bk', 'Cf', 'Al', 'Ga', 'In', 'Sn', 'Pb', 'Bi',
        'Mg', 'Ca', 'Sr', 'Ba', 'Li', 'Na', 'Ra',
    }

    # Find metal atoms and donor atoms via dative bonds
    metal_indices = set()
    donor_indices = set()

    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.rdchem.BondType.DATIVE:
            begin = bond.GetBeginAtom()
            end = bond.GetEndAtom()
            begin_sym = begin.GetSymbol()
            end_sym = end.GetSymbol()
            # Dative bond: donor -> acceptor (ligand -> metal)
            if end_sym in metal_set:
                metal_indices.add(end.GetIdx())
                donor_indices.add(begin.GetIdx())
            elif begin_sym in metal_set:
                metal_indices.add(begin.GetIdx())
                donor_indices.add(end.GetIdx())

    atoms = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        sym = atom.GetSymbol()
        atoms.append({
            'element': sym,
            'atom_idx': idx,
            'is_metal': idx in metal_indices,
            'is_donor': idx in donor_indices,
        })

    return atoms


# =============================================================================
# ANALYSIS 1: DONOR ATTRIBUTION BY METAL
# =============================================================================

def analyze_donor_attributions(xai_df, xai_attrs, figures_dir):
    """
    For each complex, identify donor atoms and their attribution values.
    Aggregate by (metal_type, donor_element) to see which donors drive binding.
    """
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS 1: DONOR ATTRIBUTION BY METAL TYPE")
    logger.info("=" * 60)

    # Collect (metal, donor_element) -> [attribution values]
    metal_donor_attrs = defaultdict(list)
    # Also collect metal -> donor vs backbone attribution
    metal_role_attrs = defaultdict(lambda: {'metal': [], 'donor': [], 'backbone': []})
    parse_failures = 0

    for rec in xai_attrs:
        smiles = rec.get('smiles', '')
        metal_type = rec.get('metal_type', '')
        attrs = rec.get('all_signed_attrs', [])
        if not smiles or not attrs:
            continue

        atom_roles = parse_atom_roles(smiles)
        if atom_roles is None or len(atom_roles) != len(attrs):
            parse_failures += 1
            continue

        for atom_info, attr_val in zip(atom_roles, attrs):
            if atom_info['is_metal']:
                metal_role_attrs[metal_type]['metal'].append(attr_val)
            elif atom_info['is_donor']:
                metal_role_attrs[metal_type]['donor'].append(attr_val)
                donor_elem = atom_info['element']
                metal_donor_attrs[(metal_type, donor_elem)].append(attr_val)
            else:
                metal_role_attrs[metal_type]['backbone'].append(attr_val)

    if parse_failures > 0:
        logger.info(f"  Parse failures (SMILES/attr mismatch): {parse_failures}")

    # --- Table 1: Mean |attribution| by role (metal, donor, backbone) ---
    logger.info("\n--- Mean |attribution| by atom role (top 15 metals by count) ---")
    role_summary = []
    for metal in sorted(metal_role_attrs.keys(),
                        key=lambda m: len(metal_role_attrs[m]['donor']), reverse=True):
        d = metal_role_attrs[metal]
        if len(d['donor']) < 10:
            continue
        role_summary.append({
            'metal': metal,
            'n_complexes': len(d['metal']),
            'metal_mean_abs': np.mean(np.abs(d['metal'])) if d['metal'] else 0,
            'donor_mean_abs': np.mean(np.abs(d['donor'])) if d['donor'] else 0,
            'backbone_mean_abs': np.mean(np.abs(d['backbone'])) if d['backbone'] else 0,
            'donor_mean_signed': np.mean(d['donor']) if d['donor'] else 0,
        })

    role_df = pd.DataFrame(role_summary)
    logger.info(f"  {'Metal':<6} {'n':>6}  {'|Metal|':>8} {'|Donor|':>8} {'|Backbone|':>10}  {'Donor(signed)':>13}")
    for _, r in role_df.head(15).iterrows():
        logger.info(f"  {r['metal']:<6} {r['n_complexes']:>6}  {r['metal_mean_abs']:>8.3f} "
                     f"{r['donor_mean_abs']:>8.3f} {r['backbone_mean_abs']:>10.3f}  "
                     f"{r['donor_mean_signed']:>+13.3f}")

    # --- Table 2: Donor element preference by metal ---
    logger.info("\n--- Donor element preference by metal (mean signed attribution) ---")
    donor_pref = defaultdict(dict)
    for (metal, donor_elem), vals in metal_donor_attrs.items():
        if len(vals) >= 5:
            donor_pref[metal][donor_elem] = {
                'mean_signed': np.mean(vals),
                'mean_abs': np.mean(np.abs(vals)),
                'count': len(vals),
            }

    logger.info(f"  {'Metal':<6} {'N(n)':>12} {'O(n)':>12} {'S(n)':>12} {'P(n)':>12}")
    for metal in sorted(donor_pref.keys()):
        parts = []
        for elem in ['N', 'O', 'S', 'P']:
            if elem in donor_pref[metal]:
                d = donor_pref[metal][elem]
                parts.append(f"{d['mean_signed']:+.2f}({d['count']})")
            else:
                parts.append(f"{'---':>7}")
        logger.info(f"  {metal:<6} {'  '.join(f'{p:>12}' for p in parts)}")

    # --- Figure ---
    if HAS_MPL and figures_dir:
        _plot_donor_attribution_heatmap(donor_pref, figures_dir)

    return role_df, donor_pref


def _plot_donor_attribution_heatmap(donor_pref, figures_dir):
    """Heatmap of mean signed donor attribution by metal x donor element."""
    donor_elems = ['N', 'O', 'S', 'P']
    # Filter metals with >= 2 donor types and reasonable count
    metals = [m for m in sorted(donor_pref.keys())
              if len(donor_pref[m]) >= 1 and
              sum(donor_pref[m].get(d, {}).get('count', 0) for d in donor_elems) >= 20]

    if len(metals) < 3:
        return

    # Sort metals by total count
    metals = sorted(metals, key=lambda m: sum(
        donor_pref[m].get(d, {}).get('count', 0) for d in donor_elems), reverse=True)[:25]

    matrix = np.full((len(metals), len(donor_elems)), np.nan)
    for i, metal in enumerate(metals):
        for j, elem in enumerate(donor_elems):
            if elem in donor_pref[metal] and donor_pref[metal][elem]['count'] >= 5:
                matrix[i, j] = donor_pref[metal][elem]['mean_signed']

    fig, ax = plt.subplots(figsize=(8, 10))
    vmax = np.nanmax(np.abs(matrix))
    im = ax.imshow(matrix, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')

    ax.set_xticks(range(len(donor_elems)))
    ax.set_xticklabels(donor_elems, fontsize=14)
    ax.set_yticks(range(len(metals)))
    ax.set_yticklabels(metals, fontsize=12)
    ax.set_xlabel('Donor Element', fontsize=14)
    ax.set_ylabel('Metal', fontsize=14)
    ax.set_title('Mean Signed Donor Attribution by Metal', fontsize=14)

    # Annotate cells
    for i in range(len(metals)):
        for j in range(len(donor_elems)):
            val = matrix[i, j]
            if not np.isnan(val):
                n = donor_pref[metals[i]].get(donor_elems[j], {}).get('count', 0)
                color = 'white' if abs(val) > 0.6 * vmax else 'black'
                ax.text(j, i, f'{val:+.2f}\n(n={n})', ha='center', va='center',
                        fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label='Mean Signed Attribution', shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'donor_attribution_heatmap.png'), dpi=300)
    plt.close()
    logger.info(f"  Saved: donor_attribution_heatmap.png")


# =============================================================================
# ANALYSIS 2: IRVING-WILLIAMS SERIES
# =============================================================================

def analyze_irving_williams(xai_df, dataset_df, figures_dir):
    """
    For ligands that bind >= 4 of {Mn, Fe, Co, Ni, Cu, Zn},
    check if predicted logK follows the Irving-Williams order.
    """
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS 2: IRVING-WILLIAMS SERIES VALIDATION")
    logger.info("=" * 60)
    logger.info("  Expected: Mn < Fe < Co < Ni < Cu > Zn")

    iw_metals = set(IRVING_WILLIAMS_ORDER)

    # Extract ligand base (SMILES without metal) for grouping
    ligand_groups = _extract_ligand_groups(xai_df, target_metals=iw_metals)

    # Filter ligands binding >= 4 IW metals
    iw_ligands = {lig: metals for lig, metals in ligand_groups.items()
                  if len(set(metals.keys()) & iw_metals) >= 4}

    logger.info(f"  Ligands binding >= 4 Irving-Williams metals: {len(iw_ligands)}")

    if not iw_ligands:
        logger.info("  No ligands found with >= 4 IW metals. Relaxing to >= 3...")
        iw_ligands = {lig: metals for lig, metals in ligand_groups.items()
                      if len(set(metals.keys()) & iw_metals) >= 3}
        logger.info(f"  Ligands binding >= 3 Irving-Williams metals: {len(iw_ligands)}")

    if not iw_ligands:
        logger.info("  Insufficient data for Irving-Williams analysis.")
        return {}

    # Check ordering for each ligand
    correct_pairs = 0
    total_pairs = 0
    cu_max_count = 0
    cu_total = 0
    ligand_results = []

    for lig_key, metals_data in iw_ligands.items():
        present_metals = sorted(
            [m for m in IRVING_WILLIAMS_ORDER if m in metals_data],
            key=lambda m: IRVING_WILLIAMS_RANK[m]
        )
        if len(present_metals) < 3:
            continue

        # Use predicted logK (from XAI results) if available, else true
        values = {}
        for m in present_metals:
            entry = metals_data[m]
            values[m] = entry.get('logK1_pred', entry.get('logK1_true', np.nan))

        # Check pairwise ordering (ascending Mn->Cu, Cu > Zn)
        for i in range(len(present_metals) - 1):
            m1, m2 = present_metals[i], present_metals[i + 1]
            if m1 == 'Zn' or m2 == 'Mn':
                continue
            # Expected: logK increases Mn < Fe < Co < Ni < Cu
            if m2 == 'Zn':
                # Cu > Zn expected
                if 'Cu' in values:
                    expected = values.get('Cu', 0) > values.get('Zn', 0)
                    total_pairs += 1
                    if expected:
                        correct_pairs += 1
            else:
                expected_increase = values[m2] > values[m1]
                total_pairs += 1
                if expected_increase:
                    correct_pairs += 1

        # Check if Cu is maximum
        if 'Cu' in values and len(values) >= 3:
            cu_total += 1
            if values['Cu'] == max(values.values()):
                cu_max_count += 1

        ligand_results.append({
            'ligand_key': lig_key[:60],
            'metals': present_metals,
            'values': {m: round(values[m], 2) for m in present_metals},
            'cu_is_max': 'Cu' in values and values['Cu'] == max(values.values()),
        })

    pairwise_accuracy = correct_pairs / total_pairs * 100 if total_pairs > 0 else 0
    cu_max_pct = cu_max_count / cu_total * 100 if cu_total > 0 else 0

    logger.info(f"\n  Results:")
    logger.info(f"    Pairwise ordering accuracy: {correct_pairs}/{total_pairs} "
                f"({pairwise_accuracy:.1f}%)")
    logger.info(f"    Cu is maximum: {cu_max_count}/{cu_total} ({cu_max_pct:.1f}%)")

    # Show top examples
    logger.info(f"\n  Example ligands (first 10):")
    for lr in ligand_results[:10]:
        vals_str = ', '.join(f"{m}={lr['values'][m]}" for m in lr['metals'])
        cu_flag = ' *Cu=max*' if lr['cu_is_max'] else ''
        logger.info(f"    {lr['ligand_key']}: {vals_str}{cu_flag}")

    # --- Figure: Box plot of predicted logK by IW metal ---
    if HAS_MPL and figures_dir:
        _plot_irving_williams(xai_df, iw_metals, figures_dir)

    summary = {
        'n_ligands_tested': len(iw_ligands),
        'pairwise_correct': correct_pairs,
        'pairwise_total': total_pairs,
        'pairwise_accuracy_pct': round(pairwise_accuracy, 1),
        'cu_is_max_count': cu_max_count,
        'cu_is_max_total': cu_total,
        'cu_is_max_pct': round(cu_max_pct, 1),
    }
    return summary


def _plot_irving_williams(xai_df, iw_metals, figures_dir):
    """Box plot of predicted logK1 across Irving-Williams metals."""
    iw_data = xai_df[xai_df['metal_type'].isin(iw_metals)].copy()
    if len(iw_data) < 20:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: All data (predicted)
    pred_col = 'logK1_pred'
    box_data = [iw_data[iw_data['metal_type'] == m][pred_col].dropna().values
                for m in IRVING_WILLIAMS_ORDER if m in iw_data['metal_type'].values]
    box_labels = [m for m in IRVING_WILLIAMS_ORDER if m in iw_data['metal_type'].values]

    medians = [np.median(d) for d in box_data]

    bp = axes[0].boxplot(box_data, tick_labels=box_labels, patch_artist=True,
                         showfliers=False, widths=0.6)
    for patch in bp['boxes']:
        patch.set_facecolor('#4C72B0')
        patch.set_alpha(0.7)
    axes[0].plot(range(1, len(medians) + 1), medians, 'ro-', markersize=8,
                 linewidth=2, label='Median')
    axes[0].set_xlabel('Metal (Irving-Williams order)', fontsize=14)
    axes[0].set_ylabel('Predicted logK1', fontsize=14)
    axes[0].set_title('(a) Predicted logK1 distribution', fontsize=14)
    axes[0].tick_params(labelsize=12)
    axes[0].legend(fontsize=12)

    # Panel B: True values
    true_col = 'logK1_true'
    if true_col in iw_data.columns:
        box_data_true = [iw_data[iw_data['metal_type'] == m][true_col].dropna().values
                         for m in IRVING_WILLIAMS_ORDER if m in iw_data['metal_type'].values]
        medians_true = [np.median(d) for d in box_data_true]

        bp2 = axes[1].boxplot(box_data_true, tick_labels=box_labels, patch_artist=True,
                              showfliers=False, widths=0.6)
        for patch in bp2['boxes']:
            patch.set_facecolor('#DD8452')
            patch.set_alpha(0.7)
        axes[1].plot(range(1, len(medians_true) + 1), medians_true, 'ro-', markersize=8,
                     linewidth=2, label='Median')
        axes[1].set_xlabel('Metal (Irving-Williams order)', fontsize=14)
        axes[1].set_ylabel('True logK1', fontsize=14)
        axes[1].set_title('(b) True logK1 distribution', fontsize=14)
        axes[1].tick_params(labelsize=12)
        axes[1].legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'irving_williams_series.png'), dpi=300)
    plt.close()
    logger.info(f"  Saved: irving_williams_series.png")


# =============================================================================
# ANALYSIS 3: HSAB VALIDATION
# =============================================================================

def analyze_hsab(xai_df, dataset_df, figures_dir):
    """
    Validate HSAB (Hard-Soft Acid-Base) patterns.

    Hard acids prefer hard bases: Fe3+, La3+, Ca2+ prefer O-donors
    Soft acids prefer soft bases: Hg2+, Pd2+, Ag+ prefer S-donors
    Borderline: Cu2+, Ni2+, Zn2+ prefer N-donors
    """
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS 3: HSAB (HARD-SOFT ACID-BASE) VALIDATION")
    logger.info("=" * 60)
    logger.info("  Expected: Hard metals + hard donors = strong binding")
    logger.info("            Soft metals + soft donors = strong binding")

    # Merge donor_elements from dataset into xai results
    if 'donor_elements' not in xai_df.columns:
        # Match by smiles
        donor_map = dataset_df.set_index('smiles')['donor_elements'].to_dict()
        xai_df = xai_df.copy()
        xai_df['donor_elements'] = xai_df['smiles'].map(donor_map)

    # Classify metals
    def classify_metal(m):
        if m in SOFT_METALS:
            return 'Soft'
        if m in HARD_METALS - BORDERLINE_METALS:
            return 'Hard'
        return 'Borderline'

    # Classify donors
    def classify_donor(d):
        if not isinstance(d, str):
            return 'Unknown'
        elems = set(d.split('+'))
        if elems <= {'O'}:
            return 'Hard (O)'
        if elems <= {'S'} or elems <= {'P'} or elems <= {'S', 'P'}:
            return 'Soft (S/P)'
        if elems <= {'N'}:
            return 'Borderline (N)'
        # Mixed donors
        if 'S' in elems or 'P' in elems:
            return 'Mixed (incl. soft)'
        return 'Mixed'

    xai_df = xai_df.copy()
    xai_df['metal_class'] = xai_df['metal_type'].apply(classify_metal)
    xai_df['donor_class'] = xai_df['donor_elements'].apply(classify_donor)

    # Crosstab: mean logK1 by metal_class x donor_class
    pred_col = 'logK1_pred'
    ct_mean = xai_df.pivot_table(values=pred_col, index='metal_class',
                                  columns='donor_class', aggfunc='mean')
    ct_count = xai_df.pivot_table(values=pred_col, index='metal_class',
                                   columns='donor_class', aggfunc='count')

    logger.info(f"\n  Mean predicted logK1 by HSAB class:")
    logger.info(f"  {ct_mean.round(2).to_string()}")
    logger.info(f"\n  Sample counts:")
    logger.info(f"  {ct_count.to_string()}")

    # Check HSAB predictions
    checks = []

    # Check 1: Soft metals should have higher logK with soft donors vs hard donors
    for mclass, dsoft, dhard in [
        ('Soft', 'Soft (S/P)', 'Hard (O)'),
        ('Soft', 'Soft (S/P)', 'Borderline (N)'),
    ]:
        if mclass in ct_mean.index and dsoft in ct_mean.columns and dhard in ct_mean.columns:
            val_soft = ct_mean.loc[mclass, dsoft]
            val_hard = ct_mean.loc[mclass, dhard]
            passed = val_soft > val_hard
            checks.append({
                'test': f'{mclass} metals prefer {dsoft} over {dhard}',
                'soft_logK': round(val_soft, 2),
                'hard_logK': round(val_hard, 2),
                'passed': passed,
            })

    # Check 2: Hard metals should have higher logK with hard donors
    for mclass, dhard, dsoft in [
        ('Hard', 'Hard (O)', 'Soft (S/P)'),
    ]:
        if mclass in ct_mean.index and dhard in ct_mean.columns and dsoft in ct_mean.columns:
            val_hard = ct_mean.loc[mclass, dhard]
            val_soft = ct_mean.loc[mclass, dsoft]
            passed = val_hard > val_soft
            checks.append({
                'test': f'{mclass} metals prefer {dhard} over {dsoft}',
                'hard_logK': round(val_hard, 2),
                'soft_logK': round(val_soft, 2),
                'passed': passed,
            })

    # Check 3: For N-donors (borderline), borderline metals should bind strongest
    dclass = 'Borderline (N)'
    if dclass in ct_mean.columns:
        n_vals = {mc: ct_mean.loc[mc, dclass] for mc in ct_mean.index
                  if dclass in ct_mean.columns and not pd.isna(ct_mean.loc[mc, dclass])}
        if n_vals:
            max_class = max(n_vals, key=n_vals.get)
            checks.append({
                'test': f'Borderline metals prefer N-donors most',
                'N_donor_logK_by_class': {k: round(v, 2) for k, v in n_vals.items()},
                'highest_class': max_class,
                'passed': max_class == 'Borderline',
            })

    logger.info(f"\n  HSAB Checks:")
    n_passed = 0
    for c in checks:
        status = 'PASS' if c['passed'] else 'FAIL'
        n_passed += c['passed']
        logger.info(f"    [{status}] {c['test']}")
        for k, v in c.items():
            if k not in ('test', 'passed'):
                logger.info(f"          {k}: {v}")

    logger.info(f"\n  HSAB Score: {n_passed}/{len(checks)} checks passed")

    # --- Figure ---
    if HAS_MPL and figures_dir:
        _plot_hsab_heatmap(ct_mean, ct_count, figures_dir)

    return {
        'checks': checks,
        'n_passed': n_passed,
        'n_total': len(checks),
        'crosstab_mean': ct_mean.round(2).to_dict(),
    }


def _plot_hsab_heatmap(ct_mean, ct_count, figures_dir):
    """Heatmap of mean logK1 by HSAB metal class x donor class."""
    # Reorder
    row_order = ['Hard', 'Borderline', 'Soft']
    col_order = ['Hard (O)', 'Borderline (N)', 'Soft (S/P)', 'Mixed (incl. soft)', 'Mixed']
    row_order = [r for r in row_order if r in ct_mean.index]
    col_order = [c for c in col_order if c in ct_mean.columns]

    matrix = ct_mean.reindex(index=row_order, columns=col_order).values
    counts = ct_count.reindex(index=row_order, columns=col_order).fillna(0).astype(int).values

    fig, ax = plt.subplots(figsize=(10, 5))
    vmax = np.nanmax(np.abs(matrix))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(col_order, fontsize=12, rotation=30, ha='right')
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels(row_order, fontsize=14)
    ax.set_xlabel('Donor Class', fontsize=14)
    ax.set_ylabel('Metal Class (HSAB)', fontsize=14)
    ax.set_title('Mean Predicted logK1 by HSAB Classification', fontsize=14)

    for i in range(len(row_order)):
        for j in range(len(col_order)):
            val = matrix[i, j]
            n = counts[i, j]
            if not np.isnan(val):
                color = 'white' if val > 0.7 * np.nanmax(matrix) else 'black'
                ax.text(j, i, f'{val:.1f}\n(n={n})', ha='center', va='center',
                        fontsize=10, color=color)

    plt.colorbar(im, ax=ax, label='Mean logK1', shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'hsab_heatmap.png'), dpi=300)
    plt.close()
    logger.info(f"  Saved: hsab_heatmap.png")


# =============================================================================
# ANALYSIS 4: LANTHANIDE CONTRACTION
# =============================================================================

def analyze_lanthanide_contraction(xai_df, figures_dir):
    """
    For ligands binding >= 5 lanthanides, check if predicted logK
    increases from La to Lu (lanthanide contraction = smaller ion = tighter binding).
    """
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS 4: LANTHANIDE CONTRACTION")
    logger.info("=" * 60)
    logger.info("  Expected: logK generally increases La -> Lu (smaller ions bind tighter)")

    ln_data = xai_df[xai_df['metal_type'].isin(LANTHANIDE_SET)].copy()
    if len(ln_data) < 20:
        logger.info("  Insufficient lanthanide data.")
        return {}

    # Group by ligand to find shared ligands
    ligand_groups = _extract_ligand_groups(xai_df, target_metals=LANTHANIDE_SET)

    # Filter ligands with >= 5 lanthanides
    ln_ligands = {lig: metals for lig, metals in ligand_groups.items()
                  if len(set(metals.keys()) & LANTHANIDE_SET) >= 5}

    logger.info(f"  Ligands binding >= 5 lanthanides: {len(ln_ligands)}")

    if not ln_ligands:
        logger.info("  Relaxing to >= 3...")
        ln_ligands = {lig: metals for lig, metals in ligand_groups.items()
                      if len(set(metals.keys()) & LANTHANIDE_SET) >= 3}
        logger.info(f"  Ligands binding >= 3 lanthanides: {len(ln_ligands)}")

    if not ln_ligands:
        logger.info("  Insufficient data for lanthanide contraction analysis.")
        return {}

    # Check Spearman correlation: atomic number vs logK for each ligand
    from scipy import stats
    correlations = []
    positive_trend_count = 0
    total_tested = 0

    for lig_key, metals_data in ln_ligands.items():
        ln_metals = [(m, d) for m, d in metals_data.items() if m in LANTHANIDE_SET]
        if len(ln_metals) < 3:
            continue

        ranks = [LANTHANIDE_RANK[m] for m, _ in ln_metals]
        logks = [d.get('logK1_pred', d.get('logK1_true', np.nan)) for _, d in ln_metals]

        if any(np.isnan(logks)):
            continue

        rho, pval = stats.spearmanr(ranks, logks)
        correlations.append(rho)
        total_tested += 1
        if rho > 0:
            positive_trend_count += 1

    if not correlations:
        logger.info("  Could not compute correlations.")
        return {}

    mean_rho = np.mean(correlations)
    median_rho = np.median(correlations)
    positive_pct = positive_trend_count / total_tested * 100

    logger.info(f"\n  Results ({total_tested} ligands tested):")
    logger.info(f"    Mean Spearman rho (La->Lu rank vs logK): {mean_rho:.3f}")
    logger.info(f"    Median Spearman rho: {median_rho:.3f}")
    logger.info(f"    Positive trend (logK increases La->Lu): "
                f"{positive_trend_count}/{total_tested} ({positive_pct:.1f}%)")

    # --- Figure: Aggregate mean logK across lanthanide series ---
    if HAS_MPL and figures_dir:
        _plot_lanthanide_contraction(xai_df, figures_dir)

    return {
        'n_ligands_tested': total_tested,
        'mean_spearman_rho': round(mean_rho, 3),
        'median_spearman_rho': round(median_rho, 3),
        'positive_trend_pct': round(positive_pct, 1),
    }


def _plot_lanthanide_contraction(xai_df, figures_dir):
    """Plot mean logK1 across the lanthanide series."""
    ln_data = xai_df[xai_df['metal_type'].isin(LANTHANIDE_SET)].copy()
    if len(ln_data) < 20:
        return

    # Order by lanthanide position
    ln_data['ln_rank'] = ln_data['metal_type'].map(LANTHANIDE_RANK)
    ln_data = ln_data.dropna(subset=['ln_rank'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, (col, title) in enumerate([('logK1_pred', 'Predicted'),
                                            ('logK1_true', 'True')]):
        if col not in ln_data.columns:
            continue
        grouped = ln_data.groupby('metal_type')[col].agg(['mean', 'std', 'count'])
        # Reorder
        ordered_metals = [m for m in LANTHANIDES if m in grouped.index]
        grouped = grouped.loc[ordered_metals]

        x = range(len(ordered_metals))
        axes[ax_idx].errorbar(x, grouped['mean'], yerr=grouped['std'] / np.sqrt(grouped['count']),
                               fmt='o-', capsize=4, markersize=8, linewidth=2, color='#4C72B0')
        axes[ax_idx].set_xticks(x)
        axes[ax_idx].set_xticklabels(ordered_metals, fontsize=10, rotation=45)
        axes[ax_idx].set_xlabel('Lanthanide (La → Lu)', fontsize=14)
        axes[ax_idx].set_ylabel(f'{title} logK1', fontsize=14)
        axes[ax_idx].set_title(f'({chr(97 + ax_idx)}) {title} logK1 across lanthanide series',
                                fontsize=14)
        axes[ax_idx].tick_params(labelsize=11)

        # Add count annotations
        for xi, m in enumerate(ordered_metals):
            n = int(grouped.loc[m, 'count'])
            axes[ax_idx].annotate(f'n={n}', (xi, grouped.loc[m, 'mean']),
                                   textcoords='offset points', xytext=(0, 12),
                                   fontsize=8, ha='center', color='gray')

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'lanthanide_contraction.png'), dpi=300)
    plt.close()
    logger.info(f"  Saved: lanthanide_contraction.png")


# =============================================================================
# ANALYSIS 5: METAL SELECTIVITY (SHARED LIGANDS)
# =============================================================================

def analyze_selectivity(xai_df, dataset_df, figures_dir):
    """
    For ligands that bind to both metals in a selectivity pair,
    compute logK difference and identify structural drivers.
    """
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS 5: METAL SELECTIVITY (SHARED LIGANDS)")
    logger.info("=" * 60)

    # Merge donor_elements from dataset
    if 'donor_elements' not in xai_df.columns:
        donor_map = dataset_df.set_index('smiles')['donor_elements'].to_dict()
        xai_df = xai_df.copy()
        xai_df['donor_elements'] = xai_df['smiles'].map(donor_map)

    all_metals = set(xai_df['metal_type'].unique())
    ligand_groups = _extract_ligand_groups(xai_df, target_metals=all_metals)

    results = {}

    for metal_a, metal_b, description in SELECTIVITY_PAIRS:
        if metal_a not in all_metals or metal_b not in all_metals:
            continue

        # Find shared ligands
        shared = {}
        for lig_key, metals_data in ligand_groups.items():
            if metal_a in metals_data and metal_b in metals_data:
                shared[lig_key] = metals_data

        if len(shared) < 3:
            logger.info(f"\n  {metal_a}/{metal_b} ({description}): "
                        f"only {len(shared)} shared ligands, skipping")
            continue

        # Compute selectivity = logK(metal_a) - logK(metal_b)
        selectivities = []
        donor_selectivity = defaultdict(list)

        for lig_key, metals_data in shared.items():
            logk_a = metals_data[metal_a].get('logK1_pred',
                     metals_data[metal_a].get('logK1_true', np.nan))
            logk_b = metals_data[metal_b].get('logK1_pred',
                     metals_data[metal_b].get('logK1_true', np.nan))
            donor = metals_data[metal_a].get('donor_elements', 'Unknown')

            if np.isnan(logk_a) or np.isnan(logk_b):
                continue

            sel = logk_a - logk_b
            selectivities.append(sel)
            donor_selectivity[donor].append(sel)

        if not selectivities:
            continue

        mean_sel = np.mean(selectivities)
        std_sel = np.std(selectivities)
        pct_positive = sum(1 for s in selectivities if s > 0) / len(selectivities) * 100

        logger.info(f"\n  {metal_a}/{metal_b} ({description}):")
        logger.info(f"    Shared ligands: {len(selectivities)}")
        logger.info(f"    Mean selectivity (logK_{metal_a} - logK_{metal_b}): "
                    f"{mean_sel:+.2f} +/- {std_sel:.2f}")
        logger.info(f"    {metal_a}-selective: {pct_positive:.1f}%")

        # Donor-specific selectivity
        logger.info(f"    By donor type:")
        for donor in sorted(donor_selectivity.keys()):
            vals = donor_selectivity[donor]
            if len(vals) >= 2:
                logger.info(f"      {donor}: mean={np.mean(vals):+.2f} "
                            f"(n={len(vals)}, {sum(1 for v in vals if v > 0)/len(vals)*100:.0f}% "
                            f"{metal_a}-selective)")

        results[f"{metal_a}_vs_{metal_b}"] = {
            'description': description,
            'n_shared_ligands': len(selectivities),
            'mean_selectivity': round(mean_sel, 3),
            'std_selectivity': round(std_sel, 3),
            'pct_metal_a_selective': round(pct_positive, 1),
            'by_donor': {d: {'mean': round(np.mean(v), 3), 'n': len(v)}
                         for d, v in donor_selectivity.items() if len(v) >= 2},
        }

    # --- Figure ---
    if HAS_MPL and figures_dir and results:
        _plot_selectivity(results, figures_dir)

    return results


def _plot_selectivity(results, figures_dir):
    """Bar chart of mean selectivity by metal pair, colored by donor type."""
    pairs = list(results.keys())
    if not pairs:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    x_positions = []
    x_labels = []
    bar_data = []

    for i, pair in enumerate(pairs):
        r = results[pair]
        donor_data = r.get('by_donor', {})
        metals = pair.split('_vs_')
        label = f"{metals[0]}/{metals[1]}"

        if not donor_data:
            bar_data.append((i, r['mean_selectivity'], r['n_shared_ligands'], 'All', label))
        else:
            for j, (donor, dinfo) in enumerate(sorted(donor_data.items())):
                bar_data.append((i, dinfo['mean'], dinfo['n'], donor, label))

    # Group bars
    colors = {'N': '#4C72B0', 'O': '#DD8452', 'S': '#55A868', 'P': '#C44E52',
              'N+O': '#8172B3', 'N+S': '#937860', 'O+S': '#DA8BC3', 'All': '#8C8C8C',
              'N+P': '#CCB974', 'O+P': '#64B5CD', 'P+S': '#AABBCC'}

    unique_pairs = list(dict.fromkeys(bd[4] for bd in bar_data))
    pair_positions = {p: i for i, p in enumerate(unique_pairs)}

    width = 0.12
    donor_types_seen = list(dict.fromkeys(bd[3] for bd in bar_data))
    n_donors = len(donor_types_seen)
    donor_offsets = {d: (j - n_donors / 2 + 0.5) * width for j, d in enumerate(donor_types_seen)}

    for pos, mean_sel, n, donor, pair_label in bar_data:
        x = pair_positions[pair_label] + donor_offsets.get(donor, 0)
        color = colors.get(donor, '#8C8C8C')
        ax.bar(x, mean_sel, width=width, color=color, edgecolor='black', linewidth=0.5)
        ax.text(x, mean_sel + (0.1 if mean_sel >= 0 else -0.3), f'n={n}',
                ha='center', fontsize=7, rotation=90)

    ax.set_xticks(range(len(unique_pairs)))
    ax.set_xticklabels(unique_pairs, fontsize=12)
    ax.set_ylabel('Mean Selectivity (logK difference)', fontsize=14)
    ax.set_title('Metal Selectivity by Donor Type', fontsize=14)
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax.tick_params(labelsize=11)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors.get(d, '#8C8C8C'), label=d)
                       for d in donor_types_seen]
    ax.legend(handles=legend_elements, title='Donor Type', fontsize=10, title_fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'metal_selectivity.png'), dpi=300)
    plt.close()
    logger.info(f"  Saved: metal_selectivity.png")


# =============================================================================
# HELPER: EXTRACT LIGAND GROUPS (SHARED LIGANDS ACROSS METALS)
# =============================================================================

def _extract_ligand_groups(xai_df, target_metals=None):
    """
    Group complexes by ligand identity (SMILES with metal removed).

    Returns:
        dict: ligand_key -> {metal_type: {logK1_pred, logK1_true, smiles, donor_elements, ...}}
    """
    ligand_groups = defaultdict(dict)

    for _, row in xai_df.iterrows():
        smiles = row.get('smiles', '')
        metal = row.get('metal_type', '')

        if target_metals and metal not in target_metals:
            continue

        lig_key = _extract_ligand_key(smiles)
        if lig_key is None:
            continue

        entry = {
            'smiles': smiles,
            'metal_type': metal,
        }
        for col in ['logK1_pred', 'logK1_true', 'uncertainty', 'scenario',
                     'donor_elements', 'agreement_ratio', 'mean_attr']:
            if col in row.index and pd.notna(row[col]):
                entry[col] = row[col]

        ligand_groups[lig_key][metal] = entry

    return ligand_groups


def _extract_ligand_key(smiles):
    """
    Extract a canonical ligand identifier by removing the metal atom.

    This allows grouping the same ligand across different metals.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        metal_set = {
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
            'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
            'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho',
            'Er', 'Tm', 'Yb', 'Lu', 'Th', 'U', 'Np', 'Pu', 'Am', 'Cm',
            'Bk', 'Cf', 'Al', 'Ga', 'In', 'Sn', 'Pb', 'Bi',
            'Mg', 'Ca', 'Sr', 'Ba', 'Li', 'Na', 'Ra',
        }

        # Find metal atom indices
        metal_indices = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() in metal_set:
                metal_indices.append(atom.GetIdx())

        if not metal_indices:
            return None

        # Remove metal atoms (reverse order to preserve indices)
        emol = Chem.RWMol(mol)
        for idx in sorted(metal_indices, reverse=True):
            emol.RemoveAtom(idx)

        try:
            Chem.SanitizeMol(emol)
            return Chem.MolToSmiles(emol)
        except Exception:
            # If sanitization fails, use a simpler key
            return Chem.MolToSmiles(emol, canonical=False)

    except Exception:
        return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Post-XAI chemical validation for logK1 stability constants'
    )
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Path to xai_results/ directory containing xai_results_logK1.csv '
                             'and xai_attributions_logK1.json')
    parser.add_argument('--dataset_csv', type=str, default=None,
                        help='Path to stability_constants_dative_clean.csv '
                             '(default: auto-detect from ../data/)')
    parser.add_argument('--figures_dir', type=str, default=None,
                        help='Directory for output figures (default: results_dir/figures/)')
    args = parser.parse_args()

    # Auto-detect dataset CSV
    if args.dataset_csv is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.dataset_csv = os.path.join(script_dir, 'data',
                                        'stability_constants_dative_clean.csv')
        if not os.path.exists(args.dataset_csv):
            # Try relative to results dir
            args.dataset_csv = os.path.join(os.path.dirname(args.results_dir),
                                            'data', 'stability_constants_dative_clean.csv')

    if args.figures_dir is None:
        args.figures_dir = os.path.join(args.results_dir, 'chemistry_validation')

    os.makedirs(args.figures_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("POST-XAI CHEMICAL VALIDATION")
    logger.info("=" * 60)
    logger.info(f"Results: {args.results_dir}")
    logger.info(f"Dataset: {args.dataset_csv}")
    logger.info(f"Figures: {args.figures_dir}")

    # Load data
    xai_df, xai_attrs, dataset_df = load_data(args.results_dir, args.dataset_csv)

    # Run all analyses
    all_results = {}

    # 1. Donor attribution
    role_df, donor_pref = analyze_donor_attributions(xai_df, xai_attrs, args.figures_dir)
    all_results['donor_attribution'] = {
        'donor_dominance': bool(
            role_df['donor_mean_abs'].mean() > role_df['backbone_mean_abs'].mean()
        ) if len(role_df) > 0 else None,
        'mean_donor_abs_attr': round(role_df['donor_mean_abs'].mean(), 4) if len(role_df) > 0 else None,
        'mean_backbone_abs_attr': round(role_df['backbone_mean_abs'].mean(), 4) if len(role_df) > 0 else None,
    }

    # 2. Irving-Williams
    iw_results = analyze_irving_williams(xai_df, dataset_df, args.figures_dir)
    all_results['irving_williams'] = iw_results

    # 3. HSAB
    hsab_results = analyze_hsab(xai_df, dataset_df, args.figures_dir)
    all_results['hsab'] = {
        'n_passed': hsab_results.get('n_passed'),
        'n_total': hsab_results.get('n_total'),
        'checks': hsab_results.get('checks'),
    }

    # 4. Lanthanide contraction
    ln_results = analyze_lanthanide_contraction(xai_df, args.figures_dir)
    all_results['lanthanide_contraction'] = ln_results

    # 5. Selectivity
    sel_results = analyze_selectivity(xai_df, dataset_df, args.figures_dir)
    all_results['selectivity'] = sel_results

    # Save summary
    summary_path = os.path.join(args.figures_dir, 'chemistry_validation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nSummary saved: {summary_path}")

    # Final scorecard
    logger.info("\n" + "=" * 60)
    logger.info("CHEMICAL VALIDATION SCORECARD")
    logger.info("=" * 60)

    score = 0
    total = 0

    # Donor dominance
    total += 1
    if all_results['donor_attribution'].get('donor_dominance'):
        score += 1
        logger.info("  [PASS] Donor atoms have higher |attribution| than backbone")
    else:
        logger.info("  [FAIL] Donor atoms do NOT dominate attribution")

    # Irving-Williams
    if iw_results:
        total += 1
        if iw_results.get('pairwise_accuracy_pct', 0) >= 60:
            score += 1
            logger.info(f"  [PASS] Irving-Williams pairwise accuracy: "
                        f"{iw_results['pairwise_accuracy_pct']}%")
        else:
            logger.info(f"  [FAIL] Irving-Williams pairwise accuracy: "
                        f"{iw_results.get('pairwise_accuracy_pct', 0)}%")
        total += 1
        if iw_results.get('cu_is_max_pct', 0) >= 60:
            score += 1
            logger.info(f"  [PASS] Cu is maximum: {iw_results['cu_is_max_pct']}%")
        else:
            logger.info(f"  [FAIL] Cu is maximum: {iw_results.get('cu_is_max_pct', 0)}%")

    # HSAB
    if hsab_results.get('n_total', 0) > 0:
        total += 1
        if hsab_results['n_passed'] >= hsab_results['n_total'] / 2:
            score += 1
            logger.info(f"  [PASS] HSAB: {hsab_results['n_passed']}/{hsab_results['n_total']}")
        else:
            logger.info(f"  [FAIL] HSAB: {hsab_results['n_passed']}/{hsab_results['n_total']}")

    # Lanthanide contraction
    if ln_results:
        total += 1
        if ln_results.get('positive_trend_pct', 0) >= 50:
            score += 1
            logger.info(f"  [PASS] Lanthanide contraction positive trend: "
                        f"{ln_results['positive_trend_pct']}%")
        else:
            logger.info(f"  [FAIL] Lanthanide contraction positive trend: "
                        f"{ln_results.get('positive_trend_pct', 0)}%")

    logger.info(f"\n  TOTAL SCORE: {score}/{total}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
