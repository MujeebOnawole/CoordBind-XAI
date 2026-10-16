"""
Scripted re-run of the CoordBind-XAI vs Zhang et al. 2026 DFT external
validation (manuscript Section 3.12, Table 8 and Table S13).

It calls the exact functions the deployed web tool uses (predict_donors,
build_dative_smiles, ensemble_predict, predict_with_xai / _classify_scenario)
so the numbers are reproducible and not subject to manual transcription error.

The web tool is maintained in a separate repository (it carries the model
checkpoints and the Gradio app) and is not vendored here. Clone it first:

  git clone https://huggingface.co/spaces/catenate/metal-ligand-binding

then point this script at the clone and run it from the repository root:

  export COORDBIND_WEBTOOL_DIR=/path/to/metal-ligand-binding
  LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6 python src/validation/run_dft_external_validation.py

(LD_PRELOAD is only needed in conda environments where the system libstdc++
is older than the one PyTorch Geometric was built against.)

Output: results/validation/dft_validation_results.json
"""
import sys, os, json

HF_SPACE_DIR = os.environ.get("COORDBIND_WEBTOOL_DIR")
if not HF_SPACE_DIR or not os.path.isfile(os.path.join(HF_SPACE_DIR, "app.py")):
    sys.exit(
        "Set COORDBIND_WEBTOOL_DIR to a local clone of "
        "https://huggingface.co/spaces/catenate/metal-ligand-binding (app.py not found)."
    )
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT_PATH = os.path.join(REPO_ROOT, "results", "validation", "dft_validation_results.json")
sys.path.insert(0, HF_SPACE_DIR)
os.chdir(HF_SPACE_DIR)

import numpy as np
from scipy.stats import spearmanr
import app  # loads all 5 ensembles + coord_only ensemble, exactly as the deployed Space does

DONOR_THRESHOLD = app.COMPARISON_DONOR_THRESHOLD  # 0.75, same as Design Mode/Metal Comparison


def predict_metal(ligand_smiles, donor_indices, metal, charge):
    """Full Scenario A-D prediction for one metal, via predict_with_xai --
    identical call to what the Metal Comparison 'Include trust Scenario'
    toggle uses."""
    complex_smiles = app.build_dative_smiles(ligand_smiles, donor_indices, metal, charge)
    if complex_smiles is None:
        return {'metal': metal, 'charge': charge, 'status': 'SMILES construction failed'}

    result = app.predict_with_xai(
        models=app.MODELS,
        smiles=complex_smiles,
        device=app.DEVICE,
        property_names=app.PROPERTY_NAMES,
        construct_graph_fn=app.construct_tmc_graph,
    )
    if not result or 'error' in result:
        return {'metal': metal, 'charge': charge, 'status': 'Prediction failed', 'smiles': complex_smiles}

    pred = result['predictions']['logK1']
    scenario = result['scenarios']['logK1']
    scen_detail = result['scenario_details']['logK1']
    stab_label, _ = app._stability_label(pred['mean'])
    return {
        'metal': metal,
        'charge': charge,
        'status': 'OK',
        'complex_smiles': complex_smiles,
        'logK1_mean': round(float(pred['mean']), 3),
        'logK1_std': round(float(pred['std']), 3),
        'cv': round(float(pred['cv']), 4),
        'scenario': scenario,
        'scenario_description': scen_detail['description'],
        'binding_category': stab_label,
        'num_coord_sites': result['metadata'].get('num_coord_sites'),
    }


def run_ligand(name, ligand_smiles, metals, donor_threshold=DONOR_THRESHOLD, manual_donor_indices=None):
    """metals: list of (element, charge) tuples.

    manual_donor_indices: if given, OVERRIDES the automatic donor prediction
    with this exact atom-index list (used only for BQ, where the donor model
    flags four aromatic ring CH carbons as high-confidence donors alongside
    the two real carbonyl oxygens -- a chemically implausible prediction we
    do not want silently driving the ranking test). The automatic prediction
    is still run and recorded, for transparency about the model's failure
    mode there.
    """
    donor_probs, auto_donor_indices, graph_data = app.predict_donors(
        models=app.COORD_MODELS, smiles=ligand_smiles, threshold=donor_threshold, device=app.DEVICE
    )
    atom_symbols = graph_data.atom_symbols if (graph_data is not None and hasattr(graph_data, 'atom_symbols')) else []
    all_probs = (
        [(atom_symbols[i], round(float(p), 3)) for i, p in enumerate(donor_probs)]
        if donor_probs is not None else []
    )

    donor_indices = manual_donor_indices if manual_donor_indices is not None else auto_donor_indices
    donor_syms = [atom_symbols[i] if i < len(atom_symbols) else '?' for i in (donor_indices or [])]
    donor_p = [round(float(donor_probs[i]), 3) for i in (donor_indices or [])] if donor_probs is not None else []

    rows = []
    for el, charge in metals:
        r = predict_metal(ligand_smiles, donor_indices, el, charge)
        rows.append(r)

    return {
        'ligand': name,
        'ligand_smiles': ligand_smiles,
        'donor_threshold': donor_threshold,
        'donor_source': 'manual_override' if manual_donor_indices is not None else 'automatic',
        'predicted_donor_atoms': donor_syms,
        'predicted_donor_probs': donor_p,
        'automatic_donor_indices': list(auto_donor_indices) if auto_donor_indices else [],
        'automatic_donor_atoms': [atom_symbols[i] for i in (auto_donor_indices or [])],
        'all_atom_donor_probs': all_probs,
        'predicted_donor_indices': list(donor_indices) if donor_indices else [],
        'metals': rows,
    }


def spearman_pairwise(order_labels, coordbind_scores, ref_scores_higher_is_stronger):
    """
    order_labels: list of metal labels in the same order as coordbind_scores/ref_scores.
    ref_scores_higher_is_stronger: reference values already sign-flipped so that
      HIGHER = stronger (i.e. -1 * DFT eV, or raw logKstab).
    Returns (spearman_rho, spearman_p, pairwise_pct, n_pairs, n_concordant).
    """
    cb = np.array(coordbind_scores, dtype=float)
    ref = np.array(ref_scores_higher_is_stronger, dtype=float)
    rho, p = spearmanr(cb, ref)

    n = len(cb)
    concordant = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            cb_sign = np.sign(cb[i] - cb[j])
            ref_sign = np.sign(ref[i] - ref[j])
            if cb_sign == ref_sign or cb_sign == 0 or ref_sign == 0:
                concordant += 1
    pct = 100.0 * concordant / total if total else float('nan')
    return rho, p, pct, total, concordant


def top1_top3(order_labels, coordbind_scores, ref_scores_higher_is_stronger, k=3):
    cb_ranked = [x for _, x in sorted(zip(coordbind_scores, order_labels), reverse=True)]
    ref_ranked = [x for _, x in sorted(zip(ref_scores_higher_is_stronger, order_labels), reverse=True)]
    top1_agree = cb_ranked[0] == ref_ranked[0]
    topk_overlap = len(set(cb_ranked[:k]) & set(ref_ranked[:k]))
    return top1_agree, topk_overlap, cb_ranked, ref_ranked


# =============================================================================
# TEST DEFINITIONS (reference values transcribed from Zhang et al. 2026, J. Phys. Chem. A)
# =============================================================================

TESTS = {}

TESTS['BPY'] = dict(
    ligand_smiles='n1ccccc1-c2ccccn2',
    metals=[('Mg', 2), ('Mn', 2), ('Fe', 2), ('Co', 2), ('Ni', 2), ('Zn', 2), ('Cd', 2), ('Pb', 2)],
    exp_logKstab={'Mg': 0.28, 'Mn': 2.62, 'Fe': 4.20, 'Co': 5.80, 'Ni': 7.04, 'Zn': 5.13, 'Cd': 4.18, 'Pb': 2.90},
    dft_BE_eV={'Mg': -1.36, 'Mn': -1.65, 'Fe': -1.92, 'Co': -2.12, 'Ni': -2.30, 'Zn': -1.95, 'Cd': -1.79, 'Pb': -1.59},
)

TESTS['PHEN'] = dict(
    ligand_smiles='c1ccc2nc3ccccc3nc2c1',
    metals=[('Mg', 2), ('Mn', 2), ('Fe', 2), ('Co', 2), ('Ni', 2), ('Zn', 2), ('Cd', 2), ('Pb', 2)],
    exp_logKstab={'Mg': 1.20, 'Mn': 4.00, 'Fe': 5.85, 'Co': 7.08, 'Ni': 8.60, 'Zn': 6.40, 'Cd': 5.80, 'Pb': 4.65},
    dft_BE_eV={'Mg': -1.44, 'Mn': -1.73, 'Fe': -1.92, 'Co': -2.17, 'Ni': -2.35, 'Zn': -2.02, 'Cd': -1.88, 'Pb': -1.67},
)

TESTS['BQ'] = dict(
    ligand_smiles='O=C1C=CC=CC1=O',
    metals=[('Mg', 2), ('Mn', 2), ('Fe', 2), ('Co', 2), ('Ni', 2), ('Zn', 2), ('Cd', 2), ('Cu', 2), ('Al', 3), ('Ga', 3)],
    exp_logKstab={'Mg': 5.70, 'Mn': 7.72, 'Fe': 7.95, 'Co': 8.60, 'Ni': 8.92, 'Zn': 9.90, 'Cd': 8.20, 'Cu': 13.90, 'Al': 16.30, 'Ga': 18.90},
    # No DFT BEcal reported in Table S1 for Fe2+ / Cd2+
    dft_BE_eV={'Mg': -0.462, 'Mn': -0.571, 'Co': -0.943, 'Ni': -0.862, 'Zn': -0.664, 'Cu': -1.280, 'Al': -1.665, 'Ga': -1.807},
    # The automatic donor model flags SIX atoms >=0.75 for this ligand: both
    # real carbonyl oxygens (idx 0, 7; P=0.836) AND all four ring CH carbons
    # (idx 2-5; P=0.969-0.972) -- chemically implausible (plain vinylic CH
    # carbons are not donor atoms) and reported separately as a limitation.
    # Manually override with the chemically correct O,O bidentate site
    # (idx 0, 7) for the ranking test itself.
    manual_donor_indices=[0, 7],
)

TESTS['BDT'] = dict(
    ligand_smiles='Sc1ccccc1S',
    metals=[('Ga', 3), ('Pt', 2), ('Pd', 2), ('Ni', 2), ('Al', 3), ('Co', 2), ('Cd', 2), ('Fe', 2), ('Zn', 2), ('Mn', 2), ('Mg', 2)],
    exp_logKstab=None,  # not reported for BDT in the test file
    dft_BE_eV={'Ga': -1.289, 'Pt': -1.267, 'Pd': -1.117, 'Ni': -0.582, 'Al': -0.401, 'Co': -0.381, 'Cd': -0.243, 'Fe': -0.188, 'Zn': -0.173, 'Mn': 0.435, 'Mg': 0.477},
    # At the default 0.75 threshold BOTH thiol sulfurs score 0.576 -- below
    # cutoff, so no complex could be built at all. This is itself a
    # reportable finding (donor confidence for S is systematically lower,
    # consistent with S being only ~3% of the binding training set vs 6% in
    # Toney CSD). Using 0.50 (Design Mode's documented minimum, not an ad hoc
    # value) correctly recovers both S atoms as donors, matching the
    # bidentate S,S structure Zhang et al. describe.
    donor_threshold=0.50,
)

# Targeted spot-check from the main text prose (no full metal panel published):
# "metal ions Pd2+ and Pt2+ have stronger binding to BDT (-1.117 and -1.267 eV)
#  than to AP (-1.036 and -1.166 eV)" (Zhang et al. 2026, Section 3.3.2)
TESTS['AP'] = dict(
    ligand_smiles='Nc1ccccc1S',
    metals=[('Pd', 2), ('Pt', 2)],
    exp_logKstab=None,
    dft_BE_eV={'Pd': -1.036, 'Pt': -1.166},
    # N clears 0.75 (P=0.988) but the thiol S scores 0.713 -- just below
    # cutoff, so at the default threshold only N would be used (monodentate),
    # not the bidentate S,N site Zhang et al. describe. Same 0.50 threshold
    # as BDT recovers both donors.
    donor_threshold=0.50,
)


def main():
    all_results = {}

    # BQ run using the RAW automatic donor prediction (all 6 flagged atoms),
    # kept separate from the main TESTS loop so it doesn't feed into the
    # headline ranking metrics -- this is purely to document and quantify
    # the donor-model failure mode noted above.
    bq_spec = TESTS['BQ']
    print(f"\n{'='*70}\nBQ (automatic donor prediction, uncorrected)\n{'='*70}")
    bq_auto = run_ligand('BQ_auto', bq_spec['ligand_smiles'], bq_spec['metals'])
    bq_auto['exp_logKstab'] = bq_spec.get('exp_logKstab')
    bq_auto['dft_BE_eV'] = bq_spec.get('dft_BE_eV')
    for r in bq_auto['metals']:
        print(f"  {r['metal']}{r['charge']:+d}: {r.get('status')}"
              + (f" logK1={r['logK1_mean']:.2f}+/-{r['logK1_std']:.2f} Scenario={r['scenario']}"
                 if r.get('status') == 'OK' else ""))
    all_results['BQ_auto_uncorrected'] = bq_auto

    for name, spec in TESTS.items():
        print(f"\n{'='*70}\n{name}\n{'='*70}")
        run = run_ligand(name, spec['ligand_smiles'], spec['metals'],
                          donor_threshold=spec.get('donor_threshold', DONOR_THRESHOLD),
                          manual_donor_indices=spec.get('manual_donor_indices'))
        print(f"Predicted donor atoms: {run['predicted_donor_atoms']} (indices {run['predicted_donor_indices']})")
        for r in run['metals']:
            print(f"  {r['metal']}{r['charge']:+d}: {r.get('status')}"
                  + (f" logK1={r['logK1_mean']:.2f}+/-{r['logK1_std']:.2f} CV={r['cv']:.3f} "
                     f"Scenario={r['scenario']} ({r['binding_category']})" if r.get('status') == 'OK' else ""))
        run['exp_logKstab'] = spec.get('exp_logKstab')
        run['dft_BE_eV'] = spec.get('dft_BE_eV')
        all_results[name] = run

    # ---------------------------------------------------------------
    # Ranking metrics
    # ---------------------------------------------------------------
    summary = {}
    for name, run in all_results.items():
        ok_rows = {r['metal']: r for r in run['metals'] if r.get('status') == 'OK'}
        metrics = {}

        # vs DFT (restrict to metals with both a CoordBind prediction and a reported DFT value)
        dft = run['dft_BE_eV'] or {}
        common_dft = [m for m in ok_rows if m in dft]
        if len(common_dft) >= 3:
            cb_scores = [ok_rows[m]['logK1_mean'] for m in common_dft]
            ref_scores = [-dft[m] for m in common_dft]  # flip sign: higher = stronger
            rho, p, pct, n_pairs, n_conc = spearman_pairwise(common_dft, cb_scores, ref_scores)
            top1, top3, cb_rank, ref_rank = top1_top3(common_dft, cb_scores, ref_scores)
            metrics['vs_DFT'] = {
                'n_metals': len(common_dft), 'spearman_rho': round(rho, 3), 'spearman_p': round(p, 4),
                'pairwise_pct': round(pct, 1), 'n_pairs': n_pairs, 'n_concordant': n_conc,
                'top1_agree': top1, 'top3_overlap': top3,
                'coordbind_ranking': cb_rank, 'dft_ranking': ref_rank,
            }

        # vs experiment
        exp = run['exp_logKstab'] or {}
        common_exp = [m for m in ok_rows if m in exp]
        if len(common_exp) >= 3:
            cb_scores = [ok_rows[m]['logK1_mean'] for m in common_exp]
            ref_scores = [exp[m] for m in common_exp]  # already higher = stronger
            rho, p, pct, n_pairs, n_conc = spearman_pairwise(common_exp, cb_scores, ref_scores)
            top1, top3, cb_rank, ref_rank = top1_top3(common_exp, cb_scores, ref_scores)
            metrics['vs_experiment'] = {
                'n_metals': len(common_exp), 'spearman_rho': round(rho, 3), 'spearman_p': round(p, 4),
                'pairwise_pct': round(pct, 1), 'n_pairs': n_pairs, 'n_concordant': n_conc,
                'top1_agree': top1, 'top3_overlap': top3,
                'coordbind_ranking': cb_rank, 'experiment_ranking': ref_rank,
            }

        # Scenario A/B vs C/D subset agreement (vs DFT, where available)
        if len(common_dft) >= 3:
            scenA_B = [m for m in common_dft if ok_rows[m]['scenario'] in ('A', 'B')]
            scenC_D = [m for m in common_dft if ok_rows[m]['scenario'] in ('C', 'D')]
            for subset_name, subset in [('scenario_AB', scenA_B), ('scenario_CD', scenC_D)]:
                if len(subset) >= 3:
                    cb_scores = [ok_rows[m]['logK1_mean'] for m in subset]
                    ref_scores = [-dft[m] for m in subset]
                    rho, p, pct, n_pairs, n_conc = spearman_pairwise(subset, cb_scores, ref_scores)
                    metrics[subset_name + '_vs_DFT'] = {
                        'n_metals': len(subset), 'spearman_rho': round(rho, 3),
                        'pairwise_pct': round(pct, 1),
                    }
                else:
                    metrics[subset_name + '_vs_DFT'] = {'n_metals': len(subset), 'note': 'too few for correlation'}

        summary[name] = metrics
        print(f"\n--- {name} ranking metrics ---")
        print(json.dumps(metrics, indent=2, default=str))

    out_path = OUT_PATH
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'raw': all_results, 'summary': summary}, f, indent=2, default=str)
    print(f"\n\nFull results written to {out_path}")


if __name__ == '__main__':
    main()
