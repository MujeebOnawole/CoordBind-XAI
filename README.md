# Using Coordination-Aware Neural Networks to Map Metal-Ligand Binding

Predicts metal-ligand stability constants (log K1) from dative-bond SMILES using a coordination-aware neural network with a dedicated METAL_COORD edge type and donor-aware triple readout. Includes perturbation-based XAI with Scenario A/B/C/D trust framework and chemical validation against Irving-Williams series, HSAB theory, and lanthanide contraction.

**Web app:** https://huggingface.co/spaces/catenate/metal-ligand-binding

## Results

| Metric | Value |
|--------|-------|
| Test MAE | 0.979 logK units |
| Test R² | 0.870 |
| Pearson r | 0.934 |
| CV MAE | 1.040 ± 0.021 (3x5 fold) |
| Scenario A (trustworthy) | 65.7% |
| Metals covered | 57 |
| Training data | 19,788 experimental stability constants |

### Chemical Validation

| Test | Result |
|------|--------|
| Irving-Williams series | 87% pairwise accuracy (Mn < Fe < Co < Ni < Cu > Zn) |
| Lanthanide contraction | 80% positive trend (La → Lu) |
| HSAB (cyanide) | Hg (16.0) >> Cu (9.4) >> Ca (2.6) — soft donor prefers soft metals |
| Donor dominance | Donor |attr| = 0.95 > backbone 0.74 |
| Fe2+/Fe3+ discrimination | 7.75 logK difference (ethylenediamine) |

### Zero-Shot Generalization (Unseen Metals)

The model generalizes to metals entirely absent from training. Four metals (Ga, In, Tl, Be) were recovered from the NIST SRD 46 after fixing an RDKit valence-checking limitation that had silently excluded them during data curation. Rather than retraining, the existing ensemble was evaluated on these 176 complexes as a zero-shot test.

| Metal | N | MAE | Spearman | Scenario A | logK1 range |
|-------|---|-----|----------|------------|-------------|
| Ga | 35 | 7.01 | 0.549 | 51.4% | 1.4 - 30.7 |
| In | 35 | 5.55 | 0.620 | 54.3% | 2.6 - 25.9 |
| Tl | 60 | 3.76 | 0.482 | 35.0% | -0.8 - 18.9 |
| Be | 46 | 2.33 | 0.330 | 50.0% | -0.9 - 13.2 |
| **Overall** | **176** | **4.39** | **0.448** | **46.0%** | -0.9 - 30.7 |

All Spearman correlations are statistically significant (p < 10^-4). The model correctly predicts:
- **HSAB theory for Ga3+:** weak binding to cyanide (soft donor, predicted 0.98 vs exp 1.36) and strong binding to NOTA (hard N/O chelator, predicted 20.67 vs exp 30.7)
- **No binding for Be2+-nitrate:** predicted -0.23 vs experimental -0.9

The ranking ability is sufficient for screening applications (e.g., ranking chelators for the radiopharmaceutical metal Ga-68) even without any Ga training data. Absolute offsets can be corrected with a simple metal-specific linear calibration from 2-3 known values.

## Architecture

```
Input: Dative-bond SMILES (e.g., NCC1N->[Cu+2]<-N1)
    |
[100-dim Node Features]
    |  Metal: element one-hot + formal charge + ionic radius + period/group
    |  Ligand: atom type + hybridization + aromaticity + metal context + donor type
    |
[Input Projection: Linear -> ReLU -> Dropout]
    |
[RGCNConv x 4 layers, 4 edge types: SINGLE, DOUBLE, TRIPLE, METAL_COORD]
    |  Residual connections + LayerNorm
    |
[Triple Readout]
    |-- Metal pool:  mean over metal atom(s)      [node_role=1]
    |-- Donor pool:  mean over donor atoms         [node_role=2]
    |-- Ligand pool: mean over other ligand atoms  [node_role=0]
    |
[Concatenate -> MLP -> logK1 (unconstrained scalar)]
```

## Dataset

**19,788** unique metal-ligand complexes, **57 metals**, from 4 independent sources:

| Source | Description | Pre-dedup | Post-dedup |
|--------|-------------|-----------|------------|
| JCIM 2025 IUPAC | Karunaratne et al. public release, dot SMILES converted to dative | 26,805 | ~14,000 |
| LOGKPREDICT IUPAC | Zahariev et al. 2024 subset | 37 | 37 |
| LOGKPREDICT NIST | Already in dative SMILES | 1,560 | ~1,560 |
| NIST SRD 46 | SQL dump, ligand names resolved via PubChem | ~8,999 | ~4,191 |

logK1 range: -1.91 to 29.70. Scaffold split: 15,207 train / 2,226 val / 2,355 test (417 unique scaffolds, zero overlap).

### Metal Coverage

57 metals spanning transition metals, lanthanides, actinides, and main-group metals:

**Transition metals (3d):** Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn
**Transition metals (4d):** Y, Zr, Mo, Ru, Rh, Pd, Ag, Cd
**Transition metals (5d):** Hf, W, Pt, Au, Hg
**Lanthanides:** La, Ce, Pr, Nd, Pm, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu
**Actinides:** Ac, Th, U, Np, Pu, Am, Cm, Bk, Cf, Es
**Main-group:** Li, Na, K, Mg, Ca, Sr, Ba, Al, Sn, Pb, Bi

**Charge-state awareness:** The dataset encodes formal charge in the dative SMILES (e.g., `[Fe+2]` vs `[Fe+3]`), enabling the model to learn charge-dependent binding. This is critical — Fe2+ and Fe3+ have a 7.75 logK difference for ethylenediamine.

---

## Data Collection and Curation Pipeline

### Overview

The dataset was assembled from 4 independent sources through a multi-step curation pipeline. Each source required different processing to produce standardized dative-bond SMILES with experimental logK1 values.

```
Source 1: JCIM 2025 IUPAC ──→ convert_iupac_to_dative.py ──→ iupac_logk_dative.csv
Source 2: LOGKPREDICT IUPAC ──→ (already dative SMILES)   ──→ logkpredict_iupac.csv
Source 3: LOGKPREDICT NIST  ──→ (already dative SMILES)   ──→ logkpredict_nist.csv
Source 4: NIST SRD 46       ──→ process_nist_srd46.py     ──→ nist_srd46_dative.csv
                                     │
                                     ├─ Parse SQL dump (liganden.txt, metal.txt, etc.)
                                     ├─ Filter for K-type constants at 25°C
                                     ├─ Resolve ligand names → SMILES via PubChem
                                     └─ Convert dot SMILES → dative SMILES
                                            │
All 4 sources ──→ merge ──→ stability_constants_dative_master.csv
                                            │
                              curate_final_dataset.py
                                            │
                              stability_constants_dative_clean.csv (19,788 entries)
```

### Source 1: JCIM 2025 IUPAC (Karunaratne et al.)

- **Origin:** Public dataset from Karunaratne et al. (2025), JCIM, containing 26,805 metal-ligand stability constants from the IUPAC Stability Constants Database
- **Format:** Dot-separated SMILES (`ligand.[Metal+n]`) with logK1 values
- **Processing:** `convert_iupac_to_dative.py`
  1. Parse dot SMILES into ligand and metal fragments
  2. Identify donor atoms using chemical priority rules:
     - Deprotonated carboxylate O > thiolate S > amine N > pyridine N > carbonyl O > P > thioether S
  3. Insert dative bonds (donor→metal) using RDKit's DATIVE bond type
  4. Validate round-trip SMILES parsing with relaxed sanitization (skips valence check for metals like Ga, In, Tl, Be whose dative bond valences exceed RDKit's default model)

### Source 2-3: LOGKPREDICT (Zahariev et al. 2024)

- **Origin:** Two subsets from the LOGKPREDICT tool:
  - IUPAC subset: 37 entries (after deduplication against Source 1)
  - NIST subset: 1,560 entries already in dative-bond SMILES format
- **Processing:** Minimal — already in target format. Merged directly.

### Source 4: NIST SRD 46 (Critical Stability Constants)

- **Origin:** NIST Standard Reference Database 46, obtained as SQL dump files
- **Raw files:** `liganden.txt` (ligand names), `metal.txt` (metal ions), `beta_definition.txt` (equilibrium equations), `verkn_ligand_metal.txt` (stability constants)
- **Processing:** `process_nist_srd46.py`
  1. Parse tab-delimited SQL dump tables
  2. Filter for K-type (formation constant) records at 25°C
  3. Match beta_definition to identify simple 1:1 K1 constants ([ML]/[M][L])
  4. Resolve ligand names to SMILES via PubChem REST API with NCI CACTUS fallback
  5. Cache resolved SMILES in `pubchem_smiles_cache.json` (rate-limited: 5 req/sec)
  6. Assemble `ligand_SMILES.[Metal+charge]` dot SMILES
  7. Convert to dative SMILES using `convert_dot_to_dative()`

**PubChem resolution statistics:** ~60% of NIST ligand names successfully resolved to SMILES. Failures are typically due to complex IUPAC names with parenthetical substituents, macrocyclic ligands with unusual naming, or deprecated chemical names.

### Final Curation: `curate_final_dataset.py`

Applied to the merged master dataset:

| Step | Filter | Removed | Remaining |
|------|--------|---------|-----------|
| 1 | logK outliers (< -3 or > 30) | ~200 | ~24,000 |
| 2 | Deduplicate (same SMILES → median logK) | ~4,000 | ~20,000 |
| 3 | Remove entries with no metal detected | ~50 | ~19,950 |
| 4 | Remove very small molecules (< 3 atoms) | ~20 | ~19,930 |
| 5 | Remove metals with < 5 entries | ~150 | 19,788 |

**Deduplication strategy:** When the same complex appears multiple times (from different sources or measurements), the **median** logK1 is taken. The `n_measurements` column records how many independent measurements contributed.

### Known Issue: RDKit Valence Rejection (Fixed)

**Problem:** RDKit's default valence model does not include dative bond valences for Ga, In, Tl, and Be. When the dative conversion creates valid SMILES like `O=C(O)CN->[Ga+3]`, the round-trip validation `Chem.MolFromSmiles(smi)` rejects them with "Explicit valence for atom Ga is greater than permitted."

**Impact:** 159 entries lost (44 Ga, 47 In, 18 Tl, 50 Be), including clinically important chelator-radiometal pairs (NOTA-Ga, DOTA-Ga).

**Fix:** Replaced strict `MolFromSmiles()` with relaxed sanitization that skips `SANITIZE_PROPERTIES` (valence check) — the same approach already used when *creating* the dative mol. Applied to `convert_iupac_to_dative.py` and `build_data.py`.

**Recovery:** `extract_recovered_metals.py` converts the 159 lost entries and exports them as a zero-shot test set for evaluating the model on metals it was never trained on.

### Data Splitting

Scaffold split based on **ligand scaffold** (Murcko decomposition with metal removed):
- All complexes sharing the same ligand scaffold are in the same split
- Example: EDTA-Cu, EDTA-Fe, EDTA-Zn are all in the same partition
- This tests generalization to **novel ligand chemotypes**, not just novel metal-ligand combinations

| Split | Complexes | Scaffolds |
|-------|-----------|-----------|
| Training | 15,207 | ~330 |
| Validation | 2,226 | ~40 |
| Test | 2,355 | ~47 |

Zero scaffold overlap between splits.

---

## Repository Structure

```
metal-binding-xai/
|
|-- config.py                     # Constants, metal properties, search space
|-- build_data.py                 # SMILES -> PyG graphs (formal charge from RDKit)
|-- model.py                      # RGCNStability with triple readout
|-- loss.py                       # MSE loss
|-- data_module.py                # Data loading, splitting
|-- hyper.py                      # Optuna 25-trial search
|-- stat_val.py                   # 3x5 CV (15 models)
|-- final_eval.py                 # Best-per-fold ensemble evaluation
|-- xai.py                        # Perturbation-based XAI + Scenario A
|-- analyze_xai_chemistry.py      # Post-XAI chemical validation (5 analyses)
|
|-- data/
|   |-- stability_constants_dative_clean.csv  # Final curated dataset (19,788)
|   |-- nist_srd46_logK_25C.csv              # NIST SRD46 at 25C (14,947)
|   |-- nist_srd46_logK_with_smiles.csv      # NIST with PubChem SMILES (8,999)
|   |-- nist_srd46_dative.csv                # NIST dative converted (7,404)
|   |-- pubchem_smiles_cache.json            # Cached PubChem lookups
|   |-- recovered_metals_zeroshot.csv        # Ga/In/Tl/Be + oxidation states
|   |-- external/
|       |-- NIST_SRD46/                      # Raw SQL dump files
|       |-- LOGKPREDICT/                     # Zahariev et al. data + code
|
|-- results/                      # Model outputs
|   |-- best_hyperparameters.json
|   |-- ensemble_summary.json
|   |-- dataset_stats.json
|   |-- test_predictions.csv       # 2,355 test set predictions
|   |-- cv_fold_results.csv        # 15-fold CV results
|   |-- cv_statistics.json
|   |-- xai_summary.json
|   |-- xai_results_logK1.csv      # Full XAI attributions (19,788 complexes)
|   |-- chemistry_validation_summary.json
```

## Pipeline

The training pipeline runs in 5 sequential steps:

```
Step 0: build_data.py    — Parse SMILES, extract features, scaffold split
Step 1: hyper.py         — 25 Optuna trials (3-fold internal CV)
Step 2: stat_val.py      — 3-repeat x 5-fold CV (15 models)
Step 3: final_eval.py    — Best-per-fold ensemble on held-out test set
Step 4: xai.py           — Perturbation-based XAI on full dataset
```

### Data Curation Pipeline (run before training)

```
Step A: process_nist_srd46.py          — Parse NIST SRD46, resolve SMILES via PubChem
Step B: convert_iupac_to_dative.py     — Convert dot SMILES to dative-bond SMILES
Step C: (manual merge)                 — Combine all 4 sources into master CSV
Step D: curate_final_dataset.py        — Filter, deduplicate, validate → clean CSV
Step E: extract_recovered_metals.py    — Recover Ga/In/Tl/Be for zero-shot testing
```

### Post-XAI Chemical Validation

```bash
python analyze_xai_chemistry.py \
    --results_dir results/ \
    --dataset_csv data/stability_constants_dative_clean.csv
```

Generates 5 figures + JSON summary: Irving-Williams, HSAB heatmap, lanthanide contraction, donor attribution heatmap, metal selectivity by donor type.

## Requirements

- Python 3.10+
- PyTorch 2.x, PyTorch Geometric 2.x
- RDKit 2023+
- Optuna 3.x, scikit-learn 1.x
- pandas, numpy, matplotlib, scipy

## Citation

> Onawole, A.T. "Using Coordination-Aware Neural Networks to Map Metal-Ligand Binding." (2026). In preparation.
