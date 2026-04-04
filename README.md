# Using Coordination-Aware Neural Networks to Map Metal-Ligand Binding

Predicts metal-ligand stability constants (logK1) from dative-bond SMILES using a relational graph convolutional network with a dedicated METAL_COORD edge type and donor-aware triple readout. Includes perturbation-based explainability with a 4-scenario trust framework, validated against the Irving-Williams series, HSAB theory, and the lanthanide contraction.

**Web app:** https://huggingface.co/spaces/catenate/metal-ligand-binding

## Results

| Metric | Value |
|--------|-------|
| Test MAE | 0.979 logK units |
| Test R2 | 0.870 |
| Pearson r | 0.934 |
| CV MAE | 1.040 +/- 0.021 (3x5 fold) |
| Scenario A (trustworthy) | 65.7% |
| Metals covered | 57 |
| Training data | 19,788 experimental stability constants |

### Chemical Validation

| Test | Result |
|------|--------|
| Irving-Williams series | 87% pairwise accuracy (Mn < Fe < Co < Ni < Cu > Zn) |
| Lanthanide contraction | 80% positive trend (La to Lu) |
| HSAB (cyanide) | Hg (16.0) >> Cu (9.4) >> Ca (2.6) |
| Donor dominance | Donor attr 0.95 > backbone 0.74 |
| Fe2+/Fe3+ discrimination | 7.75 logK difference (ethylenediamine) |

### Zero-Shot Generalization

The model generalizes to metals absent from training. 176 complexes of Ga, In, Tl, and Be were evaluated without retraining:

| Metal | N | MAE | Spearman | Scenario A |
|-------|---|-----|----------|------------|
| Ga | 35 | 7.01 | 0.549 | 51.4% |
| In | 35 | 5.55 | 0.620 | 54.3% |
| Tl | 60 | 3.76 | 0.482 | 35.0% |
| Be | 46 | 2.33 | 0.330 | 50.0% |
| **Overall** | **176** | **4.39** | **0.448** | **46.0%** |

All rankings are statistically significant (p < 1e-4). The model correctly predicts HSAB behavior for Ga3+ (weak cyanide binding, strong NOTA binding) and negligible Be2+-nitrate interaction. Sufficient for screening chelators for radiopharmaceutical metals like Ga-68 without any Ga training data.

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

**19,788** unique metal-ligand complexes spanning **57 metals**, assembled from 4 sources:

- **IUPAC Stability Constants Database** (Karunaratne et al. 2025, JCIM) — 26,805 pre-dedup
- **LOGKPREDICT** (Zahariev et al. 2024) — IUPAC and NIST subsets
- **NIST SRD 46** — SQL dump with ligand names resolved to SMILES via PubChem

Curation: logK outliers removed, duplicates merged (median logK), metals with <5 entries excluded, RDKit validation on all SMILES. Scaffold split by ligand (Murcko decomposition, metal removed): 15,207 train / 2,226 val / 2,355 test with zero scaffold overlap.

**57 metals** across transition metals (3d/4d/5d), all 15 lanthanides, 10 actinides (including Ac, Pu, Am, Cm), and main-group metals (Al, Ga, In, Sn, Pb, Bi, Li, Na, K, Mg, Ca, Sr, Ba). Formal charge is encoded in the SMILES (`[Fe+2]` vs `[Fe+3]`), enabling oxidation-state-dependent predictions.

## Pipeline

```
Step 0: build_data.py    -- Parse SMILES, extract features, scaffold split
Step 1: hyper.py         -- 25 Optuna trials (3-fold internal CV)
Step 2: stat_val.py      -- 3-repeat x 5-fold CV (15 models)
Step 3: final_eval.py    -- Best-per-fold ensemble on held-out test set
Step 4: xai.py           -- Perturbation-based XAI on full dataset
```

Data curation scripts (`process_nist_srd46.py`, `convert_iupac_to_dative.py`, `curate_final_dataset.py`) are included for reproducibility.

### Zero-Shot Prediction on External Data

```bash
python xai.py --top_k 5 --external_csv path/to/new_complexes.csv
```

The CSV must have columns: `smiles` (dative-bond SMILES), `logK1` (experimental, for comparison), `metal_type`. Graphs are built on-the-fly. Any metal in the periodic table is accepted.

## Repository Structure

```
metal-binding-xai/
|-- config.py                  # Metal properties, search space, periodic table detection
|-- build_data.py              # SMILES -> PyG graphs with 4 edge types
|-- model.py                   # RGCNStability with triple readout
|-- loss.py                    # MSE loss
|-- data_module.py             # Data loading, scaffold splitting
|-- hyper.py                   # Optuna hyperparameter optimization
|-- stat_val.py                # 3x5 repeated cross-validation
|-- final_eval.py              # Ensemble evaluation on test set
|-- xai.py                     # Perturbation-based XAI + Scenario A framework
|-- analyze_xai_chemistry.py   # Post-XAI validation (Irving-Williams, HSAB, etc.)
|-- convert_iupac_to_dative.py # Dot SMILES -> dative SMILES conversion
|-- curate_final_dataset.py    # Dataset merging, deduplication, quality control
|-- process_nist_srd46.py      # NIST SRD 46 extraction + PubChem resolution
|-- results/                   # Model outputs, XAI attributions, validation
```

## Requirements

- Python 3.10+
- PyTorch 2.x, PyTorch Geometric 2.x
- RDKit 2023+
- Optuna 3.x, scikit-learn 1.x
- pandas, numpy, matplotlib, scipy

## Citation

> Onawole, A.T. "Using Coordination-Aware Neural Networks to Map Metal-Ligand Binding." (2026). In preparation.

## License

MIT License
