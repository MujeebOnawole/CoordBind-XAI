# CoordBind-XAI

**Predicting metal-ligand binding strength and coordination sites from SMILES with explainable AI.**

CoordBind-XAI is a relational graph convolutional network that predicts the first stability
constant (log K1) **and** the donor atoms of a metal-ligand complex from a plain
ligand SMILES string, with per-atom explanations for both tasks.

Dative coordination bonds are encoded as a dedicated `METAL_COORD` edge type, the
metal is connected to the ligand through that edge rather than left as an
isolated fragment, and a per-atom donor classifier sits on the same shared graph
layers as the binding readout.

## Results

Held-out **scaffold split** (entire Murcko scaffolds held out, so every test
complex carries a ligand framework absent from training).

| Metric | Value |
|---|---|
| Test MAE | **0.961** log K units |
| Test R² | **0.872** |
| Pearson r | **0.935** |
| Cross-validated MAE | **1.040 ± 0.030** (3 × 5 folds) |
| Mean ensemble uncertainty | 0.480 log K |
| Scenario A (trustworthy) | 65.6% |
| Metals | **61** |
| Training data | **19,964** experimental stability constants at 25 °C |
| Split | 15,346 train / 2,248 val / 2,370 test |

### Coordination prediction

The same architecture, stripped of the binding head and trained on the 59,739
Cambridge Structural Database ligands of Toney et al. (2025), evaluated on their
held-out 6,616-ligand test set:

| Metric | Toney D-MPNN | This work (t = 0.50) | This work (t = 0.75) |
|---|---|---|---|
| Balanced accuracy | 96.9% | **97.6%** | 97.0% |
| Donor recall | 94.6% | **97.1%** | 95.0% |
| Molecular accuracy | **84.8%** | 74.6% | 81.2% |
| MCC | — | 0.897 | **0.922** |

### Chemistry validation

| Test | Result |
|---|---|
| Irving-Williams series | 87% pairwise accuracy (Mn < Fe < Co < Ni < Cu > Zn) |
| Lanthanide contraction | 80.1% positive trend across 382 shared ligands |
| HSAB (cyanide) | Hg(II) 16.0 ≫ Cu(II) 9.4 ≫ Ca(II) 2.6 |
| Donor dominance | donor attribution 0.948 vs backbone 0.740 |
| Oxidation-state resolution | encoding formal charge cuts Fe MAE by 54% |

### Architecture ablations

No single coordination-specific component drives accuracy; each shift is smaller
than the cross-validation standard deviation. They are retained for the
interpretability and pipeline capability they provide.

| Model | Test MAE | CV MAE | Δ Test |
|---|---|---|---|
| Full model | 0.961 | 1.040 ± 0.030 | — |
| Without triple readout | 0.972 | 1.060 ± 0.026 | +1.1% |
| Without coordination supervision | 0.967 | 1.039 ± 0.027 | +0.6% |
| Without `METAL_COORD` relation | 0.972 | 1.063 ± 0.027 | +1.1% |

## Scope

Reliable as an early-stage ranking and triage step for **mono- and bidentate
chelators** with N, O, S and P donors, binding metals of well-defined oxidation
state, at 25 °C in aqueous solution.

Explicitly out of scope:

- **Hexadentate chelators.** EDTA is underpredicted by 10–17 log K units; the
  model is trained on first stability constants and has never seen cooperative
  chelate stabilisation.
- **Fine rare-earth selectivity.** Adjacent lanthanides differ by 0.1–0.5 log K,
  below the model's resolution.
- **Relativistic heavy metals.** Hg, Au, Pd and Bi coordination is governed by
  scalar-relativistic orbital contraction that 2D SMILES does not encode.
- **Carbon-donor coordination.** Carbenes, carbonyls and cyclopentadienyl rings
  account for 63.6% of coordination false negatives — a representation ceiling
  shared by every 2D-graph model, not an architectural deficit.
- **Non-standard conditions.** No temperature, ionic strength or solvent features.

An ensemble-plus-attribution trust framework labels each prediction Scenario A
through D at inference time so out-of-scope predictions are flagged without
needing ground truth.

## Layout

```
src/binding/      canonical Run 3 pipeline (binding + coordination head)
src/coord_only/   coordination-only model for the Toney benchmark
src/data_prep/    dataset curation: IUPAC, NIST SRD 46, LOGKPREDICT, Toney
data/             curated dataset (19,964 entries) + 99-ligand benchmark
results/canonical    Run 3 metrics, predictions, XAI attributions
results/ablations    T1/T2/T3 architecture ablations
results/diagnostics  calibration, variance decomposition, scaffold overlap
results/coord_only   Toney benchmark evaluation
```

## Pipeline

```
src/binding/build_data.py    SMILES -> PyG graphs, 4 edge types, scaffold split
src/binding/hyper.py         25 Optuna trials, 3-fold internal CV
src/binding/stat_val.py      3 repeats x 5 folds = 15 models
src/binding/final_eval.py    best-per-fold ensemble on the held-out test set
src/binding/xai.py           perturbation XAI + Scenario A/B/C/D framework
src/binding/analyze_xai_chemistry.py   Irving-Williams, HSAB, lanthanide checks
```

The coordination-only model follows the same order under `src/coord_only/`,
ending at `eval_toney.py`.

Scripts resolve `data/` from the repository root automatically. Training was run
on a SLURM cluster; the job scripts are deliberately not included because they
carry site-specific account and path details. Each step is a plain Python entry
point and can be run directly.

## Requirements

Python 3.10+, PyTorch 2.x, PyTorch Geometric 2.x, RDKit 2023+, Optuna 3.x,
scikit-learn, pandas, numpy, scipy, matplotlib. See `requirements.txt`.

## Data provenance

See `data/README.md` for the schema and per-source licensing. The curated set
merges the IUPAC Stability Constants Database, the IUPAC and NIST subsets of
LOGKPREDICT, NIST Standard Reference Database 46, and 176 manually curated
Ga/In/Tl/Be entries. The Toney coordination benchmark is **not** redistributed
here; download it from Zenodo record 13840776.

## Run history

`PROVENANCE.md` records which run produced which numbers. The canonical run is
Run 3. Earlier runs used a smaller 57-metal dataset and are not shipped.

## Citation

> Onawole, A. T.; Sulaiman, K. O.; Alli, Y. A.; Aderinto, S. O.; Anumah, A. O.
> "CoordBind-XAI: Predicting Metal-Ligand Binding Strength and Coordination
> Sites from SMILES." 2026. Preprint.

## License

MIT — see `LICENSE`.
