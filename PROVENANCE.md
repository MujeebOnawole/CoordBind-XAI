# Provenance

Which run produced which numbers, and why several intermediate runs exist.

## Canonical run

**Run 3** is the canonical pipeline. Every number in `results/canonical/`, in the
README, and in the manuscript comes from it.

| Property | Value |
|---|---|
| Dataset | 19,964 entries, 61 metals |
| Split | 15,346 train / 2,248 val / 2,370 test (scaffold) |
| Node features | 120-dimensional |
| Hyperparameters | hidden 256, 4 conv layers, dropout 0.054, lr 1.87e-3, weight decay 2.76e-5, batch 32, coord_weight 1.25 |
| Test | MAE 0.961, R² 0.872, Pearson r 0.935 |
| Cross-validation | 1.040 ± 0.030 across 15 folds |

## Superseded runs (not shipped)

These are recorded because they explain numbers that would otherwise look
inconsistent between the repository history and the manuscript.

| Run | Test n | Test MAE | What it was |
|---|---|---|---|
| Early run | 2,355 | 0.979 | 57 metals, 100-dimensional node features. Produced the repository's April 2026 contents and the original parity figure. |
| Charge-undefined | 2,355 | 1.053 | Formal charge hardcoded to zero. Source of the "hardcoded charge" column in the manuscript's charge-encoding table. |
| **Run 3** | **2,370** | **0.961** | **Canonical.** 61 metals, 120-dimensional features. |
| Run 4 | 2,370 | 0.966 | Joint training with the Toney coordination corpus merged into the pool. Reported in the Supporting Information only. |

### Why the test partition changed from 2,355 to 2,370

Gallium, indium, thallium and beryllium were added after the early run, taking
the dataset from 57 to 61 metals and 19,788 to 19,964 entries. The scaffold split
was recomputed, moving the test partition from 2,355 to 2,370 complexes.

Two consequences worth knowing:

1. The formal-charge comparison (1.053 → 0.961) spans this change in test-set
   composition as well as the change in charge encoding. The improvement
   attributable to charge encoding alone is bounded by those figures rather than
   given exactly by them. The manuscript footnotes this.
2. Ga, In, Tl and Be were originally evaluated as a zero-shot generalisation set
   against a 57-metal model. They are now inside the training data, so that
   result is a **historical** finding about the architecture's inductive biases,
   not a property of the shipped model. Do not read it as a zero-shot benchmark
   of the current release.

## Ablations

`results/ablations/` holds three from-scratch retrains on the Run 3 split.

| Ablation | Test MAE | CV MAE |
|---|---|---|
| T1, triple readout replaced by global mean pool | 0.972 | 1.060 ± 0.026 |
| T2, coordination supervision removed (`coord_weight = 0`) | 0.967 | 1.039 ± 0.027 |
| T3, `METAL_COORD` collapsed to `SINGLE` at graph construction | 0.972 | 1.063 ± 0.027 |

**Caveat:** the ablation runs reuse the hyperparameters of the tuning round that
preceded the final Optuna search, not the Run 3 hyperparameters. Each row
therefore reflects the ablated component *and* that hyperparameter difference.
Since every shift is well inside the cross-validation standard deviation, the
conclusion — no single component drives accuracy — is unaffected.

## Diagnostics

`results/diagnostics/` holds three analyses.

- **L1, calibration.** Area under the calibration error curve 0.233. Empirical
  coverage falls below nominal at every confidence level, so ensemble spread
  alone is overconfident. Computed on the earlier 2,355-entry partition.
- **L2, variance decomposition.** Donor-element signature explains 37.5% of
  training-label variance against 4.4% for metal identity, a ratio of 8.6. This
  is model-independent support for the donor-dominance attribution result, which
  matters because the node features already carry donor-type inputs.
- **L3, scaffold overlap.** Computed on the earlier partition. Recomputed on the
  Run 3 split: 415 unique scaffolds, 413 confined entirely to training. The two
  shared groups are the acyclic chelators and cyclohexane, the large groups
  deliberately split 80:10:10 internally to avoid extreme imbalance.

## Coordination evaluation

Two different evaluations exist and report different ligand counts.

- **Coordination-only model**, `results/coord_only/`: the full 6,616-ligand Toney
  test set, zero parse failures. 97.6% balanced accuracy at threshold 0.50.
- **Probing the joint binding model**, `results/canonical/coord_probing_summary.json`:
  6,336 ligands, because 280 of the 6,616 fail automated dative-bond
  construction. Reported per metal; the manuscript quotes the copper figures.

6,616 − 280 = 6,336. Both numbers are correct and refer to different experiments.
