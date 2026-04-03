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
| Fe²⁺/Fe³⁺ discrimination | 7.75 logK difference (ethylenediamine) |

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

| Source | Description | Contribution |
|--------|-------------|-------------|
| JCIM 2025 IUPAC | Karunaratne et al. public release, dot SMILES converted to dative | 26,805 (pre-dedup) |
| LOGKPREDICT IUPAC | Zahariev et al. 2024 subset | 37 (after dedup) |
| LOGKPREDICT NIST | Already in dative SMILES | 1,560 |
| NIST SRD 46 | SQL dump, ligand names resolved via PubChem | ~2,000 |

logK1 range: -1.91 to 29.70. Scaffold split: 15,207 train / 2,226 val / 2,355 test (417 unique scaffolds, zero overlap).

## Repository Structure

```
TMC_binding/
|
|-- data/
|   |-- stability_constants_dative_clean.csv   # Final dataset (19,788 rows)
|
|-- stability_gnn/                    # Model training and evaluation
|   |-- config.py                     # Constants, search space, Configuration
|   |-- build_data.py                 # SMILES -> PyG graphs (formal charge from RDKit)
|   |-- model.py                      # RGCNStability with triple readout
|   |-- loss.py                       # MSE loss
|   |-- data_module.py                # Data loading, splitting
|   |-- hyper.py                      # Optuna 25-trial search
|   |-- stat_val.py                   # 3x5 CV (15 models)
|   |-- final_eval.py                 # Best-per-fold ensemble evaluation
|   |-- xai.py                        # Perturbation-based XAI + Scenario A
|   |-- analyze_xai_chemistry.py      # Post-XAI chemical validation (5 analyses)
|   |-- slurm/                        # HPC job scripts
|
|-- hf_space/                         # Gradio web app (deployed to HuggingFace)
|   |-- app.py                        # Main UI (Predict, XAI, Metal Comparison, Batch, About)
|   |-- model.py                      # Inference-only RGCNStability
|   |-- graph_builder.py              # SMILES -> PyG graph with node_roles
|   |-- config.py                     # 1 property, 32 metals, 18 popular ligands
|   |-- substructure_xai.py           # XAI for single property
|   |-- visualize.py                  # 2D molecular drawing
|   |-- models/                       # 5 checkpoints + manifest
|
|-- manuscript/                       # Paper draft + SI
|   |-- stability_constant_gnn_xai_draft.docx   # Main manuscript (8 figures)
|   |-- supporting_information.docx              # SI (9 tables)
|   |-- figures/                                 # All generated figures
|
|-- Results/output/                   # Run 2 results (final)
|-- Results_charge_undefnd/output/    # Run 1 results (for comparison)
|
|-- methods.md                        # Methods diary (640 lines)
|-- project_status.md                 # Current status and results
|-- plan.txt                          # Execution plan
|-- demo_guide_JKMRC.txt              # Mining industry demo guide
```

## Running the Pipeline

### Local (data curation)

```bash
python convert_iupac_to_dative.py
python process_nist_srd46.py
python curate_final_dataset.py
```

### HPC (model training)

```bash
cd /scratch/.../TMC_BE/slurm
bash run_all.sh 0    # full pipeline: build_data -> hyper -> CV -> eval -> XAI
```

### Post-XAI Analysis (local)

```bash
python stability_gnn/analyze_xai_chemistry.py \
    --results_dir Results/output/xai_results \
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
