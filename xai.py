"""
xai.py - Explainable AI for logK1 Stability Constant Prediction

Same perturbation-based substructure masking (occlusion) as tmGNN-XAI,
adapted for single-property logK1 prediction.

SCENARIO A DEFINITION FOR logK1 REGRESSION:
============================================
- Consensus: Low ensemble CV (< 10%) -- models agree on value
- XAI Agreement (DUAL CRITERIA - either satisfies):
  * Criterion 1: >= 70% of ALL node attributions have expected direction
  * Criterion 2: Mean signed attribution has expected direction

Direction logic:
  * If predicted logK1 > dataset mean (6.76): expect POSITIVE attributions
    (masking node DECREASES logK1 -> node contributes to stronger binding)
  * If predicted logK1 < dataset mean (6.76): expect NEGATIVE attributions
    (masking node INCREASES logK1 -> node contributes to weaker binding)

COMPUTE EFFICIENCY:
  - Use ALL models for ensemble PREDICTION (consensus)
  - Use ONLY best model for XAI ATTRIBUTION (save compute)

Usage:
    python xai.py --top_k 5
    python xai.py --top_k 5 --full_dataset
"""

import os
import json
import argparse
import glob
import re
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Batch
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
import logging

from config import get_config, LOGK1_STATS
from model import RGCNStability
from data_module import load_dataset, get_data_splits
from build_data import construct_tmc_graph

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# SCENARIO A THRESHOLDS
# =============================================================================

CV_THRESHOLD = 0.10       # 10% relative std = high confidence
AGREEMENT_THRESHOLD = 0.70  # 70% agreement required (Criterion 1)
TOP_K_NODES = 5            # Top-k nodes to report


# =============================================================================
# ENSEMBLE LOADING
# =============================================================================

def load_ensemble_models(checkpoints_dir, config, hp, device, node_feature_dim, top_k=5):
    """
    Load ensemble models using best-per-fold selection.

    Returns:
        models: List of models (all for prediction)
        best_model: Single best model (for XAI attribution)
    """
    ckpt_files = glob.glob(os.path.join(checkpoints_dir, '*.pt'))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoints in {checkpoints_dir}")

    checkpoints = []
    for f in ckpt_files:
        ckpt = torch.load(f, map_location='cpu')
        mae = ckpt.get('val_mae', float('inf'))
        basename = os.path.basename(f)
        fold = None
        if '_fold' in basename:
            try:
                fold = int(basename.split('_fold')[1].split('_')[0])
            except (ValueError, IndexError):
                pass
        checkpoints.append((f, mae, ckpt, fold))

    # Best-per-fold
    fold_best = {}
    for f, mae, ckpt, fold in checkpoints:
        if fold is not None:
            if fold not in fold_best or mae < fold_best[fold][1]:
                fold_best[fold] = (f, mae, ckpt, fold)

    if len(fold_best) >= top_k:
        selected = sorted(fold_best.values(), key=lambda x: x[1])[:top_k]
    else:
        checkpoints.sort(key=lambda x: x[1])
        selected = checkpoints[:top_k]

    top_checkpoints = [(f, mae, ckpt) for f, mae, ckpt, _ in selected]

    logger.info(f"Loading ensemble ({len(top_checkpoints)} models):")
    for f, mae, _ in top_checkpoints:
        logger.info(f"  {os.path.basename(f)}: MAE = {mae:.4f}")

    models = []
    for _, _, ckpt in top_checkpoints:
        model = RGCNStability(
            node_feature_dim=node_feature_dim,
            hidden_dim=hp.get('hidden_dim', 128),
            num_conv_layers=hp.get('num_conv_layers', 3),
            num_edge_types=config.num_edge_types,
            dropout=0.0,  # No dropout during inference
        )
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)
        model.eval()
        models.append(model)

    best_model = models[0]
    logger.info(f"Best model for XAI: {os.path.basename(top_checkpoints[0][0])}")

    return models, best_model


# =============================================================================
# SUBSTRUCTURE MASKING (OCCLUSION)
# =============================================================================

def substructure_masking(model, data, device='cpu'):
    """
    Compute node attributions using occlusion for logK1.

    SIGNED attribution: base_pred - masked_pred
      Positive: masking DECREASES logK1 (node contributes to stronger binding)
      Negative: masking INCREASES logK1 (node contributes to weaker binding)

    Args:
        model: Single RGCN model (best model)
        data: PyG Data object (single graph)
        device: torch device

    Returns:
        signed_attributions: np.array (n_nodes,)
        magnitude_attributions: np.array (n_nodes,)
        base_pred: float
    """
    model.eval()
    n_nodes = data.x.shape[0]
    batch = Batch.from_data_list([data]).to(device)

    with torch.no_grad():
        output = model(batch)
        base_pred = output.item() if output.numel() == 1 else output[0, 0].item()

    signed_attributions = np.zeros(n_nodes)

    for i in range(n_nodes):
        masked_data = data.clone()
        masked_x = data.x.clone()
        masked_x[i] = 0
        masked_data.x = masked_x

        masked_batch = Batch.from_data_list([masked_data]).to(device)

        with torch.no_grad():
            output = model(masked_batch)
            masked_pred = output.item() if output.numel() == 1 else output[0, 0].item()

        signed_attributions[i] = base_pred - masked_pred

    magnitude_attributions = np.abs(signed_attributions)
    return signed_attributions, magnitude_attributions, base_pred


# =============================================================================
# ENSEMBLE PREDICTION
# =============================================================================

def get_ensemble_prediction(models, data, device):
    """
    Get ensemble prediction for logK1.

    Returns:
        mean_pred, std_pred, cv
    """
    predictions = []

    for model in models:
        model.eval()
        batch = Batch.from_data_list([data]).to(device)

        with torch.no_grad():
            output = model(batch)
            pred = output.item() if output.numel() == 1 else output[0, 0].item()
            predictions.append(pred)

    mean_pred = np.mean(predictions)
    std_pred = np.std(predictions)
    cv = std_pred / abs(mean_pred) if abs(mean_pred) > 1e-8 else std_pred

    return mean_pred, std_pred, cv


# =============================================================================
# SCENARIO A CLASSIFICATION
# =============================================================================

def check_xai_agreement(signed_attrs, ensemble_mean_pred, property_mean):
    """
    Check XAI agreement for logK1 using DUAL CRITERIA.

    Direction logic:
      If predicted logK1 > property_mean (6.76): expect POSITIVE attributions
      If predicted logK1 < property_mean (6.76): expect NEGATIVE attributions

    DUAL CRITERIA (EITHER satisfies):
      Criterion 1: >= 70% of ALL node attributions have expected direction
      Criterion 2: Mean signed attribution has expected direction

    Returns:
        agreement_ratio, mean_attr, is_agreed, criterion_used
    """
    n_nodes = len(signed_attrs)
    expected_positive = ensemble_mean_pred > property_mean

    # Criterion 1: ratio of matching nodes
    if expected_positive:
        matching_count = np.sum(signed_attrs > 0)
    else:
        matching_count = np.sum(signed_attrs < 0)

    agreement_ratio = matching_count / n_nodes if n_nodes > 0 else 0.0
    criterion1_satisfied = agreement_ratio >= AGREEMENT_THRESHOLD

    # Criterion 2: mean attribution direction
    mean_attr = np.mean(signed_attrs)
    criterion2_satisfied = (mean_attr > 0) if expected_positive else (mean_attr < 0)

    is_agreed = criterion1_satisfied or criterion2_satisfied

    if criterion1_satisfied and criterion2_satisfied:
        criterion_used = 'both'
    elif criterion1_satisfied:
        criterion_used = 'ratio'
    elif criterion2_satisfied:
        criterion_used = 'mean'
    else:
        criterion_used = 'none'

    return agreement_ratio, mean_attr, is_agreed, criterion_used


def classify_scenario(cv, is_agreed):
    """
    Classify into Scenario A/B/C/D.

    A: HIGH consensus + HIGH agreement (Trustworthy)
    B: HIGH consensus + LOW agreement (Overconfident)
    C: LOW consensus + HIGH agreement (Underconfident)
    D: LOW consensus + LOW agreement (Unreliable)
    """
    high_consensus = cv < CV_THRESHOLD

    if high_consensus and is_agreed:
        return 'A', 'Trustworthy (HIGH consensus + HIGH XAI agreement)'
    elif high_consensus and not is_agreed:
        return 'B', 'Overconfident (HIGH consensus + LOW XAI agreement)'
    elif not high_consensus and is_agreed:
        return 'C', 'Underconfident (LOW consensus + HIGH XAI agreement)'
    else:
        return 'D', 'Unreliable (LOW consensus + LOW XAI agreement)'


# =============================================================================
# XAI ANALYZER
# =============================================================================

class XAIAnalyzer:
    """XAI Analyzer for logK1 stability constant prediction."""

    def __init__(self, models, best_model, config, device='cpu'):
        self.models = models
        self.best_model = best_model
        self.config = config
        self.device = device
        self.property_mean = config.property_mean  # 6.76

    def analyze_graph(self, data):
        """Analyze a single graph."""
        # Step 1: Ensemble prediction
        mean_pred, std_pred, cv = get_ensemble_prediction(
            self.models, data, self.device
        )

        # Step 2: XAI attribution (best model only)
        signed_attrs, magnitude_attrs, base_pred = substructure_masking(
            self.best_model, data, self.device
        )

        # Step 3: Agreement check
        agreement_ratio, mean_attr, is_agreed, criterion_used = check_xai_agreement(
            signed_attrs, mean_pred, self.property_mean
        )

        # Step 4: Classify scenario
        scenario, scenario_desc = classify_scenario(cv, is_agreed)

        # Top-k nodes
        top_k_indices = np.argsort(magnitude_attrs)[-TOP_K_NODES:][::-1]

        return {
            'prediction': float(mean_pred),
            'uncertainty': float(std_pred),
            'cv': float(cv),
            'agreement_ratio': float(agreement_ratio),
            'mean_attr': float(mean_attr),
            'is_agreed': is_agreed,
            'criterion_used': criterion_used,
            'scenario': scenario,
            'scenario_description': scenario_desc,
            'n_nodes': len(signed_attrs),
            'top_k_node_indices': top_k_indices.tolist(),
            'top_k_signed_attrs': signed_attrs[top_k_indices].tolist(),
            'top_k_magnitudes': magnitude_attrs[top_k_indices].tolist(),
            'all_signed_attrs': signed_attrs.tolist(),
        }

    def analyze_batch(self, graphs, meta_df=None, output_dir=None):
        """Analyze multiple graphs."""
        logger.info(f"\n{'='*60}")
        logger.info("logK1 SUBSTRUCTURE MASKING XAI ANALYSIS")
        logger.info(f"{'='*60}")
        logger.info(f"Samples: {len(graphs)}")
        logger.info(f"Ensemble: {len(self.models)} models")
        logger.info(f"Property mean (direction threshold): {self.property_mean:.2f}")
        logger.info(f"CV threshold: {CV_THRESHOLD*100:.0f}%")
        logger.info(f"Agreement threshold: {AGREEMENT_THRESHOLD*100:.0f}%")

        results = []
        scenario_counts = defaultdict(int)
        errors = []

        for i, data in enumerate(tqdm(graphs, desc="XAI Analysis")):
            try:
                result = self.analyze_graph(data)

                # Add ID
                if meta_df is not None and i < len(meta_df):
                    result['idx'] = i
                    if 'smiles' in meta_df.columns:
                        result['smiles'] = meta_df.iloc[i]['smiles']
                    if 'metal_type' in meta_df.columns:
                        result['metal_type'] = meta_df.iloc[i]['metal_type']
                    if 'logK1' in meta_df.columns:
                        result['logK1_true'] = float(meta_df.iloc[i]['logK1'])
                else:
                    result['idx'] = i

                scenario_counts[result['scenario']] += 1
                results.append(result)

            except Exception as e:
                errors.append((i, str(e)))

        logger.info(f"\nSuccessfully analyzed: {len(results)}")
        if errors:
            logger.info(f"Errors: {len(errors)}")
            for idx, err in errors[:5]:
                logger.info(f"  Graph {idx}: {err}")

        # Summary
        self._print_summary(results, scenario_counts)

        # Save
        if output_dir:
            self._save_results(results, scenario_counts, output_dir)

        return results

    def _print_summary(self, results, scenario_counts):
        """Print analysis summary."""
        total = sum(scenario_counts.values())
        logger.info(f"\n{'='*60}")
        logger.info("SCENARIO DISTRIBUTION (logK1)")
        logger.info(f"{'='*60}")

        descriptions = {
            'A': 'Trustworthy',
            'B': 'Overconfident',
            'C': 'Underconfident',
            'D': 'Unreliable',
        }

        for s in ['A', 'B', 'C', 'D']:
            count = scenario_counts[s]
            pct = count / total * 100 if total > 0 else 0
            logger.info(f"  {s}: {count:4d} ({pct:5.1f}%) - {descriptions[s]}")

        # Agreement by metal type
        if results and 'metal_type' in results[0]:
            logger.info(f"\nScenario A rate by metal type:")
            metal_results = defaultdict(list)
            for r in results:
                metal_results[r['metal_type']].append(r['scenario'] == 'A')

            for metal in sorted(metal_results.keys()):
                vals = metal_results[metal]
                rate = sum(vals) / len(vals) * 100
                logger.info(f"  {metal}: {rate:.1f}% Scenario A (n={len(vals)})")

    def _save_results(self, results, scenario_counts, output_dir):
        """Save results to files."""
        os.makedirs(output_dir, exist_ok=True)

        # CSV with main results
        rows = []
        for r in results:
            row = {
                'idx': r.get('idx', ''),
                'smiles': r.get('smiles', ''),
                'metal_type': r.get('metal_type', ''),
                'logK1_true': r.get('logK1_true', float('nan')),
                'logK1_pred': r['prediction'],
                'uncertainty': r['uncertainty'],
                'cv': r['cv'],
                'agreement_ratio': r['agreement_ratio'],
                'mean_attr': r['mean_attr'],
                'is_agreed': r['is_agreed'],
                'criterion_used': r['criterion_used'],
                'scenario': r['scenario'],
                'n_nodes': r['n_nodes'],
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(output_dir, 'xai_results_logK1.csv'), index=False)

        # Detailed attributions (JSON)
        attr_data = [{
            'idx': r.get('idx', ''),
            'smiles': r.get('smiles', ''),
            'metal_type': r.get('metal_type', ''),
            'scenario': r['scenario'],
            'prediction': r['prediction'],
            'agreement_ratio': r['agreement_ratio'],
            'criterion_used': r['criterion_used'],
            'top_k_node_indices': r['top_k_node_indices'],
            'top_k_signed_attrs': r['top_k_signed_attrs'],
            'all_signed_attrs': r['all_signed_attrs'],
        } for r in results]

        with open(os.path.join(output_dir, 'xai_attributions_logK1.json'), 'w') as f:
            json.dump(attr_data, f, indent=2)

        # Summary
        total = sum(scenario_counts.values())
        summary = {
            'n_samples': len(results),
            'property_mean': self.property_mean,
            'cv_threshold': CV_THRESHOLD,
            'agreement_threshold': AGREEMENT_THRESHOLD,
            'dual_criteria': {
                'criterion_1': '>=70% of ALL node attributions have expected direction',
                'criterion_2': 'Mean signed attribution has expected direction',
                'logic': 'EITHER criterion satisfies agreement',
            },
            'scenario_distribution': dict(scenario_counts),
            'scenario_A_pct': scenario_counts['A'] / total * 100 if total > 0 else 0,
        }
        with open(os.path.join(output_dir, 'xai_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\nResults saved to: {output_dir}")


# =============================================================================
# MAIN
# =============================================================================

def run_xai_analysis(config, top_k=5, full_dataset=False):
    """Run XAI analysis."""

    graphs_path = os.path.join(config.output_dir, f"{config.task_name}_graphs.pt")
    meta_path = os.path.join(config.output_dir, f"{config.task_name}_meta.csv")
    checkpoints_dir = os.path.join(config.output_dir, 'cv_results', 'checkpoints')

    graphs, meta_df = load_dataset(graphs_path, meta_path)
    train_graphs, val_graphs, test_graphs = get_data_splits(graphs, meta_df)

    node_feature_dim = graphs[0].x.shape[1]
    logger.info(f"Node feature dimension: {node_feature_dim}")

    if full_dataset:
        analysis_graphs = graphs
        analysis_meta = meta_df.reset_index(drop=True)
        logger.info(f"FULL DATASET: {len(analysis_graphs)} graphs")
    else:
        analysis_graphs = test_graphs if len(test_graphs) > 0 else graphs[:500]
        test_mask = meta_df['group'] == 'test'
        analysis_meta = meta_df[test_mask].reset_index(drop=True) if len(test_graphs) > 0 else meta_df.iloc[:500]
        logger.info(f"TEST SET: {len(analysis_graphs)} graphs")

    hp_path = os.path.join(config.output_dir, 'best_hyperparameters.json')
    hp = json.load(open(hp_path)) if os.path.exists(hp_path) else {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models, best_model = load_ensemble_models(
        checkpoints_dir, config, hp, device, node_feature_dim, top_k
    )

    analyzer = XAIAnalyzer(models, best_model, config, device)

    xai_dir = os.path.join(config.output_dir, 'xai_results')
    results = analyzer.analyze_batch(analysis_graphs, analysis_meta, output_dir=xai_dir)

    return results


def run_xai_external(config, external_csv, top_k=5):
    """Run XAI analysis on an external CSV (zero-shot prediction).

    The CSV must have columns: smiles, logK1, metal_type
    Graphs are built on-the-fly from SMILES using construct_tmc_graph().
    """
    logger.info(f"Loading external CSV: {external_csv}")
    ext_df = pd.read_csv(external_csv)
    logger.info(f"Entries: {len(ext_df)}")
    logger.info(f"Metals: {ext_df['metal_type'].value_counts().to_dict()}")

    # Build graphs on-the-fly
    graphs = []
    valid_rows = []
    failed = 0
    for idx, row in tqdm(ext_df.iterrows(), total=len(ext_df), desc="Building graphs"):
        try:
            graph = construct_tmc_graph(row['smiles'], row.get('metal_type'))
            graph.y = torch.tensor([row['logK1']], dtype=torch.float32)
            graphs.append(graph)
            valid_rows.append(idx)
        except Exception as e:
            failed += 1
            logger.warning(f"Row {idx} failed: {e}")

    logger.info(f"Built {len(graphs)} graphs ({failed} failed)")
    meta_df = ext_df.loc[valid_rows].reset_index(drop=True)

    # Load ensemble
    checkpoints_dir = os.path.join(config.output_dir, 'cv_results', 'checkpoints')
    hp_path = os.path.join(config.output_dir, 'best_hyperparameters.json')
    hp = json.load(open(hp_path)) if os.path.exists(hp_path) else {}
    node_feature_dim = graphs[0].x.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models, best_model = load_ensemble_models(
        checkpoints_dir, config, hp, device, node_feature_dim, top_k
    )

    analyzer = XAIAnalyzer(models, best_model, config, device)

    # Output to a separate directory
    csv_stem = os.path.splitext(os.path.basename(external_csv))[0]
    xai_dir = os.path.join(config.output_dir, f'xai_zeroshot_{csv_stem}')
    results = analyzer.analyze_batch(graphs, meta_df, output_dir=xai_dir)

    logger.info(f"Results saved to: {xai_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description='XAI for logK1 stability constants')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--full_dataset', action='store_true', default=False)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--external_csv', type=str, default=None,
                        help='Path to external CSV for zero-shot XAI prediction. '
                             'CSV must have columns: smiles, logK1, metal_type')
    args = parser.parse_args()

    config = get_config()
    if args.output_dir:
        config.output_dir = args.output_dir

    logger.info("=" * 60)
    logger.info("logK1 Stability Constant - XAI Analysis")
    logger.info("=" * 60)
    logger.info(f"Ensemble size: {args.top_k}")
    logger.info(f"Start: {datetime.now()}")

    if args.external_csv:
        logger.info(f"ZERO-SHOT MODE: {args.external_csv}")
        run_xai_external(config, args.external_csv, args.top_k)
    else:
        logger.info(f"Full dataset: {args.full_dataset}")
        run_xai_analysis(config, args.top_k, args.full_dataset)

    logger.info(f"End: {datetime.now()}")


if __name__ == '__main__':
    main()
