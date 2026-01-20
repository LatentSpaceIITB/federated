#!/usr/bin/env python
"""
Full MedPromptFolio Experiment with Baseline Comparisons

Compares:
1. MedPromptFolio (our method)
2. FedAvg
3. FedProx
4. Local-only training

Datasets: NIH ChestX-ray14, CheXpert, RSNA Pneumonia
"""

import sys
sys.path.insert(0, '/home/pinak/MedClip/fedtpg')

import os
import json
import time
import warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import numpy as np
from datetime import datetime

from medpromptfolio import (
    MedPromptFolio,
    FedAvgMedCLIP,
    FedProxMedCLIP,
    LocalOnlyMedCLIP,
    FederatedTrainer,
    FederatedChestXrayDataset,
    CHEXPERT_COMPETITION_TASKS,
    create_heterogeneity_report,
)

# Experiment configuration
CONFIG = {
    'data_roots': {
        'nih_chestxray': '/home/pinak/MedClip/fedtpg/data/nih_chestxray',
        'chexpert': '/home/pinak/MedClip/fedtpg/data/chexpert',
        'rsna_pneumonia': '/home/pinak/MedClip/fedtpg/data/rsna_pneumonia',
    },
    'medclip_path': '/home/pinak/MedClip/proj1/pretrained/medclip-vit',

    # Federated settings
    'use_virtual_clients': True,
    'virtual_clients_per_dataset': 3,  # 3 datasets x 3 = 9 clients
    'dirichlet_alpha': 0.3,  # High heterogeneity
    'max_samples_per_client': 2000,  # Use more data

    # Training settings
    'num_rounds': 20,
    'local_epochs': 2,
    'batch_size': 32,
    'lr': 0.002,
    'n_ctx': 4,

    # MedPromptFolio specific
    'theta': 0.3,

    # FedProx specific
    'fedprox_mu': 0.01,

    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

METHODS = ['medpromptfolio', 'fedavg', 'fedprox', 'local']


def create_model(method, config):
    """Create model based on method."""
    if method == 'medpromptfolio':
        return MedPromptFolio(
            classnames=CHEXPERT_COMPETITION_TASKS,
            medclip_checkpoint=config['medclip_path'],
            n_ctx=config['n_ctx'],
            theta=config['theta'],
            device=config['device'],
        )
    elif method == 'fedavg':
        return FedAvgMedCLIP(
            classnames=CHEXPERT_COMPETITION_TASKS,
            medclip_checkpoint=config['medclip_path'],
            n_ctx=config['n_ctx'],
            device=config['device'],
        )
    elif method == 'fedprox':
        return FedProxMedCLIP(
            classnames=CHEXPERT_COMPETITION_TASKS,
            medclip_checkpoint=config['medclip_path'],
            n_ctx=config['n_ctx'],
            mu=config['fedprox_mu'],
            device=config['device'],
        )
    elif method == 'local':
        return LocalOnlyMedCLIP(
            classnames=CHEXPERT_COMPETITION_TASKS,
            medclip_checkpoint=config['medclip_path'],
            n_ctx=config['n_ctx'],
            device=config['device'],
        )
    else:
        raise ValueError(f"Unknown method: {method}")


def run_experiment(method, fed_dataset, config, output_dir):
    """Run single experiment."""
    print(f"\n{'#'*70}")
    print(f"# Running: {method.upper()}")
    print(f"{'#'*70}\n")

    # Create fresh model
    model = create_model(method, config)

    # Create trainer
    trainer = FederatedTrainer(
        model=model,
        federated_dataset=fed_dataset,
        num_clients=fed_dataset.num_clients,
        num_rounds=config['num_rounds'],
        local_epochs=config['local_epochs'],
        lr=config['lr'],
        batch_size=config['batch_size'],
        device=config['device'],
        output_dir=os.path.join(output_dir, method),
        seed=config['seed'],
    )

    # Train
    start_time = time.time()
    results = trainer.train()
    elapsed = time.time() - start_time

    # Add metadata
    results['method'] = method
    results['elapsed_time'] = elapsed
    results['config'] = {k: str(v) for k, v in config.items()}

    # Save results
    results_path = os.path.join(output_dir, method, 'final_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results


def main():
    print("="*70)
    print("MedPromptFolio Full Experiment")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {CONFIG['device']}")
    print(f"Methods: {METHODS}")
    print("="*70)

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'/home/pinak/MedClip/fedtpg/experiments/full_exp_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump({k: str(v) for k, v in CONFIG.items()}, f, indent=2)

    # Create federated dataset (shared across methods)
    print("\n" + "="*70)
    print("Loading Datasets")
    print("="*70)

    fed_dataset = FederatedChestXrayDataset(
        data_roots=CONFIG['data_roots'],
        use_virtual_clients=CONFIG['use_virtual_clients'],
        virtual_clients_per_dataset=CONFIG['virtual_clients_per_dataset'],
        dirichlet_alpha=CONFIG['dirichlet_alpha'],
        max_samples_per_client=CONFIG['max_samples_per_client'],
        batch_size=CONFIG['batch_size'],
        num_workers=4,
    )

    fed_dataset.print_client_summary()

    # Generate heterogeneity report
    print("\nGenerating heterogeneity report...")
    try:
        create_heterogeneity_report(fed_dataset, output_dir, prefix='data_heterogeneity')
    except Exception as e:
        print(f"Warning: Could not generate heterogeneity report: {e}")

    # Run experiments
    all_results = {}

    for method in METHODS:
        try:
            results = run_experiment(method, fed_dataset, CONFIG, output_dir)
            all_results[method] = {
                'best_auc': results.get('best_global_auc', results.get('best_auc', 0)),
                'final_auc': results.get('final_global_auc', results.get('final_auc', 0)),
                'best_round': results.get('best_round', 0),
                'elapsed_time': results.get('elapsed_time', 0),
            }
        except Exception as e:
            print(f"Error running {method}: {e}")
            import traceback
            traceback.print_exc()
            all_results[method] = {'error': str(e)}

    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"{'Method':<20} {'Best AUC':<12} {'Final AUC':<12} {'Best Round':<12} {'Time (s)':<10}")
    print("-"*70)

    for method, results in all_results.items():
        if 'error' in results:
            print(f"{method:<20} ERROR: {results['error'][:40]}")
        else:
            print(f"{method:<20} {results['best_auc']:<12.4f} {results['final_auc']:<12.4f} {results['best_round']:<12} {results['elapsed_time']:<10.1f}")

    print("="*70)

    # Save summary
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
