#!/usr/bin/env python
"""
Main training script for MedPromptFolio.

Usage:
    # Train with synthetic data (for testing)
    python scripts/train.py --method medpromptfolio --use_synthetic

    # Train with real data
    python scripts/train.py --method medpromptfolio \
        --chexpert_root /path/to/chexpert \
        --mimic_root /path/to/mimic \
        --nih_root /path/to/nih \
        --vindr_root /path/to/vindr

    # Run all baselines
    python scripts/train.py --method all --use_synthetic
"""

import argparse
import os
import sys

# Add parent directory to path for package imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)
sys.path.insert(0, parent_dir)

from medpromptfolio.federated_trainer import run_experiment, FederatedTrainer
from medpromptfolio.med_prompt_folio import MedPromptFolio, FedAvgMedCLIP, FedProxMedCLIP, LocalOnlyMedCLIP
from medpromptfolio.chest_xray_datasets import FederatedChestXrayDataset
from medpromptfolio.constants import CHEXPERT_COMPETITION_TASKS


def parse_args():
    parser = argparse.ArgumentParser(description="MedPromptFolio Training")

    # Method selection
    parser.add_argument(
        "--method",
        type=str,
        default="medpromptfolio",
        choices=["medpromptfolio", "fedavg", "fedprox", "local", "all"],
        help="Training method"
    )

    # Model configuration
    parser.add_argument("--n_ctx", type=int, default=8, help="Number of context tokens")
    parser.add_argument("--theta", type=float, default=0.3, help="Portfolio mixing coefficient")
    parser.add_argument(
        "--medclip_checkpoint",
        type=str,
        default="/home/pinak/MedClip/proj1/pretrained/medclip-vit",
        help="Path to MedCLIP checkpoint"
    )

    # Federated learning configuration
    parser.add_argument("--num_rounds", type=int, default=50, help="Number of FL rounds")
    parser.add_argument("--local_epochs", type=int, default=1, help="Local epochs per round")
    parser.add_argument("--client_fraction", type=float, default=1.0, help="Fraction of clients per round")
    parser.add_argument("--fedprox_mu", type=float, default=0.01, help="FedProx regularization")

    # Training configuration
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Data configuration
    parser.add_argument("--use_synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--chexpert_root", type=str, default=None, help="CheXpert data root")
    parser.add_argument("--mimic_root", type=str, default=None, help="MIMIC-CXR data root")
    parser.add_argument("--nih_root", type=str, default=None, help="NIH ChestX-ray14 data root")
    parser.add_argument("--vindr_root", type=str, default=None, help="VinDr-CXR data root")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples per dataset")

    # Output configuration
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    return parser.parse_args()


def get_data_roots(args):
    """Build data roots dictionary from arguments."""
    data_roots = {}

    if args.chexpert_root:
        data_roots["chexpert"] = args.chexpert_root
    if args.mimic_root:
        data_roots["mimic_cxr"] = args.mimic_root
    if args.nih_root:
        data_roots["nih_chestxray"] = args.nih_root
    if args.vindr_root:
        data_roots["vindr_cxr"] = args.vindr_root

    return data_roots if data_roots else None


def run_single_method(args, method: str):
    """Run experiment for a single method."""
    output_dir = os.path.join(args.output_dir, method)
    os.makedirs(output_dir, exist_ok=True)

    data_roots = get_data_roots(args)

    return run_experiment(
        method=method,
        medclip_checkpoint=args.medclip_checkpoint,
        data_roots=data_roots,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        lr=args.lr,
        theta=args.theta,
        n_ctx=args.n_ctx,
        batch_size=args.batch_size,
        client_fraction=args.client_fraction,
        fedprox_mu=args.fedprox_mu,
        output_dir=output_dir,
        use_synthetic=args.use_synthetic or (data_roots is None),
        seed=args.seed,
        device=args.device,
    )


def main():
    args = parse_args()

    print("=" * 60)
    print("MedPromptFolio Training")
    print("=" * 60)
    print(f"Method: {args.method}")
    print(f"MedCLIP checkpoint: {args.medclip_checkpoint}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Use synthetic: {args.use_synthetic}")
    print("=" * 60)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    if args.method == "all":
        # Run all methods
        methods = ["medpromptfolio", "fedavg", "fedprox", "local"]
        results = {}

        for method in methods:
            print(f"\n{'#' * 60}")
            print(f"Running {method}")
            print(f"{'#' * 60}")

            history = run_single_method(args, method)
            results[method] = {
                "best_auc": max(history["test_auc"]),
                "final_auc": history["test_auc"][-1],
            }

        # Print comparison
        print("\n" + "=" * 60)
        print("Results Comparison")
        print("=" * 60)
        print(f"{'Method':<20} {'Best AUC':<12} {'Final AUC':<12}")
        print("-" * 44)
        for method, res in results.items():
            print(f"{method:<20} {res['best_auc']:<12.4f} {res['final_auc']:<12.4f}")
        print("=" * 60)

    else:
        # Run single method
        run_single_method(args, args.method)


if __name__ == "__main__":
    main()
