"""
Federated Learning Trainer for MedPromptFolio

Orchestrates the federated learning process across multiple hospital clients.
"""

import os
import json
import copy
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .med_prompt_folio import (
    MedPromptFolio,
    MedPromptFolioTrainer,
    FedAvgMedCLIP,
    FedProxMedCLIP,
    LocalOnlyMedCLIP,
)
from .chest_xray_datasets import FederatedChestXrayDataset, SyntheticChestXrayDataset
from .constants import CHEXPERT_COMPETITION_TASKS, FL_DEFAULT_CONFIG


class FederatedTrainer:
    """
    Federated learning trainer for medical prompt learning.

    Implements the standard FL training loop:
    1. Server broadcasts global model to selected clients
    2. Clients perform local training
    3. Server aggregates client updates
    4. Repeat for multiple rounds
    """

    def __init__(
        self,
        model: MedPromptFolio,
        federated_dataset: FederatedChestXrayDataset = None,
        num_clients: int = 4,
        num_rounds: int = 50,
        local_epochs: int = 1,
        client_fraction: float = 1.0,
        lr: float = 0.002,
        batch_size: int = 32,
        device: str = "cuda",
        output_dir: str = "./output",
        use_synthetic: bool = False,
        fedprox_mu: float = -1,  # -1 means don't use FedProx
        seed: int = 42,
    ):
        """
        Args:
            model: MedPromptFolio model
            federated_dataset: Federated dataset manager
            num_clients: Number of clients
            num_rounds: Number of communication rounds
            local_epochs: Number of local epochs per round
            client_fraction: Fraction of clients participating per round
            lr: Learning rate
            batch_size: Batch size
            device: Device
            output_dir: Directory for saving outputs
            use_synthetic: Whether to use synthetic data for testing
            fedprox_mu: FedProx regularization (if > 0)
            seed: Random seed
        """
        self.model = model
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.client_fraction = client_fraction
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.output_dir = output_dir
        self.fedprox_mu = fedprox_mu
        self.seed = seed

        np.random.seed(seed)
        torch.manual_seed(seed)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Client data statistics (initialize before dataset setup)
        self.client_data_sizes = {}

        # Training history
        self.history = {
            'rounds': [],
            'train_loss': [],
            'train_auc': [],
            'test_auc': [],
            'per_client_auc': [],
            'time': [],
        }

        # Setup datasets
        self.federated_dataset = None
        if federated_dataset is not None:
            self.federated_dataset = federated_dataset
            self.num_clients = federated_dataset.num_clients
        elif use_synthetic:
            self._setup_synthetic_data()

        # Initialize trainer
        self.trainer = MedPromptFolioTrainer(
            model=model,
            lr=lr,
            device=device,
        )

    def _setup_synthetic_data(self):
        """Setup synthetic data for testing."""
        print("Setting up synthetic data for testing...")

        # Create synthetic datasets with different label distributions
        # to simulate heterogeneity
        self.train_loaders = {}
        self.test_loaders = {}

        for client_id in range(self.num_clients):
            # Create heterogeneous label distributions
            np.random.seed(self.seed + client_id)
            label_dist = {
                'Atelectasis': 0.2 + 0.2 * np.random.random(),
                'Cardiomegaly': 0.15 + 0.2 * np.random.random(),
                'Consolidation': 0.1 + 0.15 * np.random.random(),
                'Edema': 0.1 + 0.2 * np.random.random(),
                'Pleural Effusion': 0.2 + 0.2 * np.random.random(),
            }

            train_dataset = SyntheticChestXrayDataset(
                num_samples=500,
                tasks=CHEXPERT_COMPETITION_TASKS,
                label_dist=label_dist,
                seed=self.seed + client_id,
            )
            test_dataset = SyntheticChestXrayDataset(
                num_samples=100,
                tasks=CHEXPERT_COMPETITION_TASKS,
                label_dist=label_dist,
                seed=self.seed + client_id + 1000,
            )

            self.train_loaders[client_id] = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
            )
            self.test_loaders[client_id] = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
            )

            self.client_data_sizes[client_id] = len(train_dataset)

        print(f"Created {self.num_clients} synthetic clients")

    def get_train_loader(self, client_id: int) -> DataLoader:
        """Get training data loader for client."""
        if self.federated_dataset is not None:
            return self.federated_dataset.get_client_loader(client_id, train=True)
        return self.train_loaders.get(client_id)

    def get_test_loader(self, client_id: int) -> DataLoader:
        """Get test data loader for client."""
        if self.federated_dataset is not None:
            return self.federated_dataset.get_client_loader(client_id, train=False)
        return self.test_loaders.get(client_id)

    def get_client_name(self, client_id: int) -> str:
        """Get client name/hospital."""
        if self.federated_dataset is not None:
            return self.federated_dataset.get_client_name(client_id)
        return f"synthetic_client_{client_id}"

    def select_clients(self) -> List[int]:
        """Select clients for current round."""
        num_selected = max(1, int(self.client_fraction * self.num_clients))
        selected = np.random.choice(
            range(self.num_clients),
            num_selected,
            replace=False
        )
        return selected.tolist()

    def get_aggregation_weights(self, client_ids: List[int]) -> List[float]:
        """
        Get aggregation weights based on data size.

        Args:
            client_ids: List of participating clients

        Returns:
            List of weights (sum to 1)
        """
        if not self.client_data_sizes:
            # Equal weights if data sizes unknown
            return [1.0 / len(client_ids)] * len(client_ids)

        total = sum(self.client_data_sizes.get(cid, 1) for cid in client_ids)
        weights = [self.client_data_sizes.get(cid, 1) / total for cid in client_ids]
        return weights

    def train_round(self, round_num: int, client_ids: List[int]) -> Dict:
        """
        Execute one federated round.

        Args:
            round_num: Current round number
            client_ids: List of participating client IDs

        Returns:
            Dictionary with round metrics
        """
        round_metrics = {
            'round': round_num,
            'clients': client_ids,
            'train_loss': [],
            'train_auc': [],
        }

        # Store global weights for FedProx
        global_weights = self.model.get_prompt_state_dict() if self.fedprox_mu > 0 else None

        # Local training on each client
        for client_id in client_ids:
            print(f"  Training on client {client_id} ({self.get_client_name(client_id)})...")

            # Download model to client
            self.model.fed_download_model(client_id)

            # Get client's data
            train_loader = self.get_train_loader(client_id)

            if train_loader is None:
                print(f"  Warning: No training data for client {client_id}")
                continue

            # Update client data size
            if client_id not in self.client_data_sizes:
                self.client_data_sizes[client_id] = len(train_loader.dataset)

            # Local training
            for epoch in range(self.local_epochs):
                metrics = self.trainer.train_epoch(
                    train_loader,
                    global_weights=global_weights,
                    fedprox=(self.fedprox_mu > 0),
                    mu=self.fedprox_mu,
                )

            round_metrics['train_loss'].append(metrics['loss'])
            round_metrics['train_auc'].append(metrics['auc'])

            # Upload model from client
            self.model.fed_upload_model(client_id)

        # Aggregate client updates
        weights = self.get_aggregation_weights(client_ids)
        self.model.fed_aggregate_model(client_ids, weights)

        # Average metrics
        round_metrics['avg_train_loss'] = np.mean(round_metrics['train_loss'])
        round_metrics['avg_train_auc'] = np.mean(round_metrics['train_auc'])

        return round_metrics

    def evaluate_round(self, round_num: int, client_ids: List[int] = None) -> Dict:
        """
        Evaluate model on all clients.

        Args:
            round_num: Current round number
            client_ids: Clients to evaluate (default: all)

        Returns:
            Dictionary with evaluation metrics
        """
        if client_ids is None:
            client_ids = list(range(self.num_clients))

        eval_metrics = {
            'round': round_num,
            'clients': client_ids,
            'test_loss': [],
            'test_auc': [],
            'per_class_auc': defaultdict(list),
        }

        for client_id in client_ids:
            # Download model to client
            self.model.fed_download_model(client_id)

            # Evaluate
            test_loader = self.get_test_loader(client_id)
            if test_loader is None:
                continue

            metrics = self.trainer.evaluate(test_loader)

            eval_metrics['test_loss'].append(metrics['loss'])
            eval_metrics['test_auc'].append(metrics['auc'])

            for classname, auc in metrics['per_class_auc'].items():
                eval_metrics['per_class_auc'][classname].append(auc)

        # Average metrics
        eval_metrics['avg_test_loss'] = np.mean(eval_metrics['test_loss'])
        eval_metrics['avg_test_auc'] = np.mean(eval_metrics['test_auc'])
        eval_metrics['avg_per_class_auc'] = {
            k: np.mean(v) for k, v in eval_metrics['per_class_auc'].items()
        }

        return eval_metrics

    def train(self, verbose: bool = True) -> Dict:
        """
        Run full federated training.

        Args:
            verbose: Whether to print progress

        Returns:
            Training history
        """
        print(f"\n{'='*60}")
        print(f"Starting MedPromptFolio Federated Training")
        print(f"{'='*60}")
        print(f"Num clients: {self.num_clients}")
        print(f"Num rounds: {self.num_rounds}")
        print(f"Local epochs: {self.local_epochs}")
        print(f"Client fraction: {self.client_fraction}")
        print(f"Learning rate: {self.lr}")
        print(f"Theta (mixing coef): {self.model.theta}")
        print(f"{'='*60}\n")

        start_time = time.time()

        # Initialize model
        global_weights = self.model.get_prompt_state_dict()
        self.model.fed_init_model(global_weights)

        # Initial evaluation
        print("Initial evaluation...")
        eval_metrics = self.evaluate_round(0, list(range(self.num_clients)))
        print(f"  Initial Avg AUC: {eval_metrics['avg_test_auc']:.4f}")

        self.history['rounds'].append(0)
        self.history['test_auc'].append(eval_metrics['avg_test_auc'])
        self.history['per_client_auc'].append(eval_metrics['test_auc'])
        self.history['train_loss'].append(0)
        self.history['train_auc'].append(0)
        self.history['time'].append(0)

        best_auc = eval_metrics['avg_test_auc']
        best_round = 0

        # Training loop
        for round_num in range(1, self.num_rounds + 1):
            round_start = time.time()

            if verbose:
                print(f"\n--- Round {round_num}/{self.num_rounds} ---")

            # Select clients
            client_ids = self.select_clients()
            if verbose:
                client_names = [self.get_client_name(cid) for cid in client_ids]
                print(f"Selected clients: {client_names}")

            # Train
            train_metrics = self.train_round(round_num, client_ids)

            # Evaluate
            eval_metrics = self.evaluate_round(round_num, list(range(self.num_clients)))

            # Update history
            self.history['rounds'].append(round_num)
            self.history['train_loss'].append(train_metrics['avg_train_loss'])
            self.history['train_auc'].append(train_metrics['avg_train_auc'])
            self.history['test_auc'].append(eval_metrics['avg_test_auc'])
            self.history['per_client_auc'].append(eval_metrics['test_auc'])
            self.history['time'].append(time.time() - start_time)

            # Track best
            if eval_metrics['avg_test_auc'] > best_auc:
                best_auc = eval_metrics['avg_test_auc']
                best_round = round_num
                self._save_checkpoint('best_model.pt')

            # Print progress
            if verbose:
                round_time = time.time() - round_start
                print(f"  Train Loss: {train_metrics['avg_train_loss']:.4f}")
                print(f"  Train AUC:  {train_metrics['avg_train_auc']:.4f}")
                print(f"  Test AUC:   {eval_metrics['avg_test_auc']:.4f}")
                print(f"  Best AUC:   {best_auc:.4f} (round {best_round})")
                print(f"  Per-class AUC: {eval_metrics['avg_per_class_auc']}")

            # Save periodic checkpoint
            if round_num % 10 == 0:
                self._save_checkpoint(f'model_round_{round_num}.pt')

        # Final summary
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
        print(f"Best AUC: {best_auc:.4f} at round {best_round}")
        print(f"Final AUC: {self.history['test_auc'][-1]:.4f}")
        print(f"Total time: {self.history['time'][-1]:.1f}s")
        print(f"{'='*60}")

        # Save final results
        self._save_results()
        self._save_checkpoint('final_model.pt')

        return self.history

    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = os.path.join(self.output_dir, filename)
        torch.save({
            'model_state_dict': self.model.get_prompt_state_dict(),
            'theta': self.model.theta,
            'history': self.history,
        }, path)

    def _save_results(self):
        """Save training results to JSON."""
        # Convert numpy types for JSON serialization
        results = {}
        for key, value in self.history.items():
            if isinstance(value, list):
                results[key] = [
                    float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for v in value
                ]
            else:
                results[key] = value

        path = os.path.join(self.output_dir, 'results.json')
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)


def run_experiment(
    method: str = "medpromptfolio",
    medclip_checkpoint: str = None,
    data_roots: Dict[str, str] = None,
    num_rounds: int = 50,
    local_epochs: int = 1,
    lr: float = 0.002,
    theta: float = 0.3,
    n_ctx: int = 8,
    batch_size: int = 32,
    client_fraction: float = 1.0,
    fedprox_mu: float = -1,
    output_dir: str = "./output",
    use_synthetic: bool = True,
    seed: int = 42,
    device: str = "cuda",
) -> Dict:
    """
    Run a federated learning experiment.

    Args:
        method: Algorithm to use ('medpromptfolio', 'fedavg', 'fedprox', 'local')
        medclip_checkpoint: Path to MedCLIP weights
        data_roots: Dictionary mapping dataset names to paths
        num_rounds: Number of FL rounds
        local_epochs: Local epochs per round
        lr: Learning rate
        theta: Portfolio mixing coefficient
        n_ctx: Number of context tokens
        batch_size: Batch size
        client_fraction: Fraction of clients per round
        fedprox_mu: FedProx regularization (if using FedProx)
        output_dir: Output directory
        use_synthetic: Use synthetic data
        seed: Random seed
        device: Device

    Returns:
        Training history
    """
    print(f"\n{'#'*60}")
    print(f"Running experiment: {method}")
    print(f"{'#'*60}")

    # Create model based on method
    if method == "medpromptfolio":
        model = MedPromptFolio(
            classnames=CHEXPERT_COMPETITION_TASKS,
            medclip_checkpoint=medclip_checkpoint,
            n_ctx=n_ctx,
            theta=theta,
            device=device,
        )
    elif method == "fedavg":
        model = FedAvgMedCLIP(
            classnames=CHEXPERT_COMPETITION_TASKS,
            medclip_checkpoint=medclip_checkpoint,
            n_ctx=n_ctx,
            device=device,
        )
    elif method == "fedprox":
        model = FedProxMedCLIP(
            classnames=CHEXPERT_COMPETITION_TASKS,
            medclip_checkpoint=medclip_checkpoint,
            n_ctx=n_ctx,
            mu=fedprox_mu,
            device=device,
        )
    elif method == "local":
        model = LocalOnlyMedCLIP(
            classnames=CHEXPERT_COMPETITION_TASKS,
            medclip_checkpoint=medclip_checkpoint,
            n_ctx=n_ctx,
            device=device,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Setup federated dataset
    federated_dataset = None
    if data_roots and not use_synthetic:
        federated_dataset = FederatedChestXrayDataset(
            data_roots=data_roots,
            tasks=CHEXPERT_COMPETITION_TASKS,
            batch_size=batch_size,
        )

    # Create trainer
    trainer = FederatedTrainer(
        model=model,
        federated_dataset=federated_dataset,
        num_clients=len(data_roots) if data_roots else 4,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        client_fraction=client_fraction,
        lr=lr,
        batch_size=batch_size,
        device=device,
        output_dir=output_dir,
        use_synthetic=use_synthetic,
        fedprox_mu=fedprox_mu if method == "fedprox" else -1,
        seed=seed,
    )

    # Run training
    history = trainer.train(verbose=True)

    return history
