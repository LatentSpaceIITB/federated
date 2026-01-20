"""
Utility functions for MedPromptFolio.
"""

import os
import json
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from prettytable import PrettyTable


def count_parameters(model: torch.nn.Module, name_filter: str = None) -> int:
    """
    Count trainable parameters in a model.

    Args:
        model: PyTorch model
        name_filter: If provided, only count parameters containing this string

    Returns:
        Number of trainable parameters
    """
    table = PrettyTable(["Module", "Parameters", "Trainable"])
    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        if name_filter and name_filter not in name:
            continue

        params = param.numel()
        trainable = param.requires_grad
        total_params += params
        if trainable:
            trainable_params += params

        table.add_row([name, params, trainable])

    print(table)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return trainable_params


def average_weights(
    weights: List[Dict[str, torch.Tensor]],
    aggregation_weights: List[float] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute weighted average of model weights.

    Args:
        weights: List of state dicts
        aggregation_weights: List of weights (default: uniform)

    Returns:
        Averaged state dict
    """
    if not weights:
        return {}

    if aggregation_weights is None:
        aggregation_weights = [1.0 / len(weights)] * len(weights)

    avg_weights = {}
    for key in weights[0].keys():
        stacked = torch.stack([w[key].float() for w in weights])
        weight_tensor = torch.tensor(aggregation_weights, device=stacked.device)

        # Expand dimensions for broadcasting
        for _ in range(stacked.dim() - 1):
            weight_tensor = weight_tensor.unsqueeze(-1)

        avg_weights[key] = (stacked * weight_tensor).sum(dim=0)

    return avg_weights


def compute_heterogeneity_metrics(
    client_labels: List[np.ndarray],
) -> Dict[str, float]:
    """
    Compute metrics to quantify label distribution heterogeneity.

    Args:
        client_labels: List of label arrays for each client

    Returns:
        Dictionary with heterogeneity metrics
    """
    # Label distribution for each client
    n_clients = len(client_labels)
    n_classes = client_labels[0].shape[1]

    distributions = []
    for labels in client_labels:
        dist = labels.mean(axis=0)  # Prevalence of each class
        distributions.append(dist)

    distributions = np.array(distributions)

    # Compute metrics
    metrics = {}

    # 1. Distribution divergence (average pairwise KL divergence)
    kl_divergences = []
    for i in range(n_clients):
        for j in range(i + 1, n_clients):
            p = distributions[i] + 1e-10
            q = distributions[j] + 1e-10
            kl = np.sum(p * np.log(p / q))
            kl_divergences.append(kl)
    metrics['avg_kl_divergence'] = np.mean(kl_divergences) if kl_divergences else 0.0

    # 2. Distribution variance
    metrics['dist_variance'] = np.var(distributions, axis=0).mean()

    # 3. Class imbalance (Gini coefficient)
    global_dist = np.mean(distributions, axis=0)
    sorted_dist = np.sort(global_dist)
    n = len(sorted_dist)
    index = np.arange(1, n + 1)
    metrics['gini_coefficient'] = (2 * np.sum(index * sorted_dist) / (n * np.sum(sorted_dist))) - ((n + 1) / n)

    return metrics


def compute_optimal_theta(
    snr_global: float,
    snr_local: float,
) -> float:
    """
    Compute theoretically optimal theta based on signal-to-noise ratios.

    From PromptFolio theory:
    theta* = (1 - SNR_local) / (1 - SNR_global + 1 - SNR_local)

    For medical imaging with high heterogeneity:
    - Expected SNR_global ≈ 0.3-0.5 (more noise in aggregated signal)
    - Expected SNR_local ≈ 0.6-0.8 (cleaner local signal)
    - This gives theta* ≈ 0.3-0.4

    Args:
        snr_global: Signal-to-noise ratio of global prompt
        snr_local: Signal-to-noise ratio of local prompt

    Returns:
        Optimal theta value
    """
    numerator = 1 - snr_local
    denominator = (1 - snr_global) + (1 - snr_local)

    if denominator == 0:
        return 0.5  # Default to equal mixing

    theta = numerator / denominator
    return np.clip(theta, 0.0, 1.0)


def estimate_snr_from_data(
    client_embeddings: List[torch.Tensor],
    client_labels: List[torch.Tensor],
) -> Tuple[float, float]:
    """
    Estimate SNR from client embeddings and labels.

    This is a simplified estimation based on class separability.

    Args:
        client_embeddings: List of embedding tensors per client
        client_labels: List of label tensors per client

    Returns:
        Tuple of (estimated_global_snr, estimated_local_snr)
    """
    local_snrs = []

    for embeddings, labels in zip(client_embeddings, client_labels):
        # Compute within-class and between-class variance
        embeddings = embeddings.numpy() if isinstance(embeddings, torch.Tensor) else embeddings
        labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels

        # For multi-label, use first class as proxy
        binary_labels = (labels[:, 0] > 0.5).astype(int)

        if len(np.unique(binary_labels)) < 2:
            continue

        pos_emb = embeddings[binary_labels == 1]
        neg_emb = embeddings[binary_labels == 0]

        if len(pos_emb) == 0 or len(neg_emb) == 0:
            continue

        # Between-class distance
        between = np.linalg.norm(pos_emb.mean(axis=0) - neg_emb.mean(axis=0))

        # Within-class variance
        within = (np.var(pos_emb, axis=0).mean() + np.var(neg_emb, axis=0).mean()) / 2

        if within > 0:
            snr = between / np.sqrt(within)
            local_snrs.append(snr)

    if not local_snrs:
        return 0.5, 0.7  # Default values

    # Local SNR is average of individual client SNRs
    local_snr = np.mean(local_snrs)

    # Global SNR is typically lower due to heterogeneity
    global_snr = local_snr * 0.7  # Heuristic reduction

    # Normalize to [0, 1] range
    local_snr = np.clip(local_snr / 10, 0, 1)
    global_snr = np.clip(global_snr / 10, 0, 1)

    return global_snr, local_snr


def save_results(
    results: Dict,
    output_path: str,
):
    """Save results to JSON file."""
    # Convert numpy types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    results = convert(results)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def load_results(input_path: str) -> Dict:
    """Load results from JSON file."""
    with open(input_path, 'r') as f:
        return json.load(f)


def create_comparison_table(
    results: Dict[str, Dict],
    metrics: List[str] = None,
) -> str:
    """
    Create a formatted comparison table of results.

    Args:
        results: Dictionary mapping method name to metrics dict
        metrics: List of metrics to include (default: all)

    Returns:
        Formatted table string
    """
    if not results:
        return "No results to compare"

    if metrics is None:
        metrics = list(list(results.values())[0].keys())

    table = PrettyTable()
    table.field_names = ["Method"] + metrics

    for method, method_results in results.items():
        row = [method]
        for metric in metrics:
            value = method_results.get(metric, "N/A")
            if isinstance(value, float):
                row.append(f"{value:.4f}")
            else:
                row.append(str(value))
        table.add_row(row)

    return str(table)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device: str = "auto") -> torch.device:
    """Get appropriate device."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)
