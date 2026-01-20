"""
Virtual Client Splitting for Federated Learning

Splits real datasets into multiple virtual clients to simulate
hospital federation with controllable heterogeneity.

Methods:
1. Dirichlet partitioning - Controls label distribution skew
2. Quantity skew - Controls data amount per client
3. Feature skew simulation - Adds scanner/noise variations
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import torch
from torch.utils.data import Dataset, Subset, DataLoader
import copy


class VirtualClientSplitter:
    """
    Splits a single dataset into multiple virtual clients with heterogeneous
    data distributions using Dirichlet partitioning.

    This simulates different hospitals having different patient populations
    and disease prevalences.
    """

    def __init__(
        self,
        num_clients: int = 4,
        alpha: float = 0.3,
        min_samples_per_client: int = 50,
        seed: int = 42,
    ):
        """
        Args:
            num_clients: Number of virtual clients to create
            alpha: Dirichlet concentration parameter
                   - alpha < 1: Highly heterogeneous (each client gets few classes)
                   - alpha = 1: Uniform random
                   - alpha > 1: More homogeneous (similar to IID)
                   Recommended: 0.1 (extreme), 0.3 (high), 0.5 (moderate), 1.0 (mild)
            min_samples_per_client: Minimum samples each client must have
            seed: Random seed for reproducibility
        """
        self.num_clients = num_clients
        self.alpha = alpha
        self.min_samples_per_client = min_samples_per_client
        self.seed = seed

        np.random.seed(seed)

    def split_by_dirichlet(
        self,
        labels: np.ndarray,
        multi_label: bool = True,
    ) -> Dict[int, List[int]]:
        """
        Split dataset indices using Dirichlet distribution.

        For multi-label data, we use the primary label (argmax) or
        treat each label independently.

        Args:
            labels: Label array of shape (n_samples,) or (n_samples, n_classes)
            multi_label: Whether labels are multi-label

        Returns:
            Dictionary mapping client_id -> list of sample indices
        """
        n_samples = len(labels)

        if multi_label and len(labels.shape) > 1:
            # For multi-label, use primary label (highest confidence) for splitting
            # This ensures samples with similar conditions go to same client
            primary_labels = np.argmax(labels, axis=1)
            n_classes = labels.shape[1]
        else:
            primary_labels = labels.flatten()
            n_classes = len(np.unique(primary_labels))

        # Initialize client indices
        client_indices = {i: [] for i in range(self.num_clients)}

        # For each class, distribute samples to clients via Dirichlet
        for c in range(n_classes):
            # Get indices of samples with this primary label
            class_indices = np.where(primary_labels == c)[0]

            if len(class_indices) == 0:
                continue

            # Sample proportions from Dirichlet distribution
            proportions = np.random.dirichlet([self.alpha] * self.num_clients)

            # Shuffle class indices
            np.random.shuffle(class_indices)

            # Calculate split points
            proportions = proportions / proportions.sum()
            split_points = (proportions.cumsum() * len(class_indices)).astype(int)
            split_points = np.clip(split_points, 0, len(class_indices))

            # Split and assign to clients
            prev_idx = 0
            for client_id in range(self.num_clients):
                end_idx = split_points[client_id]
                client_indices[client_id].extend(class_indices[prev_idx:end_idx].tolist())
                prev_idx = end_idx

        # Ensure minimum samples per client
        self._rebalance_if_needed(client_indices, n_samples)

        # Shuffle each client's indices
        for client_id in client_indices:
            np.random.shuffle(client_indices[client_id])

        return client_indices

    def split_by_quantity(
        self,
        n_samples: int,
        imbalance_factor: float = 2.0,
    ) -> Dict[int, List[int]]:
        """
        Split with quantity skew (different amounts per client).

        Args:
            n_samples: Total number of samples
            imbalance_factor: Ratio of largest to smallest client

        Returns:
            Dictionary mapping client_id -> list of sample indices
        """
        # Generate unequal proportions
        proportions = np.random.dirichlet([self.alpha] * self.num_clients)

        # Scale to ensure imbalance factor
        proportions = proportions / proportions.max()  # Normalize
        proportions = proportions * imbalance_factor / proportions.sum()
        proportions = proportions / proportions.sum()  # Re-normalize

        # Create indices
        all_indices = np.random.permutation(n_samples)
        split_points = (proportions.cumsum() * n_samples).astype(int)

        client_indices = {}
        prev_idx = 0
        for client_id in range(self.num_clients):
            end_idx = split_points[client_id]
            client_indices[client_id] = all_indices[prev_idx:end_idx].tolist()
            prev_idx = end_idx

        return client_indices

    def split_iid(self, n_samples: int) -> Dict[int, List[int]]:
        """
        IID split (uniform random distribution).

        Args:
            n_samples: Total number of samples

        Returns:
            Dictionary mapping client_id -> list of sample indices
        """
        all_indices = np.random.permutation(n_samples)
        splits = np.array_split(all_indices, self.num_clients)

        return {i: splits[i].tolist() for i in range(self.num_clients)}

    def _rebalance_if_needed(
        self,
        client_indices: Dict[int, List[int]],
        n_samples: int,
    ):
        """Ensure each client has minimum samples."""
        # Find clients with too few samples
        deficit_clients = []
        surplus_clients = []

        for client_id, indices in client_indices.items():
            if len(indices) < self.min_samples_per_client:
                deficit_clients.append(client_id)
            elif len(indices) > self.min_samples_per_client * 2:
                surplus_clients.append(client_id)

        # Redistribute from surplus to deficit
        for deficit_client in deficit_clients:
            needed = self.min_samples_per_client - len(client_indices[deficit_client])

            for surplus_client in surplus_clients:
                if needed <= 0:
                    break

                available = len(client_indices[surplus_client]) - self.min_samples_per_client
                transfer = min(needed, available // 2)

                if transfer > 0:
                    # Transfer samples
                    transferred = client_indices[surplus_client][:transfer]
                    client_indices[surplus_client] = client_indices[surplus_client][transfer:]
                    client_indices[deficit_client].extend(transferred)
                    needed -= transfer


class VirtualClientDataset(Dataset):
    """
    Wrapper that creates a virtual client view of a base dataset.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        indices: List[int],
        client_id: int,
        transform_fn: Optional[callable] = None,
    ):
        """
        Args:
            base_dataset: The original full dataset
            indices: Indices belonging to this virtual client
            client_id: ID of this virtual client
            transform_fn: Optional additional transform (e.g., for feature skew)
        """
        self.base_dataset = base_dataset
        self.indices = indices
        self.client_id = client_id
        self.transform_fn = transform_fn

        # Copy class attributes
        if hasattr(base_dataset, 'classnames'):
            self.classnames = base_dataset.classnames
        if hasattr(base_dataset, 'tasks'):
            self.tasks = base_dataset.tasks

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        item = self.base_dataset[real_idx]

        # Apply additional transform if specified
        if self.transform_fn is not None:
            item = self.transform_fn(item, self.client_id)

        return item

    @property
    def labels(self):
        """Get labels for this virtual client's samples."""
        if hasattr(self.base_dataset, 'labels'):
            base_labels = np.array(self.base_dataset.labels)
            return base_labels[self.indices]
        return None


class FederatedVirtualClients:
    """
    Manager for creating and handling virtual clients from multiple real datasets.

    Example:
        # Create from 2 real datasets, 4 virtual clients each = 8 total clients
        fed_clients = FederatedVirtualClients(
            datasets={'nih': nih_dataset, 'chexpert': chexpert_dataset},
            clients_per_dataset=4,
            alpha=0.3,
        )

        # Get data loader for virtual client 5 (2nd client from chexpert)
        loader = fed_clients.get_client_loader(5, train=True)
    """

    def __init__(
        self,
        datasets: Dict[str, Dataset],
        clients_per_dataset: int = 4,
        alpha: float = 0.3,
        batch_size: int = 32,
        num_workers: int = 4,
        seed: int = 42,
    ):
        """
        Args:
            datasets: Dictionary mapping dataset name -> Dataset object
            clients_per_dataset: Number of virtual clients per real dataset
            alpha: Dirichlet concentration (lower = more heterogeneous)
            batch_size: Batch size for data loaders
            num_workers: Number of data loading workers
            seed: Random seed
        """
        self.datasets = datasets
        self.clients_per_dataset = clients_per_dataset
        self.alpha = alpha
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.dataset_names = list(datasets.keys())
        self.num_real_datasets = len(datasets)
        self.num_clients = self.num_real_datasets * clients_per_dataset

        # Create splitter
        self.splitter = VirtualClientSplitter(
            num_clients=clients_per_dataset,
            alpha=alpha,
            seed=seed,
        )

        # Split each dataset into virtual clients
        self.virtual_clients = {}  # client_id -> VirtualClientDataset
        self.client_to_dataset = {}  # client_id -> dataset_name
        self.client_splits = {}  # dataset_name -> {local_client_id -> indices}

        self._create_virtual_clients()

        # Create data loaders
        self.train_loaders = {}
        self.test_loaders = {}
        self._create_data_loaders()

        # Store class names
        first_dataset = list(datasets.values())[0]
        self.classnames = getattr(first_dataset, 'classnames', None)

    def _create_virtual_clients(self):
        """Split each real dataset into virtual clients."""
        global_client_id = 0

        for dataset_name, dataset in self.datasets.items():
            # Get labels for splitting
            if hasattr(dataset, 'labels'):
                labels = np.array(dataset.labels)
            elif hasattr(dataset, 'get_labels'):
                labels = dataset.get_labels()
            else:
                # Fallback: uniform split
                labels = np.zeros(len(dataset))

            # Split using Dirichlet
            multi_label = len(labels.shape) > 1
            client_indices = self.splitter.split_by_dirichlet(labels, multi_label)

            self.client_splits[dataset_name] = client_indices

            # Create virtual client datasets
            for local_client_id, indices in client_indices.items():
                self.virtual_clients[global_client_id] = VirtualClientDataset(
                    base_dataset=dataset,
                    indices=indices,
                    client_id=global_client_id,
                )
                self.client_to_dataset[global_client_id] = dataset_name
                global_client_id += 1

        print(f"Created {self.num_clients} virtual clients from {self.num_real_datasets} datasets")
        self._print_client_stats()

    def _create_data_loaders(self):
        """Create data loaders for all virtual clients."""
        for client_id, dataset in self.virtual_clients.items():
            self.train_loaders[client_id] = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=len(dataset) > self.batch_size,
            )
            # For test, we use the same data (in practice, you'd have separate test sets)
            self.test_loaders[client_id] = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )

    def _print_client_stats(self):
        """Print statistics about virtual client distributions."""
        print("\nVirtual Client Statistics:")
        print("-" * 60)

        for client_id in range(self.num_clients):
            dataset_name = self.client_to_dataset[client_id]
            n_samples = len(self.virtual_clients[client_id])

            # Get label distribution
            labels = self.virtual_clients[client_id].labels
            if labels is not None and len(labels.shape) > 1:
                label_dist = labels.mean(axis=0)
                dist_str = ", ".join([f"{v:.2f}" for v in label_dist])
            else:
                dist_str = "N/A"

            print(f"  Client {client_id:2d} ({dataset_name:12s}): {n_samples:5d} samples, dist=[{dist_str}]")

        print("-" * 60)

    def get_client_loader(self, client_id: int, train: bool = True) -> DataLoader:
        """Get data loader for a specific virtual client."""
        if train:
            return self.train_loaders[client_id]
        return self.test_loaders[client_id]

    def get_client_dataset(self, client_id: int) -> VirtualClientDataset:
        """Get dataset for a specific virtual client."""
        return self.virtual_clients[client_id]

    def get_client_name(self, client_id: int) -> str:
        """Get descriptive name for a virtual client."""
        dataset_name = self.client_to_dataset[client_id]
        local_id = client_id % self.clients_per_dataset
        return f"{dataset_name}_client_{local_id}"

    def get_num_samples(self, client_id: int) -> int:
        """Get number of samples for a client."""
        return len(self.virtual_clients[client_id])

    def get_heterogeneity_stats(self) -> Dict:
        """
        Compute heterogeneity statistics across virtual clients.

        Returns metrics useful for understanding/reporting data distribution.
        """
        stats = {
            'num_clients': self.num_clients,
            'alpha': self.alpha,
            'samples_per_client': [],
            'label_distributions': [],
        }

        for client_id in range(self.num_clients):
            n_samples = len(self.virtual_clients[client_id])
            stats['samples_per_client'].append(n_samples)

            labels = self.virtual_clients[client_id].labels
            if labels is not None and len(labels.shape) > 1:
                stats['label_distributions'].append(labels.mean(axis=0).tolist())

        # Compute summary stats
        samples = np.array(stats['samples_per_client'])
        stats['samples_mean'] = float(samples.mean())
        stats['samples_std'] = float(samples.std())
        stats['samples_min'] = int(samples.min())
        stats['samples_max'] = int(samples.max())

        # Compute distribution divergence if available
        if stats['label_distributions']:
            dists = np.array(stats['label_distributions'])
            # Average pairwise JS divergence
            from scipy.spatial.distance import jensenshannon
            js_divs = []
            for i in range(len(dists)):
                for j in range(i + 1, len(dists)):
                    js_divs.append(jensenshannon(dists[i] + 1e-10, dists[j] + 1e-10))
            stats['avg_js_divergence'] = float(np.mean(js_divs)) if js_divs else 0.0

        return stats


def create_federated_virtual_clients(
    dataset_configs: Dict[str, Dict],
    clients_per_dataset: int = 4,
    alpha: float = 0.3,
    batch_size: int = 32,
    seed: int = 42,
) -> FederatedVirtualClients:
    """
    Convenience function to create federated virtual clients from config.

    Args:
        dataset_configs: Dict with structure:
            {
                'nih': {'root': '/path/to/nih', 'train': True, ...},
                'chexpert': {'root': '/path/to/chexpert', 'train': True, ...},
            }
        clients_per_dataset: Number of virtual clients per real dataset
        alpha: Dirichlet concentration
        batch_size: Batch size
        seed: Random seed

    Returns:
        FederatedVirtualClients manager
    """
    from .chest_xray_datasets import get_dataset

    datasets = {}
    for name, config in dataset_configs.items():
        datasets[name] = get_dataset(
            dataset_name=name,
            root=config['root'],
            train=config.get('train', True),
            max_samples=config.get('max_samples', None),
        )

    return FederatedVirtualClients(
        datasets=datasets,
        clients_per_dataset=clients_per_dataset,
        alpha=alpha,
        batch_size=batch_size,
        seed=seed,
    )


# Feature skew simulation transforms
class ScannerNoiseTransform:
    """
    Simulates different scanner characteristics per client.
    Adds client-specific noise and intensity variations.
    """

    def __init__(self, noise_levels: Dict[int, float] = None, seed: int = 42):
        """
        Args:
            noise_levels: Dict mapping client_id -> noise std dev
            seed: Random seed
        """
        self.noise_levels = noise_levels or {}
        self.rng = np.random.RandomState(seed)

    def __call__(self, item: Dict, client_id: int) -> Dict:
        img = item['img']

        # Get noise level for this client
        noise_std = self.noise_levels.get(client_id, 0.0)

        if noise_std > 0:
            noise = torch.randn_like(img) * noise_std
            img = img + noise
            img = torch.clamp(img, 0, 1)

        item['img'] = img
        return item


class IntensityShiftTransform:
    """
    Simulates different intensity calibrations across scanners.
    """

    def __init__(self, shift_ranges: Dict[int, Tuple[float, float]] = None, seed: int = 42):
        """
        Args:
            shift_ranges: Dict mapping client_id -> (brightness_shift, contrast_scale)
        """
        self.shift_ranges = shift_ranges or {}
        self.rng = np.random.RandomState(seed)

    def __call__(self, item: Dict, client_id: int) -> Dict:
        img = item['img']

        if client_id in self.shift_ranges:
            brightness, contrast = self.shift_ranges[client_id]
            img = img * contrast + brightness
            img = torch.clamp(img, 0, 1)

        item['img'] = img
        return item
