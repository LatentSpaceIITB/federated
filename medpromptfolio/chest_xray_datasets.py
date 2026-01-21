"""
Chest X-ray Datasets for Federated Learning

Implements dataset loaders for:
- CheXpert (Stanford)
- MIMIC-CXR (Beth Israel)
- NIH ChestX-ray14 (NIH)
- VinDr-CXR (Vietnam)

Each dataset represents a different "hospital" in the federated setting.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Optional, Callable, Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms

from .constants import (
    IMG_SIZE,
    IMG_MEAN,
    IMG_STD,
    CHEXPERT_COMPETITION_TASKS,
    DATASET_CONFIGS,
)


def get_transforms(train: bool = True, img_size: int = IMG_SIZE):
    """
    Get image transforms for chest X-ray images.

    Args:
        train: Whether to use training augmentations
        img_size: Target image size

    Returns:
        torchvision transform
    """
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[IMG_MEAN], std=[IMG_STD]),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[IMG_MEAN], std=[IMG_STD]),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        ])


class BaseChestXrayDataset(Dataset):
    """
    Base class for chest X-ray datasets.

    Handles common functionality:
    - Multi-label classification
    - Uncertainty label handling
    - Image loading and transforms
    """

    def __init__(
        self,
        root: str,
        tasks: List[str] = None,
        train: bool = True,
        transform: Callable = None,
        uncertainty_policy: str = "ones",
        max_samples: int = None,
    ):
        """
        Args:
            root: Root directory of dataset
            tasks: List of pathologies to predict (default: CheXpert competition tasks)
            train: Whether to load training set
            transform: Image transforms
            uncertainty_policy: How to handle uncertain labels ('zeros', 'ones', 'ignore')
            max_samples: Maximum number of samples (for debugging)
        """
        self.root = root
        self.tasks = tasks or CHEXPERT_COMPETITION_TASKS
        self.train = train
        self.transform = transform or get_transforms(train)
        self.uncertainty_policy = uncertainty_policy
        self.max_samples = max_samples

        # To be filled by subclasses
        self.image_paths = []
        self.labels = []
        self.classnames = self.tasks

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get a single sample.

        Returns:
            dict with 'img', 'label', 'path'
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        try:
            img = Image.open(img_path).convert('L')  # Grayscale
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return black image as fallback
            img = Image.new('L', (IMG_SIZE, IMG_SIZE), 0)

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        return {
            'img': img,
            'label': torch.tensor(label, dtype=torch.float32),
            'path': img_path,
        }

    def _handle_uncertainty(self, label: float) -> float:
        """Handle uncertain labels (-1) based on policy."""
        if label == -1:
            if self.uncertainty_policy == "zeros":
                return 0.0
            elif self.uncertainty_policy == "ones":
                return 1.0
            elif self.uncertainty_policy == "ignore":
                return np.nan
        return label

    def get_labels(self) -> np.ndarray:
        """Get all labels as numpy array."""
        return np.array(self.labels)


class CheXpertDataset(BaseChestXrayDataset):
    """
    CheXpert Dataset (Stanford Hospital).

    Contains 224,316 chest radiographs of 65,240 patients.
    """

    def __init__(
        self,
        root: str,
        tasks: List[str] = None,
        train: bool = True,
        transform: Callable = None,
        uncertainty_policy: str = "ones",
        max_samples: int = None,
    ):
        super().__init__(root, tasks, train, transform, uncertainty_policy, max_samples)

        # Try different CSV locations
        csv_path = None
        for candidate in [
            os.path.join(root, "train.csv"),
            os.path.join(root, "valid.csv"),
            os.path.join(root, "CheXpert-v1.0-small", "train.csv"),
        ]:
            if os.path.exists(candidate):
                csv_path = candidate
                break

        if csv_path is None:
            print(f"Warning: CheXpert CSV not found in {root}")
            print("Creating dummy dataset for testing...")
            self._create_dummy_data()
            return

        # Load metadata
        df = pd.read_csv(csv_path)

        # For train/test split when only train.csv available
        if train:
            df = df.sample(frac=0.8, random_state=42)
        else:
            all_df = pd.read_csv(csv_path)
            train_idx = all_df.sample(frac=0.8, random_state=42).index
            df = all_df.drop(train_idx)

        # Filter to competition tasks
        task_columns = self.tasks

        # Process each row
        for idx, row in df.iterrows():
            if self.max_samples and len(self.image_paths) >= self.max_samples:
                break

            # Get image path - handle different path formats
            rel_path = row['Path']
            # Remove prefix like "CheXpert-v1.0-small/" if present
            if rel_path.startswith('CheXpert'):
                rel_path = '/'.join(rel_path.split('/')[1:])

            img_path = os.path.join(root, rel_path)
            if not os.path.exists(img_path):
                # Try alternate location
                img_path = os.path.join(root, row['Path'])
                if not os.path.exists(img_path):
                    continue

            # Get labels
            label = []
            skip_sample = False
            for task in task_columns:
                val = row.get(task, 0.0)
                if pd.isna(val):
                    val = 0.0
                val = self._handle_uncertainty(val)
                if np.isnan(val) and self.uncertainty_policy == "ignore":
                    skip_sample = True
                    break
                label.append(max(0, val))  # Ensure non-negative

            if skip_sample:
                continue

            self.image_paths.append(img_path)
            self.labels.append(label)

        print(f"CheXpert {'train' if train else 'valid'}: {len(self)} samples")

    def _create_dummy_data(self):
        """Create dummy data for testing."""
        n_samples = self.max_samples or 100
        for i in range(n_samples):
            self.image_paths.append(f"dummy_{i}.jpg")
            self.labels.append([0.0] * len(self.tasks))


class MIMICCXRDataset(BaseChestXrayDataset):
    """
    MIMIC-CXR Dataset (Beth Israel Deaconess Medical Center).

    Contains 377,110 chest X-rays from 65,379 patients.
    Requires PhysioNet credentialed access.
    """

    def __init__(
        self,
        root: str,
        tasks: List[str] = None,
        train: bool = True,
        transform: Callable = None,
        uncertainty_policy: str = "ones",
        max_samples: int = None,
    ):
        super().__init__(root, tasks, train, transform, uncertainty_policy, max_samples)

        # MIMIC-CXR uses CheXpert labels
        split = "train" if train else "test"
        csv_path = os.path.join(root, f"mimic-cxr-2.0.0-chexpert.csv")
        split_path = os.path.join(root, f"mimic-cxr-2.0.0-split.csv")

        if not os.path.exists(csv_path):
            print(f"Warning: MIMIC-CXR labels not found: {csv_path}")
            print("Creating dummy dataset for testing...")
            self._create_dummy_data()
            return

        # Load metadata
        df_labels = pd.read_csv(csv_path)
        df_split = pd.read_csv(split_path)

        # Merge on subject_id and study_id
        df = pd.merge(df_labels, df_split, on=['subject_id', 'study_id'])

        # Filter by split
        df = df[df['split'] == split]

        # Process each row
        for idx, row in df.iterrows():
            if self.max_samples and len(self.image_paths) >= self.max_samples:
                break

            # Construct image path
            subject_id = str(row['subject_id'])
            study_id = str(row['study_id'])
            dicom_id = str(row.get('dicom_id', ''))

            # MIMIC-CXR path structure
            p_prefix = f"p{subject_id[:2]}"
            img_path = os.path.join(
                root, "files", p_prefix, f"p{subject_id}",
                f"s{study_id}", f"{dicom_id}.jpg"
            )

            if not os.path.exists(img_path):
                continue

            # Get labels
            label = []
            skip_sample = False
            for task in self.tasks:
                val = row.get(task, 0.0)
                if pd.isna(val):
                    val = 0.0
                val = self._handle_uncertainty(val)
                if np.isnan(val) and self.uncertainty_policy == "ignore":
                    skip_sample = True
                    break
                label.append(max(0, val))

            if skip_sample:
                continue

            self.image_paths.append(img_path)
            self.labels.append(label)

        print(f"MIMIC-CXR {'train' if train else 'test'}: {len(self)} samples")

    def _create_dummy_data(self):
        """Create dummy data for testing when real data unavailable."""
        n_samples = self.max_samples or 100
        for i in range(n_samples):
            self.image_paths.append(f"dummy_{i}.jpg")
            self.labels.append([0.0] * len(self.tasks))


class NIHChestXrayDataset(BaseChestXrayDataset):
    """
    NIH ChestX-ray14 Dataset (NIH Clinical Center).

    Contains 112,120 frontal-view X-ray images from 30,805 patients.
    """

    # Mapping from NIH labels to CheXpert labels
    LABEL_MAP = {
        'Atelectasis': 'Atelectasis',
        'Cardiomegaly': 'Cardiomegaly',
        'Consolidation': 'Consolidation',
        'Edema': 'Edema',
        'Effusion': 'Pleural Effusion',
    }

    def __init__(
        self,
        root: str,
        tasks: List[str] = None,
        train: bool = True,
        transform: Callable = None,
        uncertainty_policy: str = "binary",
        max_samples: int = None,
    ):
        super().__init__(root, tasks, train, transform, uncertainty_policy, max_samples)

        csv_path = os.path.join(root, "Data_Entry_2017.csv")
        train_list_path = os.path.join(root, "train_val_list.txt")
        test_list_path = os.path.join(root, "test_list.txt")

        if not os.path.exists(csv_path):
            print(f"Warning: NIH ChestX-ray14 metadata not found: {csv_path}")
            print("Creating dummy dataset for testing...")
            self._create_dummy_data()
            return

        # Load metadata
        df = pd.read_csv(csv_path)

        # Load split
        split_file = train_list_path if train else test_list_path
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                split_images = set(f.read().strip().split('\n'))
            df = df[df['Image Index'].isin(split_images)]
        else:
            # No split file - use 80/20 split based on patient ID
            df['patient_id'] = df['Patient ID'].astype(str)
            unique_patients = df['patient_id'].unique()
            np.random.seed(42)
            np.random.shuffle(unique_patients)
            split_idx = int(len(unique_patients) * 0.8)
            if train:
                train_patients = set(unique_patients[:split_idx])
                df = df[df['patient_id'].isin(train_patients)]
            else:
                test_patients = set(unique_patients[split_idx:])
                df = df[df['patient_id'].isin(test_patients)]

        # Build image path lookup (images are in images_XXX/images/ subdirectories)
        image_lookup = {}
        for subdir in os.listdir(root):
            if subdir.startswith('images_'):
                images_path = os.path.join(root, subdir, 'images')
                if os.path.isdir(images_path):
                    for img_file in os.listdir(images_path):
                        if img_file.endswith('.png'):
                            image_lookup[img_file] = os.path.join(images_path, img_file)

        # Process each row
        for idx, row in df.iterrows():
            if self.max_samples and len(self.image_paths) >= self.max_samples:
                break

            img_name = row['Image Index']
            img_path = image_lookup.get(img_name)
            if img_path is None or not os.path.exists(img_path):
                continue

            # Parse labels (format: "Finding1|Finding2|...")
            findings = row['Finding Labels'].split('|')

            # Convert to multi-label format
            label = []
            for task in self.tasks:
                # Check if any NIH label maps to this task
                found = False
                for nih_label, chexpert_label in self.LABEL_MAP.items():
                    if chexpert_label == task and nih_label in findings:
                        found = True
                        break
                label.append(1.0 if found else 0.0)

            self.image_paths.append(img_path)
            self.labels.append(label)

        print(f"NIH ChestX-ray14 {'train' if train else 'test'}: {len(self)} samples")

    def _create_dummy_data(self):
        """Create dummy data for testing."""
        n_samples = self.max_samples or 100
        for i in range(n_samples):
            self.image_paths.append(f"dummy_{i}.png")
            self.labels.append([0.0] * len(self.tasks))


class VinDrCXRDataset(BaseChestXrayDataset):
    """
    VinDr-CXR Dataset (Vingroup Big Data Institute, Vietnam).

    Contains ~18,000 chest X-ray images.
    """

    # Mapping from VinDr labels to CheXpert labels
    LABEL_MAP = {
        'Aortic enlargement': None,
        'Atelectasis': 'Atelectasis',
        'Calcification': None,
        'Cardiomegaly': 'Cardiomegaly',
        'Consolidation': 'Consolidation',
        'ILD': None,
        'Infiltration': None,
        'Lung Opacity': None,
        'Nodule/Mass': None,
        'Other lesion': None,
        'Pleural effusion': 'Pleural Effusion',
        'Pleural thickening': None,
        'Pneumothorax': None,
        'Pulmonary fibrosis': None,
    }

    def __init__(
        self,
        root: str,
        tasks: List[str] = None,
        train: bool = True,
        transform: Callable = None,
        uncertainty_policy: str = "binary",
        max_samples: int = None,
    ):
        super().__init__(root, tasks, train, transform, uncertainty_policy, max_samples)

        split = "train" if train else "test"
        csv_path = os.path.join(root, f"annotations/{split}.csv")

        if not os.path.exists(csv_path):
            print(f"Warning: VinDr-CXR metadata not found: {csv_path}")
            print("Creating dummy dataset for testing...")
            self._create_dummy_data()
            return

        # Load annotations
        df = pd.read_csv(csv_path)

        # Group by image
        image_groups = df.groupby('image_id')

        for image_id, group in image_groups:
            if self.max_samples and len(self.image_paths) >= self.max_samples:
                break

            img_path = os.path.join(root, f"{split}/{image_id}.dicom")
            # Also check for png/jpg
            for ext in ['.dicom', '.png', '.jpg']:
                test_path = os.path.join(root, f"{split}/{image_id}{ext}")
                if os.path.exists(test_path):
                    img_path = test_path
                    break

            if not os.path.exists(img_path):
                continue

            # Get all findings for this image
            findings = set(group['class_name'].values)

            # Convert to multi-label format
            label = []
            for task in self.tasks:
                found = False
                for vindr_label, chexpert_label in self.LABEL_MAP.items():
                    if chexpert_label == task and vindr_label in findings:
                        found = True
                        break
                label.append(1.0 if found else 0.0)

            self.image_paths.append(img_path)
            self.labels.append(label)

        print(f"VinDr-CXR {'train' if train else 'test'}: {len(self)} samples")

    def _create_dummy_data(self):
        """Create dummy data for testing."""
        n_samples = self.max_samples or 100
        for i in range(n_samples):
            self.image_paths.append(f"dummy_{i}.dicom")
            self.labels.append([0.0] * len(self.tasks))


class RSNAPneumoniaDataset(BaseChestXrayDataset):
    """
    RSNA Pneumonia Detection Challenge Dataset.

    Contains 26,684 chest X-ray images in DICOM format.
    Binary classification: Pneumonia vs No Pneumonia.
    """

    def __init__(
        self,
        root: str,
        tasks: List[str] = None,
        train: bool = True,
        transform: Callable = None,
        uncertainty_policy: str = "binary",
        max_samples: int = None,
    ):
        super().__init__(root, tasks, train, transform, uncertainty_policy, max_samples)

        # RSNA has binary pneumonia labels
        labels_csv = os.path.join(root, "stage_2_train_labels.csv")
        detailed_csv = os.path.join(root, "stage_2_detailed_class_info.csv")

        if not os.path.exists(labels_csv):
            print(f"Warning: RSNA labels not found: {labels_csv}")
            print("Creating dummy dataset for testing...")
            self._create_dummy_data()
            return

        # Load labels
        df = pd.read_csv(labels_csv)
        # Get unique patients (some have multiple bboxes)
        df = df.drop_duplicates(subset='patientId')

        # Load detailed class info for multi-class if available
        if os.path.exists(detailed_csv):
            df_detailed = pd.read_csv(detailed_csv)
            df = df.merge(df_detailed, on='patientId', how='left')

        # Train/test split (80/20)
        np.random.seed(42)
        patient_ids = df['patientId'].unique()
        np.random.shuffle(patient_ids)
        split_idx = int(len(patient_ids) * 0.8)

        if train:
            selected_ids = set(patient_ids[:split_idx])
        else:
            selected_ids = set(patient_ids[split_idx:])

        df = df[df['patientId'].isin(selected_ids)]

        # Find image directory
        img_dir = os.path.join(root, "stage_2_train_images")
        if not os.path.exists(img_dir):
            img_dir = os.path.join(root, "train_images")

        # Process each row
        for idx, row in df.iterrows():
            if self.max_samples and len(self.image_paths) >= self.max_samples:
                break

            patient_id = row['patientId']
            img_path = os.path.join(img_dir, f"{patient_id}.dcm")

            if not os.path.exists(img_path):
                continue

            # Map RSNA binary to multi-label format
            # Target=1 means pneumonia, which maps to Consolidation
            target = row.get('Target', 0)
            label = []
            for task in self.tasks:
                if task == 'Consolidation' and target == 1:
                    label.append(1.0)
                elif task == 'Pleural Effusion' and 'class' in row:
                    # "Lung Opacity" class might indicate effusion
                    label.append(1.0 if 'Lung Opacity' in str(row.get('class', '')) else 0.0)
                else:
                    label.append(0.0)

            self.image_paths.append(img_path)
            self.labels.append(label)

        print(f"RSNA Pneumonia {'train' if train else 'test'}: {len(self)} samples")

    def __getitem__(self, idx):
        """Override to handle DICOM files."""
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            import pydicom
            dcm = pydicom.dcmread(img_path)
            img_array = dcm.pixel_array

            # Normalize to 0-255
            img_array = img_array - img_array.min()
            if img_array.max() > 0:
                img_array = img_array / img_array.max() * 255
            img_array = img_array.astype(np.uint8)

            img = Image.fromarray(img_array).convert('L')
        except Exception as e:
            print(f"Error loading DICOM {img_path}: {e}")
            img = Image.new('L', (IMG_SIZE, IMG_SIZE), 0)

        if self.transform:
            img = self.transform(img)

        return {
            'img': img,
            'label': torch.tensor(label, dtype=torch.float32),
            'path': img_path,
        }

    def _create_dummy_data(self):
        """Create dummy data for testing."""
        n_samples = self.max_samples or 100
        for i in range(n_samples):
            self.image_paths.append(f"dummy_{i}.dcm")
            self.labels.append([0.0] * len(self.tasks))


class SyntheticChestXrayDataset(Dataset):
    """
    Synthetic chest X-ray dataset for testing.

    Generates random images with controllable label distributions.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        tasks: List[str] = None,
        label_dist: Dict[str, float] = None,
        img_size: int = IMG_SIZE,
        seed: int = 42,
    ):
        """
        Args:
            num_samples: Number of samples to generate
            tasks: List of tasks
            label_dist: Label distribution (prevalence) for each task
            img_size: Image size
            seed: Random seed
        """
        self.tasks = tasks or CHEXPERT_COMPETITION_TASKS
        self.num_samples = num_samples
        self.img_size = img_size
        self.classnames = self.tasks

        np.random.seed(seed)

        # Default label distribution (approximate CheXpert prevalence)
        if label_dist is None:
            label_dist = {
                'Atelectasis': 0.3,
                'Cardiomegaly': 0.2,
                'Consolidation': 0.1,
                'Edema': 0.15,
                'Pleural Effusion': 0.25,
            }

        # Generate labels
        self.labels = []
        for _ in range(num_samples):
            label = []
            for task in self.tasks:
                prob = label_dist.get(task, 0.1)
                label.append(1.0 if np.random.random() < prob else 0.0)
            self.labels.append(label)

        self.labels = np.array(self.labels)

        # Transform for synthetic images
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[IMG_MEAN], std=[IMG_STD]),
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random grayscale image
        np.random.seed(idx)
        img = np.random.randn(self.img_size, self.img_size).astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]

        img = Image.fromarray((img * 255).astype(np.uint8), mode='L')
        img = self.transform(img)
        img = img.repeat(3, 1, 1)  # Convert to RGB

        return {
            'img': img,
            'label': torch.tensor(self.labels[idx], dtype=torch.float32),
            'path': f'synthetic_{idx}',
        }


def get_dataset(
    dataset_name: str,
    root: str,
    train: bool = True,
    tasks: List[str] = None,
    transform: Callable = None,
    max_samples: int = None,
) -> BaseChestXrayDataset:
    """
    Factory function to get dataset by name.

    Args:
        dataset_name: One of 'chexpert', 'mimic_cxr', 'nih_chestxray', 'vindr_cxr', 'synthetic'
        root: Root directory
        train: Whether to load training set
        tasks: List of tasks
        transform: Image transforms
        max_samples: Maximum samples

    Returns:
        Dataset instance
    """
    dataset_map = {
        'chexpert': CheXpertDataset,
        'mimic_cxr': MIMICCXRDataset,
        'nih_chestxray': NIHChestXrayDataset,
        'vindr_cxr': VinDrCXRDataset,
        'rsna_pneumonia': RSNAPneumoniaDataset,
        'synthetic': SyntheticChestXrayDataset,
    }

    if dataset_name not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(dataset_map.keys())}")

    if dataset_name == 'synthetic':
        return SyntheticChestXrayDataset(
            num_samples=max_samples or 1000,
            tasks=tasks,
        )

    return dataset_map[dataset_name](
        root=root,
        tasks=tasks,
        train=train,
        transform=transform,
        max_samples=max_samples,
    )


class FederatedChestXrayDataset:
    """
    Federated dataset manager for chest X-ray datasets.

    Supports two modes:
    1. Real mode: Each hospital dataset = one client
    2. Virtual mode: Split real datasets into multiple virtual clients

    Virtual client mode is useful when you have few real datasets but want
    to simulate more hospitals with heterogeneous data distributions.
    """

    def __init__(
        self,
        data_roots: Dict[str, str],
        tasks: List[str] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        max_samples_per_client: int = None,
        use_virtual_clients: bool = False,
        virtual_clients_per_dataset: int = 4,
        dirichlet_alpha: float = 0.3,
        seed: int = 42,
    ):
        """
        Args:
            data_roots: Dictionary mapping dataset names to root paths
            tasks: List of tasks
            batch_size: Batch size for data loaders
            num_workers: Number of data loading workers
            max_samples_per_client: Max samples per client (for debugging)
            use_virtual_clients: If True, split each dataset into virtual clients
            virtual_clients_per_dataset: Number of virtual clients per real dataset
            dirichlet_alpha: Concentration parameter for Dirichlet splitting
                             (lower = more heterogeneous)
            seed: Random seed for virtual client splitting
        """
        self.data_roots = data_roots
        self.tasks = tasks or CHEXPERT_COMPETITION_TASKS
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_samples = max_samples_per_client
        self.use_virtual_clients = use_virtual_clients
        self.virtual_clients_per_dataset = virtual_clients_per_dataset
        self.dirichlet_alpha = dirichlet_alpha
        self.seed = seed

        # Will be set based on mode
        self.num_clients = 0
        self.client_names = []
        self._virtual_client_manager = None

        if use_virtual_clients:
            self._setup_virtual_clients()
        else:
            self._setup_real_clients()

        self.classnames = self.tasks

    def _setup_real_clients(self):
        """Setup with each dataset as a separate client."""
        self.num_clients = len(self.data_roots)
        self.client_names = list(self.data_roots.keys())

        # Load datasets
        self.train_datasets = {}
        self.test_datasets = {}
        self.train_loaders = {}
        self.test_loaders = {}

        for name, root in self.data_roots.items():
            print(f"\nLoading {name} dataset from {root}")

            self.train_datasets[name] = get_dataset(
                name, root, train=True, tasks=self.tasks,
                max_samples=self.max_samples
            )
            self.test_datasets[name] = get_dataset(
                name, root, train=False, tasks=self.tasks,
                max_samples=self.max_samples
            )

            self.train_loaders[name] = DataLoader(
                self.train_datasets[name],
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
            )
            self.test_loaders[name] = DataLoader(
                self.test_datasets[name],
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )

    def _setup_virtual_clients(self):
        """Setup with virtual clients split from real datasets."""
        from .virtual_clients import FederatedVirtualClients

        # Load base datasets
        base_train_datasets = {}
        base_test_datasets = {}

        for name, root in self.data_roots.items():
            print(f"\nLoading {name} dataset from {root}")

            base_train_datasets[name] = get_dataset(
                name, root, train=True, tasks=self.tasks,
                max_samples=self.max_samples
            )
            base_test_datasets[name] = get_dataset(
                name, root, train=False, tasks=self.tasks,
                max_samples=self.max_samples
            )

        # Create virtual client managers
        print(f"\nSplitting into virtual clients (alpha={self.dirichlet_alpha})...")
        self._virtual_train_manager = FederatedVirtualClients(
            datasets=base_train_datasets,
            clients_per_dataset=self.virtual_clients_per_dataset,
            alpha=self.dirichlet_alpha,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            seed=self.seed,
        )

        self._virtual_test_manager = FederatedVirtualClients(
            datasets=base_test_datasets,
            clients_per_dataset=self.virtual_clients_per_dataset,
            alpha=self.dirichlet_alpha,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            seed=self.seed + 1000,  # Different seed for test split
        )

        self.num_clients = self._virtual_train_manager.num_clients
        self.client_names = [
            self._virtual_train_manager.get_client_name(i)
            for i in range(self.num_clients)
        ]

        # Store reference for compatibility
        self.train_loaders = self._virtual_train_manager.train_loaders
        self.test_loaders = self._virtual_test_manager.test_loaders

    def get_client_loader(self, client_id: int, train: bool = True) -> DataLoader:
        """Get data loader for a specific client."""
        if self.use_virtual_clients:
            if train:
                return self._virtual_train_manager.get_client_loader(client_id, train=True)
            return self._virtual_test_manager.get_client_loader(client_id, train=False)
        else:
            name = self.client_names[client_id]
            return self.train_loaders[name] if train else self.test_loaders[name]

    def get_client_dataset(self, client_id: int, train: bool = True) -> Dataset:
        """Get dataset for a specific client."""
        if self.use_virtual_clients:
            if train:
                return self._virtual_train_manager.get_client_dataset(client_id)
            return self._virtual_test_manager.get_client_dataset(client_id)
        else:
            name = self.client_names[client_id]
            return self.train_datasets[name] if train else self.test_datasets[name]

    def get_client_name(self, client_id: int) -> str:
        """Get name of client dataset."""
        return self.client_names[client_id]

    def get_num_samples(self, client_id: int, train: bool = True) -> int:
        """Get number of samples for a client."""
        dataset = self.get_client_dataset(client_id, train)
        return len(dataset)

    def get_heterogeneity_stats(self) -> Optional[Dict]:
        """
        Get heterogeneity statistics for virtual clients.

        Returns None if not using virtual clients.
        """
        if self.use_virtual_clients and self._virtual_train_manager is not None:
            return self._virtual_train_manager.get_heterogeneity_stats()
        return None

    def get_label_distribution(self, client_id: int, train: bool = True) -> Optional[np.ndarray]:
        """
        Get label distribution (class prevalences) for a client.

        Returns:
            numpy array of shape (n_classes,) with prevalence of each class
        """
        dataset = self.get_client_dataset(client_id, train)

        if hasattr(dataset, 'labels'):
            labels = np.array(dataset.labels)
            if len(labels.shape) > 1:
                return labels.mean(axis=0)
        return None

    def print_client_summary(self):
        """Print summary of all clients and their data distributions."""
        print("\n" + "=" * 70)
        print("Federated Dataset Summary")
        print("=" * 70)
        print(f"Mode: {'Virtual Clients' if self.use_virtual_clients else 'Real Datasets'}")
        print(f"Number of clients: {self.num_clients}")
        print(f"Tasks: {self.tasks}")
        print("-" * 70)

        for client_id in range(self.num_clients):
            name = self.get_client_name(client_id)
            n_train = self.get_num_samples(client_id, train=True)
            n_test = self.get_num_samples(client_id, train=False)

            label_dist = self.get_label_distribution(client_id, train=True)
            if label_dist is not None:
                dist_str = ", ".join([f"{v:.2f}" for v in label_dist])
            else:
                dist_str = "N/A"

            print(f"Client {client_id:2d} ({name:20s}): "
                  f"train={n_train:5d}, test={n_test:5d}, dist=[{dist_str}]")

        print("=" * 70)


def create_heterogeneous_split(
    dataset: Dataset,
    num_clients: int,
    alpha: float = 0.3,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """
    Create heterogeneous data split using Dirichlet distribution.

    This simulates label skew across clients.

    Args:
        dataset: Base dataset
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter (lower = more heterogeneous)
        seed: Random seed

    Returns:
        Dictionary mapping client ID to list of sample indices
    """
    np.random.seed(seed)

    labels = np.array(dataset.labels)
    n_classes = labels.shape[1]
    n_samples = len(labels)

    client_indices = {i: [] for i in range(num_clients)}

    # For each class (multi-label: consider each label independently)
    for c in range(n_classes):
        # Get indices where this class is positive
        class_indices = np.where(labels[:, c] == 1)[0]

        if len(class_indices) == 0:
            continue

        # Sample proportions from Dirichlet
        proportions = np.random.dirichlet([alpha] * num_clients)

        # Assign indices to clients
        np.random.shuffle(class_indices)
        split_points = (proportions.cumsum() * len(class_indices)).astype(int)[:-1]
        client_splits = np.split(class_indices, split_points)

        for client_id, indices in enumerate(client_splits):
            client_indices[client_id].extend(indices.tolist())

    # Remove duplicates and shuffle
    for client_id in client_indices:
        client_indices[client_id] = list(set(client_indices[client_id]))
        np.random.shuffle(client_indices[client_id])

    return client_indices
