# Code Documentation

## Repository Structure

```
fedtpg/
├── medpromptfolio/                 # Main package
│   ├── __init__.py                 # Package exports
│   ├── constants.py                # Configuration constants
│   ├── medclip_prompt.py          # MedCLIP + prompt learning
│   ├── med_prompt_folio.py        # MedPromptFolio algorithm
│   ├── chest_xray_datasets.py     # Dataset loaders
│   ├── virtual_clients.py         # Virtual client splitting
│   ├── federated_trainer.py       # FL training loop
│   ├── visualization.py           # Heterogeneity plots
│   ├── utils.py                   # Utility functions
│   ├── configs/
│   │   └── default.yaml           # Default configuration
│   └── scripts/
│       ├── train.py               # Training script
│       └── run_experiments.sh     # Experiment runner
│
├── data/                           # Dataset directory
│   ├── nih_chestxray/
│   ├── chexpert/
│   └── rsna_pneumonia/
│
├── experiments/                    # Experiment outputs
│   └── full_exp_YYYYMMDD_HHMMSS/
│
├── docs/                           # Documentation
│   ├── 01_PROJECT_OVERVIEW.md
│   ├── 02_DATASETS.md
│   ├── 03_TRAINING_METHODOLOGY.md
│   ├── 04_NOVEL_CONTRIBUTIONS.md
│   ├── 05_EXPERIMENTAL_SETUP.md
│   └── 06_CODE_DOCUMENTATION.md
│
├── run_full_experiment.py          # Full experiment script
├── .gitignore
└── README.md
```

---

## Core Modules

### 1. `medclip_prompt.py` - MedCLIP with Prompt Learning

#### Classes

```python
class MedCLIPTextEncoder(nn.Module):
    """
    Text encoder using Bio_ClinicalBERT.
    Encodes class names with learnable prompt context.
    """

    def forward(self, prompts, tokenized_prompts):
        """
        Args:
            prompts: Tensor of shape (n_cls, n_ctx + n_tokens, dim)
            tokenized_prompts: Tokenized class names

        Returns:
            text_features: Tensor of shape (n_cls, dim)
        """


class MedCLIPVisionEncoder(nn.Module):
    """
    Vision encoder using Swin Transformer.
    """

    def forward(self, images):
        """
        Args:
            images: Tensor of shape (batch, 3, 224, 224)

        Returns:
            image_features: Tensor of shape (batch, dim)
        """


class MedCLIPPromptLearner(nn.Module):
    """
    Learnable prompt context for MedCLIP.

    Maintains two sets of prompts:
    - global_ctx: Shared across all clients
    - local_ctx: Client-specific
    """

    def __init__(self, classnames, medclip_model, n_ctx=4, ctx_init="random"):
        self.n_ctx = n_ctx  # Number of context tokens
        self.ctx_dim = 768  # BERT hidden dimension

        # Learnable context vectors
        self.global_ctx = nn.Parameter(torch.randn(n_ctx, ctx_dim))
        self.local_ctx = nn.Parameter(torch.randn(n_ctx, ctx_dim))


class CustomMedCLIP(nn.Module):
    """
    Complete MedCLIP model with prompt learning.
    """

    def forward(self, images, theta=0.3):
        """
        Args:
            images: Input images (batch, 3, 224, 224)
            theta: Portfolio mixing coefficient

        Returns:
            logits: Classification logits (batch, n_classes)
        """
        # Mix prompts
        mixed_ctx = (1 - theta) * self.prompt_learner.global_ctx + \
                    theta * self.prompt_learner.local_ctx

        # Get features
        prompts = self.prompt_learner(mixed_ctx)
        text_features = self.text_encoder(prompts)
        image_features = self.image_encoder(images)

        # Compute similarity
        logits = self.logit_scale * (image_features @ text_features.T)

        return logits
```

---

### 2. `med_prompt_folio.py` - Federated Algorithm

#### Classes

```python
class MedPromptFolio:
    """
    MedPromptFolio: Portfolio-based federated prompt learning.

    Key methods:
    - fed_init_model: Initialize client models
    - fed_download_model: Download global prompt to client
    - fed_upload_model: Upload client's global prompt to server
    - fed_aggregate_model: Aggregate global prompts from clients
    """

    def __init__(self, classnames, medclip_checkpoint, n_ctx=4, theta=0.3, device='cuda'):
        self.theta = theta
        self.model = CustomMedCLIP(classnames, medclip_checkpoint, n_ctx, device)

        # Storage for federated learning
        self.global_prompt = None  # Server's global prompt
        self.client_prompts = {}   # Client-specific local prompts

    def fed_init_model(self, global_weights):
        """Initialize global prompt."""
        self.global_prompt = copy.deepcopy(global_weights)

    def fed_download_model(self, client_id):
        """Client downloads global prompt from server."""
        self.model.load_global_prompt(self.global_prompt)
        if client_id in self.client_prompts:
            self.model.load_local_prompt(self.client_prompts[client_id])

    def fed_upload_model(self, client_id):
        """Client uploads updated global prompt."""
        return self.model.get_global_prompt()

    def fed_aggregate_model(self, client_weights, client_sizes):
        """Aggregate global prompts using weighted average."""
        total_size = sum(client_sizes.values())
        aggregated = {}

        for key in client_weights[0].keys():
            aggregated[key] = sum(
                client_weights[i][key] * client_sizes[i] / total_size
                for i in range(len(client_weights))
            )

        self.global_prompt = aggregated


class MedPromptFolioTrainer:
    """
    Training loop for MedPromptFolio.
    """

    def train_epoch(self, dataloader):
        """Train for one local epoch."""
        for batch in dataloader:
            images = batch['img'].to(self.device)
            labels = batch['label'].to(self.device)

            # Forward
            logits = self.model(images, theta=self.theta)
            loss = self.criterion(logits, labels)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def evaluate(self, dataloader):
        """Evaluate model."""
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in dataloader:
                logits = self.model(batch['img'].to(self.device))
                all_preds.append(torch.sigmoid(logits).cpu())
                all_labels.append(batch['label'])

        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_labels).numpy()

        return {'auc': roc_auc_score(y_true, y_pred, average='macro')}
```

---

### 3. `chest_xray_datasets.py` - Dataset Loaders

#### Classes

```python
class BaseChestXrayDataset(Dataset):
    """Base class for chest X-ray datasets."""

    def __getitem__(self, idx):
        return {
            'img': self.transform(Image.open(self.image_paths[idx])),
            'label': torch.tensor(self.labels[idx]),
            'path': self.image_paths[idx],
        }


class NIHChestXrayDataset(BaseChestXrayDataset):
    """NIH ChestX-ray14 dataset loader."""

    LABEL_MAP = {
        'Atelectasis': 'Atelectasis',
        'Cardiomegaly': 'Cardiomegaly',
        'Consolidation': 'Consolidation',
        'Edema': 'Edema',
        'Effusion': 'Pleural Effusion',
    }


class CheXpertDataset(BaseChestXrayDataset):
    """CheXpert dataset loader."""


class RSNAPneumoniaDataset(BaseChestXrayDataset):
    """RSNA Pneumonia dataset loader (DICOM support)."""


class FederatedChestXrayDataset:
    """
    Federated dataset manager.

    Supports two modes:
    1. Real mode: Each dataset = 1 client
    2. Virtual mode: Split datasets into virtual clients
    """

    def __init__(self, data_roots, use_virtual_clients=False,
                 virtual_clients_per_dataset=3, dirichlet_alpha=0.3, ...):
        if use_virtual_clients:
            self._setup_virtual_clients()
        else:
            self._setup_real_clients()

    def get_client_loader(self, client_id, train=True):
        """Get DataLoader for a specific client."""

    def get_client_name(self, client_id):
        """Get name of client (e.g., 'nih_chestxray_client_0')."""
```

---

### 4. `virtual_clients.py` - Heterogeneity Simulation

#### Classes

```python
class VirtualClientSplitter:
    """
    Split dataset into virtual clients using Dirichlet distribution.
    """

    def __init__(self, num_clients=4, alpha=0.3, min_samples_per_client=50):
        self.num_clients = num_clients
        self.alpha = alpha

    def split_by_dirichlet(self, labels, multi_label=True):
        """
        Split dataset indices using Dirichlet distribution.

        Returns:
            Dict[int, List[int]]: client_id -> sample indices
        """


class VirtualClientDataset(Dataset):
    """Wrapper for virtual client's subset of data."""


class FederatedVirtualClients:
    """
    Manager for creating virtual clients from multiple datasets.

    Example:
        fed = FederatedVirtualClients(
            datasets={'nih': nih_data, 'chexpert': chexpert_data},
            clients_per_dataset=3,
            alpha=0.3,
        )
        # Creates 6 virtual clients total
    """
```

---

### 5. `federated_trainer.py` - Training Orchestration

```python
class FederatedTrainer:
    """
    Orchestrates federated training across all clients.
    """

    def __init__(self, model, federated_dataset, num_rounds=50, ...):
        self.model = model
        self.federated_dataset = federated_dataset
        self.num_rounds = num_rounds

    def train(self):
        """
        Run full federated training.

        Returns:
            Dict with training history and final metrics
        """
        for round_num in range(self.num_rounds):
            # Select clients
            selected_clients = self.select_clients()

            # Local training
            client_updates = []
            for client_id in selected_clients:
                self.model.fed_download_model(client_id)
                self.train_client(client_id)
                update = self.model.fed_upload_model(client_id)
                client_updates.append(update)

            # Aggregate
            self.model.fed_aggregate_model(client_updates, self.client_sizes)

            # Evaluate
            metrics = self.evaluate_round(round_num)
            self.log_metrics(metrics)

        return self.history


def run_experiment(method, data_roots, num_rounds, ...):
    """
    Convenience function to run a complete experiment.
    """
```

---

### 6. `visualization.py` - Heterogeneity Visualization

```python
def plot_label_distribution_heatmap(distributions, client_names, class_names, ...):
    """Plot heatmap of label distributions across clients."""

def plot_sample_distribution_bar(samples_per_client, client_names, ...):
    """Plot bar chart of sample counts per client."""

def plot_js_divergence_matrix(distributions, client_names, ...):
    """Plot pairwise Jensen-Shannon divergence between clients."""

def create_heterogeneity_report(fed_dataset, output_dir, prefix='heterogeneity'):
    """Generate comprehensive heterogeneity report with all visualizations."""
```

---

## Usage Examples

### Basic Training

```python
from medpromptfolio import (
    MedPromptFolio,
    FederatedTrainer,
    FederatedChestXrayDataset,
    CHEXPERT_COMPETITION_TASKS,
)

# Load data
fed_dataset = FederatedChestXrayDataset(
    data_roots={
        'nih_chestxray': 'data/nih_chestxray',
        'chexpert': 'data/chexpert',
        'rsna_pneumonia': 'data/rsna_pneumonia',
    },
    use_virtual_clients=True,
    virtual_clients_per_dataset=3,
    dirichlet_alpha=0.3,
)

# Create model
model = MedPromptFolio(
    classnames=CHEXPERT_COMPETITION_TASKS,
    medclip_checkpoint='path/to/medclip-vit',
    n_ctx=4,
    theta=0.3,
)

# Train
trainer = FederatedTrainer(
    model=model,
    federated_dataset=fed_dataset,
    num_rounds=50,
    local_epochs=2,
)

results = trainer.train()
print(f"Best AUC: {results['best_auc']:.4f}")
```

### Custom Heterogeneity

```python
# Extreme heterogeneity
fed_dataset = FederatedChestXrayDataset(
    data_roots={...},
    use_virtual_clients=True,
    dirichlet_alpha=0.1,  # Very non-IID
)

# Visualize heterogeneity
from medpromptfolio import create_heterogeneity_report
create_heterogeneity_report(fed_dataset, 'output/plots/')
```

### Baseline Comparison

```python
from medpromptfolio import FedAvgMedCLIP, FedProxMedCLIP, LocalOnlyMedCLIP

methods = {
    'medpromptfolio': MedPromptFolio(..., theta=0.3),
    'fedavg': FedAvgMedCLIP(...),
    'fedprox': FedProxMedCLIP(..., mu=0.01),
    'local': LocalOnlyMedCLIP(...),
}

for name, model in methods.items():
    trainer = FederatedTrainer(model, fed_dataset, ...)
    results = trainer.train()
    print(f"{name}: AUC = {results['best_auc']:.4f}")
```

---

## Configuration Reference

### `configs/default.yaml`

```yaml
# Data
data:
  tasks:
    - Atelectasis
    - Cardiomegaly
    - Consolidation
    - Edema
    - Pleural Effusion
  img_size: 224
  img_mean: 0.586
  img_std: 0.279

# Federated Learning
federated:
  num_rounds: 50
  local_epochs: 2
  client_fraction: 1.0
  batch_size: 32

# Virtual Clients
virtual_clients:
  enabled: true
  clients_per_dataset: 3
  dirichlet_alpha: 0.3

# Model
model:
  backbone: medclip-vit
  n_ctx: 4
  theta: 0.3

# Optimization
optimization:
  lr: 0.002
  weight_decay: 0.01
  use_amp: true

# Logging
logging:
  output_dir: ./output
  save_every: 10
  verbose: true
```

---

## Quick Links

- [Project Overview](./01_PROJECT_OVERVIEW.md)
- [Dataset Documentation](./02_DATASETS.md)
- [Training Methodology](./03_TRAINING_METHODOLOGY.md)
- [Novel Contributions](./04_NOVEL_CONTRIBUTIONS.md)
- [Experimental Setup](./05_EXPERIMENTAL_SETUP.md)
