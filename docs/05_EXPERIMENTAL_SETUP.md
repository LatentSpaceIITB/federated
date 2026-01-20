# Experimental Setup

## Overview

This document details the complete experimental setup for MedPromptFolio, including virtual client creation, heterogeneity simulation, and evaluation protocols.

---

## 1. Virtual Client Splitting

### Why Virtual Clients?

With only 3 real datasets, we simulate a larger federation by splitting each dataset into multiple virtual clients with **heterogeneous data distributions**.

```
Real Datasets (3)              Virtual Clients (9)
┌─────────────────┐           ┌─────────────────┐
│ NIH ChestX-ray  │──split──▶│ NIH_client_0    │ (specialized in Effusion)
│    (~112K)      │           │ NIH_client_1    │ (balanced)
│                 │           │ NIH_client_2    │ (specialized in Cardiac)
└─────────────────┘           └─────────────────┘

┌─────────────────┐           ┌─────────────────┐
│    CheXpert     │──split──▶│ CheXpert_client_0│ (high Edema)
│    (~224K)      │           │ CheXpert_client_1│ (balanced)
│                 │           │ CheXpert_client_2│ (high Atelectasis)
└─────────────────┘           └─────────────────┘

┌─────────────────┐           ┌─────────────────┐
│ RSNA Pneumonia  │──split──▶│ RSNA_client_0   │ (high pneumonia)
│    (~27K)       │           │ RSNA_client_1   │ (low pneumonia)
│                 │           │ RSNA_client_2   │ (mixed)
└─────────────────┘           └─────────────────┘
```

### Dirichlet Partitioning

We use the **Dirichlet distribution** to create non-IID data splits:

```python
# For each class c, sample client proportions from Dirichlet
proportions = np.random.dirichlet([α] * num_clients)

# α controls heterogeneity:
# α = 0.1: Extreme (each client gets mostly 1-2 classes)
# α = 0.3: High (our default)
# α = 0.5: Moderate
# α = 1.0: Mild (uniform random)
# α = 10.0: Near-IID (almost equal distribution)
```

### Algorithm

```python
def split_by_dirichlet(labels, num_clients, alpha=0.3):
    """
    Split dataset into clients using Dirichlet distribution.

    Args:
        labels: Array of shape (n_samples, n_classes)
        num_clients: Number of virtual clients
        alpha: Dirichlet concentration parameter

    Returns:
        client_indices: Dict mapping client_id -> list of sample indices
    """
    n_classes = labels.shape[1]
    client_indices = {i: [] for i in range(num_clients)}

    for class_idx in range(n_classes):
        # Get samples with this class positive
        class_samples = np.where(labels[:, class_idx] == 1)[0]

        # Sample proportions from Dirichlet
        proportions = np.random.dirichlet([alpha] * num_clients)

        # Split samples according to proportions
        np.random.shuffle(class_samples)
        splits = np.split(class_samples,
                         (proportions.cumsum()[:-1] * len(class_samples)).astype(int))

        # Assign to clients
        for client_id, indices in enumerate(splits):
            client_indices[client_id].extend(indices)

    return client_indices
```

---

## 2. Heterogeneity Levels

### Configuration Presets

```python
HETEROGENEITY_CONFIGS = {
    'extreme': {
        'alpha': 0.1,
        'description': 'Each client specializes in 1-2 pathologies',
        'expected_js_div': 0.45,
    },
    'high': {
        'alpha': 0.3,
        'description': 'Significant label skew across clients',
        'expected_js_div': 0.35,
    },
    'moderate': {
        'alpha': 0.5,
        'description': 'Moderate heterogeneity',
        'expected_js_div': 0.25,
    },
    'mild': {
        'alpha': 1.0,
        'description': 'Mild heterogeneity (uniform random)',
        'expected_js_div': 0.15,
    },
    'iid': {
        'alpha': 10.0,
        'description': 'Near-IID distribution',
        'expected_js_div': 0.05,
    },
}
```

### Example Distribution (α=0.3, 9 clients)

```
Client               Atel   Card   Cons   Edem   PlEf
────────────────────────────────────────────────────
NIH_client_0         0.10   0.04   0.01   0.00   0.20
NIH_client_1         0.08   0.00   0.01   0.01   0.08
NIH_client_2         0.02   0.48   0.43   0.04   0.23
CheXpert_client_0    0.01   0.00   0.23   0.73   0.50
CheXpert_client_1    0.38   0.18   0.18   0.17   0.42
CheXpert_client_2    0.28   0.36   0.22   0.36   0.42
RSNA_client_0        0.00   0.00   0.99   0.00   0.99
RSNA_client_1        0.00   0.00   0.53   0.00   0.58
RSNA_client_2        0.00   0.00   0.02   0.00   0.97
```

Notice how different clients have very different label distributions!

---

## 3. Experiment Configurations

### Main Experiment

```python
MAIN_CONFIG = {
    # Data
    'datasets': ['nih_chestxray', 'chexpert', 'rsna_pneumonia'],
    'virtual_clients_per_dataset': 3,  # Total: 9 clients
    'max_samples_per_client': 2000,
    'dirichlet_alpha': 0.3,

    # Federated Learning
    'num_rounds': 50,
    'local_epochs': 2,
    'client_fraction': 1.0,  # All clients participate each round
    'batch_size': 32,

    # Optimization
    'lr': 0.002,
    'weight_decay': 0.01,

    # Prompt Learning
    'n_ctx': 4,
    'theta': 0.3,

    # Reproducibility
    'seed': 42,
}
```

### Ablation Studies

#### 1. Heterogeneity Ablation

```python
for alpha in [0.1, 0.3, 0.5, 1.0, 2.0]:
    run_experiment(dirichlet_alpha=alpha, ...)
```

#### 2. Theta (θ) Ablation

```python
for theta in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
    run_experiment(theta=theta, ...)
```

#### 3. Number of Clients Ablation

```python
for num_clients in [3, 6, 9, 12]:
    run_experiment(virtual_clients_per_dataset=num_clients//3, ...)
```

#### 4. Context Length Ablation

```python
for n_ctx in [2, 4, 8, 16]:
    run_experiment(n_ctx=n_ctx, ...)
```

---

## 4. Baseline Methods

### Implemented Baselines

```python
BASELINES = {
    # Standard FL
    'fedavg': FedAvgMedCLIP,      # McMahan et al., 2017
    'fedprox': FedProxMedCLIP,    # Li et al., 2020

    # Prompt FL (adapted)
    'promptfl': PromptFL,          # Guo et al., 2022

    # Local training
    'local': LocalOnlyMedCLIP,     # No aggregation

    # Ours
    'medpromptfolio': MedPromptFolio,
}
```

### Baseline Descriptions

| Method | Description | What's Shared |
|--------|-------------|---------------|
| **FedAvg** | Standard federated averaging | All prompt params |
| **FedProx** | FedAvg + proximal regularization | All prompt params |
| **PromptFL** | Federated prompt tuning | Prompt only |
| **Local** | Train on local data only | Nothing |
| **MedPromptFolio** | Portfolio of global+local prompts | Global prompt only |

---

## 5. Evaluation Protocol

### Per-Round Evaluation

```python
for round in range(num_rounds):
    # Train on all clients
    for client in clients:
        train_local(client)

    # Aggregate
    aggregate_global_prompts()

    # Evaluate
    metrics = {}
    for client in clients:
        client_metrics = evaluate(client, test_loader)
        metrics[client.name] = client_metrics

    # Aggregate metrics
    avg_auc = mean([m['auc'] for m in metrics.values()])
    per_class_auc = {cls: mean([m['per_class'][cls] for m in metrics.values()])
                     for cls in classes}

    log_metrics(round, avg_auc, per_class_auc)
```

### Final Evaluation

```python
# After training completes
final_metrics = {
    'best_auc': max(history['test_auc']),
    'best_round': argmax(history['test_auc']),
    'final_auc': history['test_auc'][-1],
    'convergence_round': first_round_within_1%_of_best,
    'per_class_auc': {...},
}
```

---

## 6. Metrics

### Primary Metric: AUC-ROC

```python
from sklearn.metrics import roc_auc_score

def compute_auc(y_true, y_pred, average='macro'):
    """
    Compute AUC for multi-label classification.

    Args:
        y_true: Ground truth (N, C)
        y_pred: Predictions/logits (N, C)
        average: 'macro' (default), 'micro', or 'weighted'
    """
    aucs = []
    for c in range(y_true.shape[1]):
        if len(np.unique(y_true[:, c])) > 1:  # Need both classes
            auc = roc_auc_score(y_true[:, c], y_pred[:, c])
            aucs.append(auc)
    return np.mean(aucs)
```

### Secondary Metrics

```python
metrics = {
    'auc': roc_auc_score(y_true, y_pred, average='macro'),
    'accuracy': accuracy_score(y_true, y_pred > 0.5),
    'f1': f1_score(y_true, y_pred > 0.5, average='macro'),
    'sensitivity': recall_score(y_true, y_pred > 0.5, average='macro'),
    'specificity': specificity(y_true, y_pred > 0.5),
}
```

---

## 7. Computational Resources

### Hardware Requirements

```
Minimum:
- GPU: 8GB VRAM (RTX 3070 or equivalent)
- RAM: 32GB
- Storage: 100GB SSD

Recommended:
- GPU: 24GB VRAM (RTX 3090/4090 or A100)
- RAM: 64GB
- Storage: 256GB NVMe SSD
```

### Training Time Estimates

| Configuration | GPU | Time per Round | Total (50 rounds) |
|---------------|-----|----------------|-------------------|
| 9 clients, 2K samples | RTX 3090 | ~3 min | ~2.5 hours |
| 9 clients, 5K samples | RTX 3090 | ~7 min | ~6 hours |
| 12 clients, 5K samples | A100 | ~5 min | ~4 hours |

### Memory Usage

```
Per-client batch (32 images):
- Image features: 32 × 768 × 4 bytes = 98 KB
- Text features: 5 × 768 × 4 bytes = 15 KB
- Gradients: ~200 KB
- Total: ~1 GB per client (with MedCLIP backbone)
```

---

## 8. Output Structure

### Experiment Directory

```
experiments/full_exp_20260121_030159/
├── config.json                      # Experiment configuration
├── summary.json                     # Final results summary
├── data_heterogeneity_stats.txt    # Heterogeneity statistics
├── data_heterogeneity_label_heatmap.png
├── data_heterogeneity_sample_distribution.png
├── data_heterogeneity_js_divergence.png
│
├── medpromptfolio/
│   ├── results.json                # Per-round metrics
│   ├── final_results.json         # Final summary
│   └── best_model.pt              # Best checkpoint
│
├── fedavg/
│   └── ...
│
├── fedprox/
│   └── ...
│
└── local/
    └── ...
```

### Results JSON Format

```json
{
  "rounds": [0, 1, 2, ...],
  "train_loss": [0.72, 0.68, 0.65, ...],
  "train_auc": [0.75, 0.78, 0.80, ...],
  "test_auc": [0.73, 0.76, 0.78, ...],
  "per_client_auc": [[...], [...], ...],
  "best_auc": 0.82,
  "best_round": 35,
  "elapsed_time": 7200.5
}
```

---

## 9. Running Experiments

### Quick Test (5 minutes)

```bash
python -c "
from medpromptfolio import FederatedTrainer, MedPromptFolio, FederatedChestXrayDataset

fed_dataset = FederatedChestXrayDataset(
    data_roots={'nih_chestxray': 'data/nih_chestxray', ...},
    use_virtual_clients=True,
    virtual_clients_per_dataset=2,
    max_samples_per_client=200,
)

model = MedPromptFolio(...)
trainer = FederatedTrainer(model, fed_dataset, num_rounds=3)
trainer.train()
"
```

### Full Experiment

```bash
# Run in background
nohup python run_full_experiment.py > experiment.log 2>&1 &

# Monitor
tail -f experiment.log
```

---

## Quick Links

- [Project Overview](./01_PROJECT_OVERVIEW.md)
- [Dataset Documentation](./02_DATASETS.md)
- [Training Methodology](./03_TRAINING_METHODOLOGY.md)
- [Novel Contributions](./04_NOVEL_CONTRIBUTIONS.md)
- [Code Documentation](./06_CODE_DOCUMENTATION.md)
