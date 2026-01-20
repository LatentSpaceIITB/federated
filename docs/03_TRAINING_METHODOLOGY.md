# Training Methodology

## Overview

MedPromptFolio uses **federated learning** combined with **prompt learning** to train a medical image classifier across multiple hospitals without sharing patient data.

---

## 1. Federated Learning Framework

### Communication Rounds

```
Round t:
┌─────────────────────────────────────────────────────────────────┐
│                         SERVER                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Global Prompt P_G^(t)                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│            │                    │                    │          │
│            ▼                    ▼                    ▼          │
│    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│    │  Client 1    │    │  Client 2    │    │  Client K    │   │
│    │  Hospital A  │    │  Hospital B  │    │  Hospital C  │   │
│    │              │    │              │    │              │   │
│    │ Local Data   │    │ Local Data   │    │ Local Data   │   │
│    │ Local Train  │    │ Local Train  │    │ Local Train  │   │
│    │              │    │              │    │              │   │
│    │ Update:      │    │ Update:      │    │ Update:      │   │
│    │ ΔP_G^1, P_L^1│    │ ΔP_G^2, P_L^2│    │ ΔP_G^K, P_L^K│   │
│    └──────┬───────┘    └──────┬───────┘    └──────┬───────┘   │
│           │                   │                   │            │
│           └───────────────────┼───────────────────┘            │
│                               ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │         Aggregate: P_G^(t+1) = Σ (n_k/n) × P_G^k        │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Algorithm: MedPromptFolio

```python
# Server-side
Initialize global prompt P_G randomly
For round t = 1 to T:
    # Broadcast
    Send P_G to all clients

    # Local training (parallel)
    For each client k:
        Download P_G
        Initialize/update local prompt P_L^k
        For epoch e = 1 to E:
            For batch (x, y) in local_data:
                # Portfolio mixing
                P_mixed = (1-θ) × P_G + θ × P_L^k

                # Forward pass with MedCLIP
                logits = MedCLIP(x, P_mixed)
                loss = BCEWithLogits(logits, y)

                # Update prompts (P_G and P_L^k)
                loss.backward()
                optimizer.step()

        Upload updated P_G^k to server

    # Aggregation (weighted by data size)
    P_G = Σ (n_k / Σn_j) × P_G^k
```

---

## 2. Prompt Learning Details

### What Are Soft Prompts?

Instead of hand-crafted text prompts like "A chest X-ray showing pneumonia", we learn **continuous vectors** that are prepended to the text:

```
Traditional: "A chest X-ray showing [CLASS]"
                    ↓
Soft Prompt: [v1][v2][v3][v4] + "showing [CLASS]"
             ↑___learnable___↑
```

### Prompt Structure

```python
# Context tokens (learnable)
ctx_vectors = nn.Parameter(torch.randn(n_ctx, ctx_dim))
# n_ctx = 4 (number of context tokens)
# ctx_dim = 768 (BERT hidden dimension)

# For each class, construct prompt:
# [CTX1][CTX2][CTX3][CTX4] "showing" [CLASS_NAME]
```

### Portfolio Prompts

We maintain two sets of prompts:

```python
class MedPromptFolio:
    def __init__(self):
        # Global prompt - shared, aggregated
        self.global_ctx = nn.Parameter(torch.randn(n_ctx, 768))

        # Local prompt - client-specific, not shared
        self.local_ctx = nn.Parameter(torch.randn(n_ctx, 768))

    def forward(self, images, theta=0.3):
        # Mix prompts
        mixed_ctx = (1 - theta) * self.global_ctx + theta * self.local_ctx

        # Get text features
        text_features = self.text_encoder(mixed_ctx, class_names)

        # Get image features
        image_features = self.vision_encoder(images)

        # Compute similarity
        logits = image_features @ text_features.T * self.logit_scale

        return logits
```

---

## 3. MedCLIP Backbone

### Architecture

```
MedCLIP
├── Vision Encoder: Swin Transformer (Tiny)
│   ├── Input: 224 × 224 × 3
│   ├── Patch size: 4 × 4
│   ├── Window size: 7
│   └── Output: 768-dim feature vector
│
├── Text Encoder: Bio_ClinicalBERT
│   ├── Pre-trained on clinical notes
│   ├── Max sequence length: 512
│   └── Output: 768-dim feature vector
│
└── Projection: Shared latent space (768-dim)
```

### Pre-trained Weights

Located at: `/home/pinak/MedClip/proj1/pretrained/medclip-vit/`

```
medclip-vit/
├── pytorch_model.bin     # MedCLIP weights
├── config.json
└── tokenizer/
```

### Why MedCLIP?

| Feature | CLIP | MedCLIP |
|---------|------|---------|
| Training data | Natural images | Medical images + reports |
| Text encoder | GPT-2 | Bio_ClinicalBERT |
| Domain | General | Medical |
| Chest X-ray AUC | ~0.65 | ~0.80 |

---

## 4. Loss Function

### Multi-Label Binary Cross Entropy

Since each image can have multiple pathologies:

```python
loss = nn.BCEWithLogitsLoss()(logits, labels)

# For each class independently:
# L = -[y × log(σ(z)) + (1-y) × log(1-σ(z))]
# where σ is sigmoid, z is logit, y is label
```

### Class Weighting (Optional)

To handle class imbalance:

```python
# Compute positive weight for each class
pos_weight = (num_negative / num_positive) for each class
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

---

## 5. Optimization

### Optimizer: AdamW

```python
optimizer = torch.optim.AdamW(
    prompt_parameters,
    lr=0.002,
    weight_decay=0.01,
    betas=(0.9, 0.999)
)
```

### Learning Rate Schedule

```python
# Cosine annealing with warm restarts
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,  # Restart every 10 rounds
    eta_min=1e-6
)
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    logits = model(images)
    loss = criterion(logits, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## 6. Aggregation Strategies

### FedAvg (Baseline)

```python
# Weighted average by data size
global_prompt = sum(n_k * prompt_k for k in clients) / sum(n_k)
```

### FedProx (Baseline)

```python
# Add proximal term to local loss
loss = task_loss + (mu/2) * ||prompt - global_prompt||^2
```

### MedPromptFolio (Ours)

```python
# Only aggregate global prompts
# Local prompts stay with each client
global_prompt = aggregate([client.global_prompt for client in clients])
# client.local_prompt unchanged
```

---

## 7. Evaluation Metrics

### Primary: Area Under ROC Curve (AUC)

```python
from sklearn.metrics import roc_auc_score

# Per-class AUC
for i, class_name in enumerate(classes):
    auc = roc_auc_score(y_true[:, i], y_pred[:, i])

# Macro-average AUC
macro_auc = np.mean(per_class_aucs)
```

### Why AUC?

1. **Class imbalance**: Medical datasets are highly imbalanced
2. **Threshold-independent**: Doesn't require choosing a decision threshold
3. **Standard metric**: Used in CheXpert competition and medical imaging benchmarks

### Secondary Metrics

- **Sensitivity (Recall)**: True positive rate
- **Specificity**: True negative rate
- **F1 Score**: Harmonic mean of precision and recall

---

## 8. Hyperparameters

### Default Configuration

```yaml
# Federated Learning
num_rounds: 50
local_epochs: 2
client_fraction: 1.0
batch_size: 32

# Prompt Learning
n_ctx: 4                 # Number of context tokens
ctx_init: "random"       # Initialization method
theta: 0.3              # Portfolio mixing coefficient

# Optimization
lr: 0.002
weight_decay: 0.01
warmup_epochs: 5

# Model
backbone: "medclip-vit"
freeze_backbone: true    # Only train prompts
```

### Hyperparameter Sensitivity

| Parameter | Range | Impact |
|-----------|-------|--------|
| θ (theta) | 0.1-0.5 | Higher = more local adaptation |
| n_ctx | 2-16 | More tokens = more capacity |
| lr | 0.0001-0.01 | Standard tuning |
| local_epochs | 1-5 | More = better local fit, risk overfitting |

---

## Quick Links

- [Project Overview](./01_PROJECT_OVERVIEW.md)
- [Dataset Documentation](./02_DATASETS.md)
- [Novel Contributions](./04_NOVEL_CONTRIBUTIONS.md)
- [Experimental Setup](./05_EXPERIMENTAL_SETUP.md)
