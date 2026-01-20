# MedPromptFolio: Federated Prompt Learning for Medical Vision-Language Models

Extends [PromptFolio (NeurIPS 2024)](https://github.com/PanBikang/PromptFolio) to medical imaging with MedCLIP for privacy-preserving hospital collaboration.

## Overview

MedPromptFolio is a federated prompt learning method for medical vision-language models that:

1. **Preserves privacy**: Only prompt parameters are shared, not medical images
2. **Handles heterogeneity**: Portfolio strategy adapts to scanner/annotation/label shifts
3. **Leverages clinical knowledge**: Uses Bio_ClinicalBERT text encoder with medical prompts
4. **Supports multi-label**: Handles multiple pathology predictions (BCEWithLogitsLoss)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MedPromptFolio                          │
│  ┌─────────────┐    ┌─────────────────────────────────┐    │
│  │   MedCLIP   │    │    Learnable Prompts             │    │
│  │  (Frozen)   │    │  ┌─────────┐  ┌─────────┐       │    │
│  │             │    │  │ Global  │  │  Local  │       │    │
│  │ Swin-ViT    │    │  │ Prompt  │  │ Prompt  │       │    │
│  │    +        │    │  └────┬────┘  └────┬────┘       │    │
│  │ BioClinBERT │    │       │   Mix θ   │             │    │
│  │             │    │       └─────┬─────┘             │    │
│  └─────────────┘    │             │                   │    │
│         ↓           │             ↓                   │    │
│  ┌─────────────┐    │    ┌─────────────┐             │    │
│  │   Image     │    │    │    Text     │             │    │
│  │  Embedding  │────┼───►│  Embedding  │──► Logits   │    │
│  └─────────────┘    │    └─────────────┘             │    │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. Portfolio Strategy
- **Global prompt** (θ=0): Aggregated across all hospital clients
- **Local prompt** (θ=1): Client-specific, not aggregated
- **Mixed prompt**: P_mixed = (1-θ)·P_global + θ·P_local

### 2. Medical-Specific Heterogeneity
Medical imaging exhibits extreme heterogeneity:
- **Scanner shifts**: Different imaging equipment across hospitals
- **Annotation shifts**: Varying radiologist labeling practices
- **Label shifts**: Different disease prevalence by region/demographics

### 3. Clinical Prompt Templates
Uses MedCLIP's clinical prompt templates:
```python
# Example prompts for Atelectasis
"subsegmental atelectasis at the mid lung zone"
"linear atelectasis at the lung bases"
"trace atelectasis at the bilateral lung bases"
```

## Installation

```bash
# Clone repository
git clone <repo-url>
cd medpromptfolio

# Install dependencies
pip install -r requirements.txt

# Verify MedCLIP checkpoint
ls /home/pinak/MedClip/proj1/pretrained/medclip-vit/
```

## Quick Start

### Test with Synthetic Data
```bash
cd medpromptfolio
python scripts/train.py --method medpromptfolio --use_synthetic --num_rounds 10
```

### Run All Baselines
```bash
bash scripts/run_experiments.sh baselines
```

### Train with Real Data
```bash
python scripts/train.py \
    --method medpromptfolio \
    --chexpert_root /path/to/chexpert \
    --mimic_root /path/to/mimic-cxr \
    --nih_root /path/to/nih-chestxray14 \
    --vindr_root /path/to/vindr-cxr \
    --num_rounds 50
```

## Datasets

| Dataset | Hospital | Images | Role |
|---------|----------|--------|------|
| CheXpert | Stanford | 224K | Client 1 |
| MIMIC-CXR | Beth Israel | 377K | Client 2 |
| NIH ChestX-ray14 | NIH | 112K | Client 3 |
| VinDr-CXR | Vietnam | 18K | Client 4 |

**Common labels** (5 CheXpert competition tasks):
- Atelectasis
- Cardiomegaly
- Consolidation
- Edema
- Pleural Effusion

## Methods

### MedPromptFolio (Proposed)
```python
from medpromptfolio import MedPromptFolio

model = MedPromptFolio(
    classnames=["Atelectasis", "Cardiomegaly", ...],
    medclip_checkpoint="/path/to/medclip-vit",
    n_ctx=8,      # Number of context tokens
    theta=0.3,    # Mixing coefficient
)
```

### Baselines
- **FedAvg**: Standard federated averaging
- **FedProx**: FedAvg with proximal regularization
- **Local**: No federation (independent training)
- **PromptFL**: PromptFolio without portfolio strategy

## Configuration

```yaml
# configs/default.yaml
model:
  n_ctx: 8
  theta: 0.3

federated:
  num_clients: 4
  num_rounds: 50
  local_epochs: 1

training:
  lr: 0.002
  batch_size: 32
```

## Project Structure

```
medpromptfolio/
├── __init__.py
├── constants.py           # Clinical prompts, config
├── medclip_prompt.py      # MedCLIP integration
├── med_prompt_folio.py    # Core algorithm
├── chest_xray_datasets.py # Dataset loaders
├── federated_trainer.py   # FL orchestration
├── utils.py               # Utilities
├── configs/
│   └── default.yaml
├── scripts/
│   ├── train.py
│   └── run_experiments.sh
└── requirements.txt
```

## Theoretical Foundation

From PromptFolio, the optimal mixing coefficient is:
```
θ* = (1 - SNR_local) / (1 - SNR_global + 1 - SNR_local)
```

For medical imaging with high heterogeneity:
- Expected SNR_global ≈ 0.3-0.5 (more noise in aggregated signal)
- Expected SNR_local ≈ 0.6-0.8 (cleaner local signal)
- **Optimal θ* ≈ 0.3-0.4** (vs 0.2 for natural images)

## Expected Results

| Method | Avg AUC | Improvement |
|--------|---------|-------------|
| PromptFL | 0.79 | baseline |
| PromptFolio | 0.82 | +3% |
| **MedPromptFolio** | **0.85** | **+6%** |

## Citation

```bibtex
@article{medpromptfolio2024,
  title={MedPromptFolio: Federated Prompt Learning for Medical Vision-Language Models},
  author={...},
  journal={arXiv preprint},
  year={2024}
}
```

## Acknowledgments

- [PromptFolio](https://github.com/PanBikang/PromptFolio) - Base codebase
- [MedCLIP](https://github.com/RyanWangZf/MedCLIP) - Medical VLM backbone
- [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) - Dataset

## License

MIT License
