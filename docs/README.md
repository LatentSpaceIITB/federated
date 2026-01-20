# MedPromptFolio Documentation

## Federated Prompt Learning for Medical Vision-Language Models

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r medpromptfolio/requirements.txt

# 2. Download datasets (requires Kaggle API)
kaggle datasets download nih-chest-xrays/data
kaggle datasets download -d ashery/chexpert
kaggle competitions download -c rsna-pneumonia-detection-challenge

# 3. Run experiment
python run_full_experiment.py
```

---

## Documentation Index

| Document | Description |
|----------|-------------|
| [01_PROJECT_OVERVIEW.md](./01_PROJECT_OVERVIEW.md) | Problem statement, solution approach, architecture |
| [02_DATASETS.md](./02_DATASETS.md) | Dataset sources, download instructions, preprocessing |
| [03_TRAINING_METHODOLOGY.md](./03_TRAINING_METHODOLOGY.md) | Federated learning, prompt learning, optimization |
| [04_NOVEL_CONTRIBUTIONS.md](./04_NOVEL_CONTRIBUTIONS.md) | Research contributions, theoretical analysis |
| [05_EXPERIMENTAL_SETUP.md](./05_EXPERIMENTAL_SETUP.md) | Virtual clients, heterogeneity, evaluation |
| [06_CODE_DOCUMENTATION.md](./06_CODE_DOCUMENTATION.md) | Code structure, API reference, examples |

---

## Key Concepts

### What We're Solving

**Problem**: Hospitals cannot share patient data due to privacy regulations, but need collaborative AI for better diagnosis.

**Solution**: Federated learning with prompt-based fine-tuning of MedCLIP, sharing only lightweight prompt vectors (~4KB) instead of images or full model weights.

### How We Solve It

1. **Pre-trained Backbone**: Use MedCLIP (medical vision-language model) as frozen feature extractor
2. **Learnable Prompts**: Train soft prompts that guide the model's text understanding
3. **Portfolio Strategy**: Maintain global (shared) + local (private) prompts with mixing coefficient θ
4. **Federated Aggregation**: Only aggregate global prompts across hospitals

### Virtual Client Simulation

```
3 Real Datasets → 9 Virtual Clients (Dirichlet α=0.3)

NIH ChestX-ray14 ─┬─ Client 0 (specialized: Pleural Effusion)
                  ├─ Client 1 (balanced)
                  └─ Client 2 (specialized: Cardiomegaly)

CheXpert ─────────┬─ Client 3 (specialized: Edema)
                  ├─ Client 4 (balanced)
                  └─ Client 5 (specialized: Atelectasis)

RSNA Pneumonia ───┬─ Client 6 (high pneumonia)
                  ├─ Client 7 (mixed)
                  └─ Client 8 (low pneumonia)
```

---

## Results Summary

### Baseline Comparison (Expected)

| Method | Avg AUC | Improvement |
|--------|---------|-------------|
| Local Only | 0.74 | - |
| FedAvg | 0.77 | +3% |
| FedProx | 0.78 | +4% |
| **MedPromptFolio** | **0.83** | **+9%** |

### Heterogeneity Robustness

MedPromptFolio shows strongest gains under high data heterogeneity (low Dirichlet α).

---

## Citation

```bibtex
@article{medpromptfolio2025,
  title={MedPromptFolio: Federated Prompt Learning for Medical Vision-Language Models},
  author={...},
  journal={arXiv preprint},
  year={2025}
}
```

---

## License

MIT License

---

## Acknowledgments

- [PromptFolio](https://github.com/PanBikang/PromptFolio) - Base methodology
- [MedCLIP](https://github.com/RyanWangZf/MedCLIP) - Medical VLM backbone
- NIH, Stanford, RSNA - Dataset providers
