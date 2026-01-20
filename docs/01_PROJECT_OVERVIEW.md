# MedPromptFolio: Federated Prompt Learning for Medical Vision-Language Models

## Project Overview

**MedPromptFolio** extends PromptFolio (NeurIPS 2024) to medical imaging, enabling privacy-preserving collaboration across hospitals using federated learning with vision-language models.

---

## 1. Problem Statement

### The Challenge

Healthcare institutions (hospitals, clinics, research centers) possess vast amounts of medical imaging data that could revolutionize AI-assisted diagnosis. However:

1. **Privacy Regulations**: HIPAA, GDPR, and other regulations prohibit sharing patient data across institutions
2. **Data Heterogeneity**: Different hospitals have:
   - Different patient populations (demographics, disease prevalence)
   - Different imaging equipment (scanners, protocols)
   - Different annotation practices (radiologist expertise, labeling criteria)
3. **Limited Local Data**: Individual hospitals may lack sufficient data for robust model training
4. **Domain Shift**: Models trained on one hospital's data often fail on another's

### Real-World Scenario

```
Hospital A (Stanford)     Hospital B (MIT)        Hospital C (NIH)
┌─────────────────┐      ┌─────────────────┐     ┌─────────────────┐
│ 50K X-rays      │      │ 80K X-rays      │     │ 30K X-rays      │
│ High pneumonia  │      │ High cardiac    │     │ Balanced        │
│ Siemens scanner │      │ GE scanner      │     │ Philips scanner │
│ CANNOT SHARE    │      │ CANNOT SHARE    │     │ CANNOT SHARE    │
└─────────────────┘      └─────────────────┘     └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   FEDERATED LEARNING    │
                    │   Share model updates,  │
                    │   NOT patient data      │
                    └─────────────────────────┘
```

---

## 2. Our Solution: MedPromptFolio

### Core Idea

Instead of training full models (expensive, prone to overfitting), we train **soft prompts** - small learnable vectors that guide a frozen pre-trained medical vision-language model (MedCLIP).

### Key Innovation: Portfolio Strategy

We maintain TWO types of prompts:

1. **Global Prompt (P_G)**: Shared across all hospitals, captures universal medical knowledge
2. **Local Prompt (P_L)**: Specific to each hospital, captures local patterns

The final prediction uses a **portfolio mix**:
```
P_final = (1 - θ) × P_G + θ × P_L
```

Where θ (theta) controls the balance:
- θ = 0: Pure global (ignores local patterns)
- θ = 1: Pure local (ignores shared knowledge)
- θ = 0.3: Our default (70% global, 30% local)

### Why This Works for Medical Imaging

| Challenge | How MedPromptFolio Addresses It |
|-----------|--------------------------------|
| Privacy | Only prompt vectors shared, not images or patient data |
| Heterogeneity | Local prompts adapt to each hospital's distribution |
| Limited data | Pre-trained MedCLIP provides strong initialization |
| Efficiency | Only ~4K parameters trained vs millions in full fine-tuning |

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MedPromptFolio                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ Chest X-ray  │    │   MedCLIP    │    │   Learnable  │      │
│  │    Image     │───▶│ Vision Enc.  │    │   Prompts    │      │
│  │  (224×224)   │    │ (Swin-ViT)   │    │              │      │
│  └──────────────┘    └──────┬───────┘    │  P_G: [4×768]│      │
│                             │            │  P_L: [4×768]│      │
│                             │            └──────┬───────┘      │
│                             │                   │              │
│                             ▼                   ▼              │
│                    ┌────────────────────────────────┐          │
│                    │     Text Encoder (BERT)        │          │
│                    │  "A chest X-ray showing [CLS]" │          │
│                    │  + Learnable Context Tokens    │          │
│                    └────────────────┬───────────────┘          │
│                                     │                          │
│                                     ▼                          │
│                    ┌────────────────────────────────┐          │
│                    │    Cosine Similarity Scores    │          │
│                    │  [Atelectasis, Cardiomegaly,   │          │
│                    │   Consolidation, Edema,        │          │
│                    │   Pleural Effusion]            │          │
│                    └────────────────────────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Target Application

### Clinical Use Case: Multi-Hospital Chest X-ray Diagnosis

**Task**: Detect 5 pathologies from the CheXpert competition:
1. Atelectasis (lung collapse)
2. Cardiomegaly (enlarged heart)
3. Consolidation (lung infection/pneumonia)
4. Edema (fluid in lungs)
5. Pleural Effusion (fluid around lungs)

**Why These 5?**
- Clinically significant conditions
- Present across all major chest X-ray datasets
- Standard benchmark in medical imaging research

---

## 5. Expected Impact

### For Healthcare
- Enable AI collaboration without compromising patient privacy
- Improve diagnostic accuracy through diverse training data
- Reduce health disparities by including underrepresented populations

### For Research
- First federated prompt learning framework for medical VLMs
- New benchmark for heterogeneous medical FL
- Theoretical analysis of optimal mixing coefficients

### Target Venues
1. **ICML 2025** - Theory + healthcare application
2. **NeurIPS 2025** - Direct follow-up to PromptFolio
3. **MICCAI 2025** - Medical imaging focus

---

## Quick Links

- [Dataset Documentation](./02_DATASETS.md)
- [Training Methodology](./03_TRAINING_METHODOLOGY.md)
- [Novel Contributions](./04_NOVEL_CONTRIBUTIONS.md)
- [Experimental Setup](./05_EXPERIMENTAL_SETUP.md)
- [Code Documentation](./06_CODE_DOCUMENTATION.md)
