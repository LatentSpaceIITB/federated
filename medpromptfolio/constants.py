"""
Clinical constants and prompt templates for MedPromptFolio.

Contains:
- CheXpert competition tasks (5 pathologies)
- Clinical prompt templates for each pathology
- Medical imaging normalization constants
- Dataset configuration
"""

import torch

# MedCLIP backbone types
BERT_TYPE = 'emilyalsentzer/Bio_ClinicalBERT'
VIT_TYPE = 'microsoft/swin-tiny-patch4-window7-224'

# Image preprocessing constants (from MedCLIP)
IMG_SIZE = 224
IMG_MEAN = 0.5862785803043838
IMG_STD = 0.27950088968644304

# CheXpert competition tasks (5 pathologies for benchmark)
CHEXPERT_COMPETITION_TASKS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]

# All CheXpert tasks (14 pathologies)
CHEXPERT_ALL_TASKS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Lung Opacity",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

# Clinical prompt templates for CheXpert tasks
# Format: severity + subtype + location combinations
CHEXPERT_CLASS_PROMPTS = {
    "Atelectasis": {
        "severity": ["", "mild", "minimal"],
        "subtype": [
            "subsegmental atelectasis",
            "linear atelectasis",
            "trace atelectasis",
            "bibasilar atelectasis",
            "retrocardiac atelectasis",
            "bandlike atelectasis",
            "residual atelectasis",
        ],
        "location": [
            "at the mid lung zone",
            "at the upper lung zone",
            "at the right lung zone",
            "at the left lung zone",
            "at the lung bases",
            "at the right lung base",
            "at the left lung base",
            "at the bilateral lung bases",
            "at the left lower lobe",
            "at the right lower lobe",
        ],
    },
    "Cardiomegaly": {
        "severity": [""],
        "subtype": [
            "cardiac silhouette size is upper limits of normal",
            "cardiomegaly which is unchanged",
            "mildly prominent cardiac silhouette",
            "portable view of the chest demonstrates stable cardiomegaly",
            "portable view of the chest demonstrates mild cardiomegaly",
            "persistent severe cardiomegaly",
            "heart size is borderline enlarged",
            "cardiomegaly unchanged",
            "heart size is at the upper limits of normal",
            "redemonstration of cardiomegaly",
            "cardiac silhouette size is mildly enlarged",
            "mildly enlarged cardiac silhouette",
            "heart size remains at mildly enlarged",
            "persistent cardiomegaly with prominent upper lobe vessels",
        ],
        "location": [""],
    },
    "Consolidation": {
        "severity": ["", "increased", "improved", "appearance of"],
        "subtype": [
            "bilateral consolidation",
            "reticular consolidation",
            "retrocardiac consolidation",
            "patchy consolidation",
            "airspace consolidation",
            "partial consolidation",
        ],
        "location": [
            "at the lower lung zone",
            "at the upper lung zone",
            "at the left lower lobe",
            "at the right lower lobe",
            "at the left upper lobe",
            "at the right upper lobe",
            "at the right lung base",
            "at the left lung base",
        ],
    },
    "Edema": {
        "severity": [
            "",
            "mild",
            "improvement in",
            "persistent",
            "moderate",
            "decreased",
        ],
        "subtype": [
            "pulmonary edema",
            "trace interstitial edema",
            "pulmonary interstitial edema",
        ],
        "location": [""],
    },
    "Pleural Effusion": {
        "severity": ["", "small", "stable", "large", "decreased", "increased"],
        "location": ["left", "right", "bilateral"],
        "subtype": [
            "pleural effusion",
            "subpulmonic pleural effusion",
            "bilateral pleural effusion",
            "loculated pleural effusion",
        ],
    },
    # Additional pathologies (full 14-class)
    "No Finding": {
        "severity": [""],
        "subtype": ["no acute cardiopulmonary abnormality", "lungs are clear", "normal chest radiograph"],
        "location": [""],
    },
    "Pneumonia": {
        "severity": ["", "suspected", "possible", "likely"],
        "subtype": ["pneumonia", "infectious process", "infiltrate suggesting pneumonia"],
        "location": ["in the lower lobes", "in the right lung", "in the left lung", "bilateral"],
    },
    "Pneumothorax": {
        "severity": ["", "small", "moderate", "large", "tension"],
        "subtype": ["pneumothorax", "collapsed lung"],
        "location": ["on the right", "on the left", "bilateral"],
    },
}

# Medical prompt templates for zero-shot classification
# These are templates that work well with Bio_ClinicalBERT
MEDICAL_PROMPT_TEMPLATES = [
    "chest x-ray showing {classname}",
    "radiograph with evidence of {classname}",
    "findings consistent with {classname}",
    "impression: {classname}",
    "x-ray demonstrating {classname}",
    "chest radiograph with {classname}",
    "frontal chest x-ray showing {classname}",
    "pa and lateral chest x-ray with {classname}",
]

# Negative prompts (for binary classification)
NEGATIVE_PROMPTS = [
    "no evidence of {classname}",
    "no {classname}",
    "chest x-ray without {classname}",
    "normal chest x-ray",
    "unremarkable chest radiograph",
]

# Dataset-specific configurations
DATASET_CONFIGS = {
    "chexpert": {
        "name": "CheXpert",
        "hospital": "Stanford",
        "num_images": 224316,
        "tasks": CHEXPERT_COMPETITION_TASKS,
        "uncertainty_policy": "ones",  # How to handle uncertain labels: 'zeros', 'ones', 'ignore'
    },
    "mimic_cxr": {
        "name": "MIMIC-CXR",
        "hospital": "Beth Israel Deaconess Medical Center",
        "num_images": 377110,
        "tasks": CHEXPERT_COMPETITION_TASKS,
        "uncertainty_policy": "ones",
    },
    "nih_chestxray": {
        "name": "NIH ChestX-ray14",
        "hospital": "NIH Clinical Center",
        "num_images": 112120,
        "tasks": CHEXPERT_COMPETITION_TASKS,
        "uncertainty_policy": "binary",
    },
    "vindr_cxr": {
        "name": "VinDr-CXR",
        "hospital": "Vingroup Big Data Institute",
        "num_images": 18000,
        "tasks": CHEXPERT_COMPETITION_TASKS,
        "uncertainty_policy": "binary",
    },
}

# Federated learning default configurations
FL_DEFAULT_CONFIG = {
    "num_clients": 4,
    "num_rounds": 50,
    "local_epochs": 1,
    "batch_size": 32,
    "learning_rate": 0.002,
    "frac_clients": 1.0,  # Fraction of clients participating per round
    "theta": 0.3,  # Portfolio mixing coefficient (0=global, 1=local)
    "num_prompts": 2,  # Number of prompts (1 global + 1 local for MedPromptFolio)
    "n_ctx": 8,  # Number of context tokens in prompt
}

# Mapping of dataset names to client IDs
CLIENT_DATASET_MAP = {
    0: "chexpert",
    1: "mimic_cxr",
    2: "nih_chestxray",
    3: "vindr_cxr",
}


def get_class_prompts(classname: str, num_prompts: int = 10) -> list:
    """
    Generate clinical prompts for a given class.

    Args:
        classname: Name of the pathology
        num_prompts: Number of prompts to generate

    Returns:
        List of clinical prompt strings
    """
    if classname not in CHEXPERT_CLASS_PROMPTS:
        # Fallback to generic medical prompts
        return [t.format(classname=classname.lower()) for t in MEDICAL_PROMPT_TEMPLATES[:num_prompts]]

    prompts = []
    class_info = CHEXPERT_CLASS_PROMPTS[classname]

    for severity in class_info.get("severity", [""]):
        for subtype in class_info.get("subtype", [classname.lower()]):
            for location in class_info.get("location", [""]):
                # Construct prompt
                parts = [p for p in [severity, subtype, location] if p]
                prompt = " ".join(parts).strip()
                if prompt:
                    prompts.append(prompt)
                    if len(prompts) >= num_prompts:
                        return prompts

    return prompts if prompts else [classname.lower()]


def get_negative_prompts(classname: str, num_prompts: int = 5) -> list:
    """
    Generate negative prompts for a given class (useful for binary classification).
    """
    return [t.format(classname=classname.lower()) for t in NEGATIVE_PROMPTS[:num_prompts]]
