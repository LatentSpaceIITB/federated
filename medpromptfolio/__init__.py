"""
MedPromptFolio: Federated Prompt Learning for Medical Vision-Language Models

Extends PromptFolio (NeurIPS 2024) to medical imaging with MedCLIP.
"""

__version__ = "0.1.0"
__author__ = "MedPromptFolio Team"

from .constants import (
    CHEXPERT_COMPETITION_TASKS,
    CHEXPERT_CLASS_PROMPTS,
    MEDICAL_PROMPT_TEMPLATES,
    IMG_SIZE,
    IMG_MEAN,
    IMG_STD,
    FL_DEFAULT_CONFIG,
    get_class_prompts,
)
from .medclip_prompt import (
    MedCLIPPromptLearner,
    MedCLIPTextEncoder,
    MedCLIPVisionEncoder,
    CustomMedCLIP,
    PromptOnlyMedCLIP,
)
from .med_prompt_folio import (
    MedPromptFolio,
    MedPromptFolioTrainer,
    FedAvgMedCLIP,
    FedProxMedCLIP,
    LocalOnlyMedCLIP,
)
from .federated_trainer import FederatedTrainer, run_experiment
from .chest_xray_datasets import (
    CheXpertDataset,
    MIMICCXRDataset,
    NIHChestXrayDataset,
    VinDrCXRDataset,
    RSNAPneumoniaDataset,
    SyntheticChestXrayDataset,
    FederatedChestXrayDataset,
    get_dataset,
)
from .virtual_clients import (
    VirtualClientSplitter,
    VirtualClientDataset,
    FederatedVirtualClients,
    create_federated_virtual_clients,
    ScannerNoiseTransform,
    IntensityShiftTransform,
)
from .utils import (
    count_parameters,
    average_weights,
    compute_heterogeneity_metrics,
    compute_optimal_theta,
    set_seed,
    get_device,
)
from .visualization import (
    plot_label_distribution_heatmap,
    plot_sample_distribution_bar,
    plot_heterogeneity_comparison,
    plot_js_divergence_matrix,
    plot_class_distribution_radar,
    create_heterogeneity_report,
)
