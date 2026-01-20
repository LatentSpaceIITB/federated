"""
MedPromptFolio: Federated Prompt Learning for Medical Vision-Language Models

Core algorithm implementation extending PromptFolio to medical imaging with MedCLIP.

Key features:
- Portfolio strategy with global + local prompts
- Multi-label classification support (BCEWithLogitsLoss)
- Clinical prompt templates
- Medical-specific heterogeneity handling
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, List, Optional, Tuple
import numpy as np

from .medclip_prompt import CustomMedCLIP, MedCLIPPromptLearner
from .constants import CHEXPERT_COMPETITION_TASKS, FL_DEFAULT_CONFIG


class MedPromptFolio(nn.Module):
    """
    MedPromptFolio: Federated Prompt Learning with Portfolio Strategy for Medical VLMs.

    Maintains two sets of prompts:
    - Global prompt (index 0): Aggregated across all clients
    - Local prompt (index 1): Client-specific, not aggregated

    Forward pass mixes: P_mixed = (1-theta) * P_global + theta * P_local
    """

    def __init__(
        self,
        classnames: List[str] = None,
        medclip_checkpoint: str = None,
        n_ctx: int = 8,
        theta: float = 0.3,
        class_specific_context: bool = False,
        device: str = "cuda",
    ):
        """
        Args:
            classnames: List of pathology names
            medclip_checkpoint: Path to pretrained MedCLIP weights
            n_ctx: Number of context tokens
            theta: Portfolio mixing coefficient (0=fully global, 1=fully local)
            class_specific_context: Whether to use class-specific prompts
            device: Device to run on
        """
        super().__init__()

        self.classnames = classnames or CHEXPERT_COMPETITION_TASKS
        self.n_cls = len(self.classnames)
        self.n_ctx = n_ctx
        self.theta = theta
        self.device = device

        # Initialize MedCLIP with prompt learning
        self.model = CustomMedCLIP(
            classnames=self.classnames,
            medclip_checkpoint=medclip_checkpoint,
            n_ctx=n_ctx,
            num_prompts=2,  # 1 global + 1 local
            class_specific_context=class_specific_context,
            theta=theta,
        )

        # Loss function for multi-label classification
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Gradient scaler for mixed precision
        self.scaler = GradScaler()

        # Local state storage for federated learning
        self.local_info = {}
        self.global_info = None

    def forward(self, pixel_values: torch.Tensor, labels: torch.Tensor = None):
        """
        Forward pass.

        Args:
            pixel_values: Input images [batch, C, H, W]
            labels: Multi-label targets [batch, n_cls]

        Returns:
            dict with 'logits', 'loss' (if labels provided)
        """
        return self.model(pixel_values, labels=labels, return_loss=labels is not None)

    def get_prompt_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get prompt learner state dict."""
        return {
            'prompt_learner.ctx': self.model.prompt_learner.ctx.data.clone(),
            'logit_scale': self.model.logit_scale.data.clone(),
        }

    def load_prompt_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Load prompt learner state dict."""
        if 'prompt_learner.ctx' in state_dict:
            self.model.prompt_learner.ctx.data.copy_(state_dict['prompt_learner.ctx'])
        if 'logit_scale' in state_dict:
            self.model.logit_scale.data.copy_(state_dict['logit_scale'])

    def fed_init_model(self, global_weights: Dict[str, torch.Tensor]):
        """Initialize model for federated learning."""
        self.global_info = None
        self.local_info = {}

    def fed_upload_model(self, client_id: int):
        """
        Upload local model to server (store locally for simulation).

        Called after local training on client.
        """
        self.local_info[client_id] = self.get_prompt_state_dict()

    def fed_download_model(self, client_id: int):
        """
        Download model from server for local training.

        For MedPromptFolio:
        - Global prompt (index 0) comes from aggregated global
        - Local prompt (index 1) comes from client's local history
        """
        if client_id in self.local_info:
            # Load client's local prompts
            local_state = self.local_info[client_id]
            self.load_prompt_state_dict(local_state)

            # Replace global prompt with aggregated global
            if self.global_info is not None:
                ctx = self.model.prompt_learner.ctx.data
                # ctx shape: [num_prompts, n_ctx, dim] or [num_prompts * n_cls, n_ctx, dim]
                ctx[0] = self.global_info  # Replace first (global) prompt
                self.model.prompt_learner.ctx.data = ctx

    def fed_aggregate_model(self, client_ids: List[int], weights: List[float] = None):
        """
        Aggregate global prompts from participating clients.

        Only aggregates the global prompt (index 0), local prompts stay client-specific.

        Args:
            client_ids: List of participating client IDs
            weights: Optional aggregation weights (default: uniform)
        """
        if len(client_ids) == 0:
            return

        if weights is None:
            weights = [1.0 / len(client_ids)] * len(client_ids)

        # Aggregate only the global prompt (index 0)
        global_prompts = []
        for idx in client_ids:
            if idx in self.local_info:
                ctx = self.local_info[idx]['prompt_learner.ctx']
                # Extract global prompt (first prompt)
                global_prompts.append(ctx[0])

        if len(global_prompts) == 0:
            return

        # Weighted average
        global_prompts = torch.stack(global_prompts)
        weights_tensor = torch.tensor(weights, device=global_prompts.device).view(-1, 1, 1)
        aggregated_global = (global_prompts * weights_tensor).sum(dim=0)

        self.global_info = aggregated_global

    def set_theta(self, theta: float):
        """Update portfolio mixing coefficient."""
        self.theta = theta
        self.model.theta = theta


class MedPromptFolioTrainer:
    """
    Trainer for MedPromptFolio with federated learning support.
    """

    def __init__(
        self,
        model: MedPromptFolio,
        lr: float = 0.002,
        weight_decay: float = 0.0,
        use_amp: bool = True,
        device: str = "cuda",
    ):
        """
        Args:
            model: MedPromptFolio model
            lr: Learning rate
            weight_decay: Weight decay
            use_amp: Whether to use automatic mixed precision
            device: Device
        """
        self.model = model
        self.device = device
        self.use_amp = use_amp

        # Optimizer for prompt parameters only
        self.optimizer = torch.optim.AdamW(
            model.model.get_prompt_params(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6,
        )

        # Gradient scaler for AMP
        self.scaler = GradScaler() if use_amp else None

    def train_epoch(
        self,
        data_loader,
        global_weights: Dict = None,
        fedprox: bool = False,
        mu: float = 0.01,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            data_loader: Training data loader
            global_weights: Global model weights (for FedProx)
            fedprox: Whether to use FedProx regularization
            mu: FedProx regularization strength

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        self.model.to(self.device)

        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_labels = []

        for batch in data_loader:
            images = batch['img'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    outputs = self.model(images, labels)
                    loss = outputs['loss']

                    # FedProx regularization
                    if fedprox and global_weights is not None:
                        prox_term = self._compute_fedprox_term(global_weights, mu)
                        loss = loss + prox_term

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images, labels)
                loss = outputs['loss']

                if fedprox and global_weights is not None:
                    prox_term = self._compute_fedprox_term(global_weights, mu)
                    loss = loss + prox_term

                loss.backward()
                self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            # Collect predictions for metrics
            preds = torch.sigmoid(outputs['logits']).detach().cpu()
            all_preds.append(preds)
            all_labels.append(labels.cpu())

        # Compute metrics
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        avg_loss = total_loss / total_samples
        auc = self._compute_auc(all_preds, all_labels)

        return {
            'loss': avg_loss,
            'auc': auc,
            'samples': total_samples,
        }

    def _compute_fedprox_term(self, global_weights: Dict, mu: float) -> torch.Tensor:
        """Compute FedProx regularization term."""
        prox_loss = 0.0
        current_ctx = self.model.model.prompt_learner.ctx
        if 'prompt_learner.ctx' in global_weights:
            global_ctx = global_weights['prompt_learner.ctx'].to(self.device)
            prox_loss = (mu / 2) * torch.norm(current_ctx - global_ctx) ** 2
        return prox_loss

    def _compute_auc(self, preds: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute mean AUC across all classes."""
        try:
            from sklearn.metrics import roc_auc_score
            preds_np = preds.numpy()
            labels_np = labels.numpy()

            aucs = []
            for i in range(labels_np.shape[1]):
                if len(np.unique(labels_np[:, i])) > 1:
                    auc = roc_auc_score(labels_np[:, i], preds_np[:, i])
                    aucs.append(auc)

            return np.mean(aucs) if aucs else 0.0
        except Exception:
            return 0.0

    @torch.no_grad()
    def evaluate(self, data_loader) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        self.model.to(self.device)

        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_labels = []

        for batch in data_loader:
            images = batch['img'].to(self.device)
            labels = batch['label'].to(self.device)

            outputs = self.model(images, labels)
            loss = outputs['loss']

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            preds = torch.sigmoid(outputs['logits']).cpu()
            all_preds.append(preds)
            all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        avg_loss = total_loss / total_samples
        auc = self._compute_auc(all_preds, all_labels)

        # Compute per-class AUC
        per_class_auc = self._compute_per_class_auc(all_preds, all_labels)

        return {
            'loss': avg_loss,
            'auc': auc,
            'per_class_auc': per_class_auc,
            'samples': total_samples,
        }

    def _compute_per_class_auc(self, preds: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Compute AUC for each class."""
        try:
            from sklearn.metrics import roc_auc_score
            preds_np = preds.numpy()
            labels_np = labels.numpy()

            per_class = {}
            for i, classname in enumerate(self.model.classnames):
                if len(np.unique(labels_np[:, i])) > 1:
                    auc = roc_auc_score(labels_np[:, i], preds_np[:, i])
                    per_class[classname] = auc
                else:
                    per_class[classname] = 0.0

            return per_class
        except Exception:
            return {}


class FedAvgMedCLIP(MedPromptFolio):
    """
    FedAvg baseline with MedCLIP prompt learning.

    Aggregates all prompts (no global/local distinction).
    """

    def __init__(self, **kwargs):
        # Set theta=0 to use only global prompts
        kwargs['theta'] = 0.0
        super().__init__(**kwargs)

    def fed_aggregate_model(self, client_ids: List[int], weights: List[float] = None):
        """Aggregate all prompts (standard FedAvg)."""
        if len(client_ids) == 0:
            return

        if weights is None:
            weights = [1.0 / len(client_ids)] * len(client_ids)

        # Aggregate all prompt parameters
        aggregated_state = {}

        for key in ['prompt_learner.ctx', 'logit_scale']:
            values = []
            for idx in client_ids:
                if idx in self.local_info and key in self.local_info[idx]:
                    values.append(self.local_info[idx][key])

            if values:
                stacked = torch.stack(values)
                weights_tensor = torch.tensor(weights, device=stacked.device)
                # Handle different tensor dimensions
                for _ in range(stacked.dim() - 1):
                    weights_tensor = weights_tensor.unsqueeze(-1)
                aggregated_state[key] = (stacked * weights_tensor).sum(dim=0)

        # Store aggregated state
        self.global_info = aggregated_state.get('prompt_learner.ctx')

    def fed_download_model(self, client_id: int):
        """Download aggregated model to client."""
        if self.global_info is not None:
            self.model.prompt_learner.ctx.data.copy_(self.global_info)


class FedProxMedCLIP(FedAvgMedCLIP):
    """
    FedProx baseline with MedCLIP prompt learning.

    Adds proximal term to local training objective.
    """

    def __init__(self, mu: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.mu = mu


class LocalOnlyMedCLIP(MedPromptFolio):
    """
    Local-only baseline (no federation).

    Each client trains independently without any aggregation.
    """

    def __init__(self, **kwargs):
        kwargs['theta'] = 1.0  # Use only local prompts
        super().__init__(**kwargs)

    def fed_aggregate_model(self, client_ids: List[int], weights: List[float] = None):
        """No aggregation for local-only."""
        pass

    def fed_download_model(self, client_id: int):
        """Just load local model."""
        if client_id in self.local_info:
            self.load_prompt_state_dict(self.local_info[client_id])
