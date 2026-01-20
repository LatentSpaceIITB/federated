"""
MedCLIP Prompt Learning Module

Integrates MedCLIP (Bio_ClinicalBERT + Swin-ViT) with learnable soft prompts
for federated prompt learning in medical imaging.
"""

import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer
import torchvision

from .constants import (
    BERT_TYPE,
    VIT_TYPE,
    CHEXPERT_COMPETITION_TASKS,
    get_class_prompts,
)


class MedCLIPTextEncoder(nn.Module):
    """
    Text encoder based on Bio_ClinicalBERT with projection head.

    Architecture:
    - Bio_ClinicalBERT backbone
    - Average pooling over layers 1, 2, and last
    - Linear projection to 512-dim
    """

    def __init__(self, bert_type=BERT_TYPE, proj_dim=512, proj_bias=False):
        super().__init__()
        self.bert_type = bert_type
        self.model = AutoModel.from_pretrained(bert_type, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_type)
        self.projection_head = nn.Linear(768, proj_dim, bias=proj_bias)
        self.embed_dim = 768
        self.proj_dim = proj_dim

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through text encoder.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            text_embeds: Projected text embeddings [batch_size, proj_dim]
        """
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Average of layers 1, 2, and last (following MedCLIP)
        last_hidden_states = torch.stack([
            output['hidden_states'][1],
            output['hidden_states'][2],
            output['hidden_states'][-1]
        ])  # [3, batch, seqlen, 768]

        embed = last_hidden_states.permute(1, 0, 2, 3).mean(2).mean(1)  # [batch, 768]
        embed = self.projection_head(embed)

        return embed

    def get_word_embeddings(self):
        """Get the word embedding layer for prompt learning."""
        return self.model.embeddings.word_embeddings


class MedCLIPVisionEncoder(nn.Module):
    """
    Vision encoder based on Swin-ViT with projection head.
    """

    def __init__(self, vit_type=VIT_TYPE, proj_dim=512):
        super().__init__()
        self.vit_type = vit_type
        self.model = AutoModel.from_pretrained(vit_type)
        self.projection_head = nn.Linear(768, proj_dim, bias=False)
        self.embed_dim = 768
        self.proj_dim = proj_dim

    def forward(self, pixel_values, project=True):
        """
        Forward pass through vision encoder.

        Args:
            pixel_values: Input images [batch_size, 3, H, W]
            project: Whether to apply projection head

        Returns:
            img_embeds: Image embeddings [batch_size, proj_dim or embed_dim]
        """
        if pixel_values.shape[1] == 1:
            pixel_values = pixel_values.repeat((1, 3, 1, 1))

        output = self.model(pixel_values)
        img_embeds = output['pooler_output']  # [batch, 768]

        if project:
            img_embeds = self.projection_head(img_embeds)

        return img_embeds


class MedCLIPPromptLearner(nn.Module):
    """
    Learnable prompt for MedCLIP text encoder.

    Implements soft prompt learning where context tokens are learnable parameters.
    Supports multiple prompts for PromptFolio-style global/local splitting.

    Prompt format: [SOS] [CTX_1] ... [CTX_N] [CLASS] [EOS]
    """

    def __init__(
        self,
        classnames: list,
        text_encoder: MedCLIPTextEncoder,
        n_ctx: int = 8,
        num_prompts: int = 2,
        class_specific_context: bool = False,
        ctx_init: str = None,
    ):
        """
        Args:
            classnames: List of class names (pathologies)
            text_encoder: MedCLIP text encoder
            n_ctx: Number of context tokens
            num_prompts: Number of prompts (for PromptFolio: 1 global + 1 local)
            class_specific_context: Whether to use class-specific context
            ctx_init: Optional initialization string for context
        """
        super().__init__()

        self.n_cls = len(classnames)
        self.n_ctx = n_ctx
        self.num_prompts = num_prompts
        self.class_specific_context = class_specific_context
        self.classnames = classnames

        tokenizer = text_encoder.tokenizer
        word_embeddings = text_encoder.get_word_embeddings()
        ctx_dim = word_embeddings.weight.shape[1]  # 768 for Bio_ClinicalBERT
        dtype = word_embeddings.weight.dtype

        # Store class name lengths for prompt construction
        self.name_lens = []
        for name in classnames:
            tokens = tokenizer.encode(name, add_special_tokens=False)
            self.name_lens.append(len(tokens))

        # Initialize context vectors
        if ctx_init:
            # Initialize from text
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            self.n_ctx = n_ctx
            init_ids = tokenizer.encode(ctx_init, add_special_tokens=False)
            with torch.no_grad():
                init_embeds = word_embeddings(torch.tensor(init_ids))
            ctx_vectors = init_embeds.unsqueeze(0).repeat(num_prompts, 1, 1)
        else:
            # Random initialization
            if class_specific_context:
                ctx_vectors = torch.empty(num_prompts * self.n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                ctx_vectors = torch.empty(num_prompts, n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)

        self.ctx = nn.Parameter(ctx_vectors)

        # Create tokenized prompts for each class
        # Format: [CLS] X X X X {classname} [SEP]
        prompt_prefix = " ".join(["X"] * n_ctx)

        prompts_text = []
        for name in classnames:
            # Use clinical variant of class name
            prompt = f"{prompt_prefix} {name.lower()}"
            prompts_text.append(prompt)

        # Tokenize all prompts
        encoded = tokenizer(
            prompts_text,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )

        tokenized_prompts = encoded['input_ids']  # [n_cls, seq_len]
        attention_mask = encoded['attention_mask']

        # Repeat for num_prompts
        self.register_buffer("tokenized_prompts", tokenized_prompts.repeat(num_prompts, 1))
        self.register_buffer("attention_mask", attention_mask.repeat(num_prompts, 1))

        # Get token embeddings for prefix and suffix
        with torch.no_grad():
            embedding = word_embeddings(tokenized_prompts).type(dtype)

        # Token prefix is [CLS] token
        self.register_buffer("token_prefix", embedding[:, :1, :])  # [n_cls, 1, dim]
        # Token suffix is everything after context (class name + [SEP])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.ctx_dim = ctx_dim
        self.dtype = dtype

        print(f"MedCLIP Prompt Learner initialized:")
        print(f"  - Number of classes: {self.n_cls}")
        print(f"  - Number of context tokens: {self.n_ctx}")
        print(f"  - Number of prompts: {self.num_prompts}")
        print(f"  - Context dim: {ctx_dim}")

    def forward(self, prompt_idx: int = None):
        """
        Construct prompts from learnable context.

        Args:
            prompt_idx: If specified, only return prompts for this index
                       (0=global, 1=local for PromptFolio)

        Returns:
            prompts: Prompt embeddings [n_cls * num_prompts, seq_len, dim]
                    or [n_cls, seq_len, dim] if prompt_idx specified
        """
        ctx = self.ctx  # [num_prompts, n_ctx, dim] or [num_prompts * n_cls, n_ctx, dim]

        if prompt_idx is not None:
            # Select specific prompt set
            if self.class_specific_context:
                ctx = ctx[prompt_idx * self.n_cls:(prompt_idx + 1) * self.n_cls]
            else:
                ctx = ctx[prompt_idx:prompt_idx + 1]
                ctx = ctx.expand(self.n_cls, -1, -1)

            prefix = self.token_prefix
            suffix = self.token_suffix
        else:
            # Return all prompts
            if not self.class_specific_context:
                ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)
                ctx = ctx.reshape(self.num_prompts * self.n_cls, self.n_ctx, -1)

            prefix = self.token_prefix.repeat(self.num_prompts, 1, 1)
            suffix = self.token_suffix.repeat(self.num_prompts, 1, 1)

        # Concatenate: [CLS] + context + suffix
        prompts = torch.cat([prefix, ctx, suffix], dim=1)

        return prompts

    def get_attention_mask(self, prompt_idx: int = None):
        """Get attention mask for prompts."""
        if prompt_idx is not None:
            return self.attention_mask[:self.n_cls]
        return self.attention_mask


class CustomMedCLIP(nn.Module):
    """
    MedCLIP model with learnable prompts for federated learning.

    Combines:
    - Frozen MedCLIP vision encoder (Swin-ViT)
    - Frozen MedCLIP text encoder backbone (Bio_ClinicalBERT)
    - Learnable prompt tokens

    For PromptFolio-style training, maintains multiple prompts
    with portfolio mixing between global and local prompts.
    """

    def __init__(
        self,
        classnames: list,
        medclip_checkpoint: str = None,
        n_ctx: int = 8,
        num_prompts: int = 2,
        class_specific_context: bool = False,
        ctx_init: str = None,
        theta: float = 0.3,  # Portfolio mixing coefficient
    ):
        """
        Args:
            classnames: List of class names
            medclip_checkpoint: Path to pretrained MedCLIP weights
            n_ctx: Number of context tokens
            num_prompts: Number of prompts
            class_specific_context: Whether to use class-specific context
            ctx_init: Optional context initialization
            theta: Portfolio mixing coefficient (0=fully global, 1=fully local)
        """
        super().__init__()

        self.classnames = classnames
        self.n_cls = len(classnames)
        self.theta = theta
        self.num_prompts = num_prompts

        # Initialize encoders
        self.text_encoder = MedCLIPTextEncoder()
        self.vision_encoder = MedCLIPVisionEncoder()

        # Learnable temperature (initialize before loading weights)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))

        # Load pretrained weights if provided
        if medclip_checkpoint:
            self._load_medclip_weights(medclip_checkpoint)

        # Initialize prompt learner
        self.prompt_learner = MedCLIPPromptLearner(
            classnames=classnames,
            text_encoder=self.text_encoder,
            n_ctx=n_ctx,
            num_prompts=num_prompts,
            class_specific_context=class_specific_context,
            ctx_init=ctx_init,
        )

        # Freeze encoders, only train prompt learner
        self._freeze_encoders()

    def _load_medclip_weights(self, checkpoint_path: str):
        """Load pretrained MedCLIP weights."""
        import os
        weight_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        if os.path.exists(weight_path):
            state_dict = torch.load(weight_path, map_location="cpu")

            # Handle different key formats
            text_state_dict = {}
            vision_state_dict = {}

            for key, value in state_dict.items():
                if key.startswith("text_model."):
                    new_key = key.replace("text_model.", "")
                    text_state_dict[new_key] = value
                elif key.startswith("vision_model."):
                    new_key = key.replace("vision_model.", "")
                    vision_state_dict[new_key] = value
                elif "logit_scale" in key:
                    self.logit_scale.data = value

            if text_state_dict:
                self.text_encoder.load_state_dict(text_state_dict, strict=False)
            if vision_state_dict:
                self.vision_encoder.load_state_dict(vision_state_dict, strict=False)

            print(f"Loaded MedCLIP weights from {checkpoint_path}")
        else:
            print(f"Warning: MedCLIP weights not found at {checkpoint_path}")

    def _freeze_encoders(self):
        """Freeze vision and text encoder parameters."""
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Only prompt learner is trainable
        for param in self.prompt_learner.parameters():
            param.requires_grad = True

        # Logit scale can be trained
        self.logit_scale.requires_grad = True

    def encode_image(self, pixel_values):
        """Encode images to embeddings."""
        img_embeds = self.vision_encoder(pixel_values, project=True)
        img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
        return img_embeds

    def encode_text_with_prompts(self, prompt_idx: int = None):
        """
        Encode text using learnable prompts.

        This is a simplified version that embeds prompts through the text encoder.
        For Bio_ClinicalBERT, we need to handle the prompt embeddings differently
        than CLIP's transformer.
        """
        prompts = self.prompt_learner(prompt_idx=prompt_idx)
        attention_mask = self.prompt_learner.get_attention_mask(prompt_idx)

        # Get sequence length
        seq_len = prompts.shape[1]

        # Pass through BERT (using embeddings directly)
        # We need to bypass the embedding layer since we have soft prompts
        encoder = self.text_encoder.model.encoder

        # Create position embeddings
        position_ids = torch.arange(seq_len, device=prompts.device).unsqueeze(0).expand(prompts.shape[0], -1)
        position_embeds = self.text_encoder.model.embeddings.position_embeddings(position_ids)

        # Add position embeddings to prompts
        hidden_states = prompts + position_embeds

        # Apply layer norm and dropout
        hidden_states = self.text_encoder.model.embeddings.LayerNorm(hidden_states)
        hidden_states = self.text_encoder.model.embeddings.dropout(hidden_states)

        # Create extended attention mask
        extended_attention_mask = attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Pass through encoder layers
        all_hidden_states = [hidden_states]
        for layer in encoder.layer:
            hidden_states = layer(hidden_states, extended_attention_mask)[0]
            all_hidden_states.append(hidden_states)

        # Average layers 1, 2, and last (following MedCLIP)
        pooled_states = torch.stack([
            all_hidden_states[1],
            all_hidden_states[2],
            all_hidden_states[-1]
        ])
        embed = pooled_states.permute(1, 0, 2, 3).mean(2).mean(1)

        # Project
        text_embeds = self.text_encoder.projection_head(embed)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        return text_embeds

    def forward(self, pixel_values, labels=None, return_loss=True):
        """
        Forward pass for classification.

        Args:
            pixel_values: Input images [batch_size, C, H, W]
            labels: Multi-label targets [batch_size, n_cls] (optional)
            return_loss: Whether to compute and return loss

        Returns:
            dict with 'logits', 'loss' (optional), 'img_embeds', 'text_embeds'
        """
        device = pixel_values.device

        # Encode images
        img_embeds = self.encode_image(pixel_values)

        # Encode text with prompts (portfolio mixing)
        if self.num_prompts >= 2 and self.training:
            # PromptFolio: mix global and local prompts
            text_embeds_global = self.encode_text_with_prompts(prompt_idx=0)
            text_embeds_local = self.encode_text_with_prompts(prompt_idx=1)
            text_embeds = (1 - self.theta) * text_embeds_global + self.theta * text_embeds_local
        else:
            # Use all prompts or just global during eval
            text_embeds = self.encode_text_with_prompts(prompt_idx=0)

        # Compute logits
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * img_embeds @ text_embeds.t()  # [batch, n_cls]

        outputs = {
            'logits': logits,
            'img_embeds': img_embeds,
            'text_embeds': text_embeds,
        }

        # Compute loss if labels provided
        if labels is not None and return_loss:
            # Multi-label BCE loss
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            outputs['loss'] = loss

        return outputs

    def get_prompt_params(self):
        """Get prompt learner parameters for optimization."""
        return list(self.prompt_learner.parameters()) + [self.logit_scale]


class PromptOnlyMedCLIP(nn.Module):
    """
    Lightweight wrapper that only trains prompts.

    Uses pre-computed text features for efficiency in federated setting.
    """

    def __init__(
        self,
        classnames: list,
        n_ctx: int = 8,
        num_prompts: int = 2,
        ctx_dim: int = 768,
        proj_dim: int = 512,
        theta: float = 0.3,
    ):
        super().__init__()

        self.n_cls = len(classnames)
        self.n_ctx = n_ctx
        self.num_prompts = num_prompts
        self.theta = theta

        # Learnable context vectors (simplified without text encoder)
        ctx_vectors = torch.empty(num_prompts, n_ctx, ctx_dim)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)

        # Projection for context to match image features
        self.ctx_proj = nn.Linear(ctx_dim, proj_dim, bias=False)

        # Class embeddings (can be initialized from text encoder)
        self.class_embeds = nn.Parameter(torch.randn(self.n_cls, proj_dim))
        nn.init.normal_(self.class_embeds, std=0.02)

        # Logit scale
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))

    def forward(self, img_embeds, labels=None, return_loss=True):
        """
        Forward pass with pre-computed image embeddings.

        Args:
            img_embeds: Pre-computed image embeddings [batch, proj_dim]
            labels: Labels for loss computation
            return_loss: Whether to return loss
        """
        # Get context features
        if self.training and self.num_prompts >= 2:
            ctx_global = self.ctx[0]  # [n_ctx, ctx_dim]
            ctx_local = self.ctx[1]
            ctx = (1 - self.theta) * ctx_global + self.theta * ctx_local
        else:
            ctx = self.ctx[0]

        # Project and pool context
        ctx_pooled = self.ctx_proj(ctx.mean(dim=0))  # [proj_dim]

        # Combine with class embeddings
        text_embeds = self.class_embeds + ctx_pooled.unsqueeze(0)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # Normalize image embeddings
        img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)

        # Compute logits
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * img_embeds @ text_embeds.t()

        outputs = {'logits': logits}

        if labels is not None and return_loss:
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            outputs['loss'] = loss

        return outputs
