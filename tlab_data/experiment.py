"""Shared training and evaluation helpers for the fixed LIBERO protocol."""

import random
from pathlib import Path

import numpy as np
import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from tlab_data.smolvla import configure_for_libero


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_libero_policy(
    checkpoint: str | Path,
    device: str,
    *,
    action_chunk_size: int | None = None,
    action_execution_steps: int | None = None,
    freeze_vision_encoder: bool | None = None,
    train_expert_only: bool | None = None,
    train_state_proj: bool | None = None,
) -> SmolVLAPolicy:
    config = SmolVLAConfig.from_pretrained(checkpoint)
    configure_for_libero(config)
    if action_chunk_size is not None:
        config.chunk_size = action_chunk_size
    if action_execution_steps is not None:
        config.n_action_steps = action_execution_steps
    if freeze_vision_encoder is not None:
        config.freeze_vision_encoder = freeze_vision_encoder
    if train_expert_only is not None:
        config.train_expert_only = train_expert_only
    if train_state_proj is not None:
        config.train_state_proj = train_state_proj
    config.device = device
    return SmolVLAPolicy.from_pretrained(checkpoint, config=config, strict=True)


def make_libero_processors(config: SmolVLAConfig, stats: dict[str, dict[str, torch.Tensor]]):
    return make_pre_post_processors(config, dataset_stats=stats)


def count_trainable_parameters(policy: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in policy.parameters() if parameter.requires_grad)
