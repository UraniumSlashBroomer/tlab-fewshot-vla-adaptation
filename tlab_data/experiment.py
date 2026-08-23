"""Shared training and evaluation helpers for the fixed LIBERO protocol."""

import random
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
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
    adapter_base_checkpoint: str | Path | None = None,
    model_dtype: torch.dtype | None = None,
    action_chunk_size: int | None = None,
    action_execution_steps: int | None = None,
    freeze_vision_encoder: bool | None = None,
    train_expert_only: bool | None = None,
    train_state_proj: bool | None = None,
    adapter_trainable: bool = False,
) -> SmolVLAPolicy:
    # The checkpoint config includes a top-level policy type, so it must be
    # dispatched through the base config class before applying LIBERO overrides.
    config = PreTrainedConfig.from_pretrained(checkpoint)
    assert isinstance(config, SmolVLAConfig)
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
    checkpoint_path = Path(checkpoint)
    adapter_config_path = checkpoint_path / "adapter_config.json"
    if adapter_config_path.exists():
        from peft import PeftConfig, PeftModel

        adapter_config = PeftConfig.from_pretrained(checkpoint_path)
        policy = SmolVLAPolicy.from_pretrained(
            adapter_base_checkpoint or adapter_config.base_model_name_or_path,
            config=config,
            strict=True,
        )
        policy = PeftModel.from_pretrained(
            policy,
            checkpoint_path,
            config=adapter_config,
            is_trainable=adapter_trainable,
        )
    else:
        policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config, strict=True)
    if model_dtype is not None:
        policy.to(dtype=model_dtype)
    return policy


def wrap_with_lora(
    policy: SmolVLAPolicy,
    base_checkpoint: str | Path,
    *,
    rank: int,
    alpha: int,
    dropout: float,
):
    policy.config.pretrained_path = str(base_checkpoint)
    return policy.wrap_with_peft(
        peft_cli_overrides={
            "method_type": "lora",
            "r": rank,
            "lora_alpha": alpha,
            "lora_dropout": dropout,
        }
    )


def save_libero_policy(policy, save_directory: str | Path) -> None:
    save_directory = Path(save_directory)
    if hasattr(policy, "peft_config"):
        policy.save_pretrained(save_directory)
        policy.get_base_model().config.save_pretrained(save_directory)
    else:
        policy.save_pretrained(save_directory)


def make_libero_processors(config: SmolVLAConfig, stats: dict[str, dict[str, torch.Tensor]]):
    return make_pre_post_processors(config, dataset_stats=stats)


def count_trainable_parameters(policy: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in policy.parameters() if parameter.requires_grad)
