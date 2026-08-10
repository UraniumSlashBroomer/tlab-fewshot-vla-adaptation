"""Train the fixed SmolVLA LIBERO baseline from raw HDF5 demonstrations."""

import itertools
import json
import logging
import math
import shutil
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from tlab_data.experiment import (
    count_trainable_parameters,
    load_libero_policy,
    make_libero_processors,
    seed_everything,
)
from tlab_data.libero_hdf5 import LiberoHDF5Dataset


def _dataset_from_config(cfg: DictConfig) -> LiberoHDF5Dataset:
    if cfg.run.stage == "seen":
        split = cfg.data.seen
        return LiberoHDF5Dataset(
            cfg.data.root,
            split.suite,
            list(split.task_ids),
            split.demos_per_task,
            cfg.policy.action_chunk_size,
        )

    if cfg.run.stage == "target":
        if cfg.run.task_id is None or cfg.run.budget is None:
            raise ValueError("Target training requires run.task_id and run.budget.")
        if cfg.run.task_id not in cfg.data.goal.task_ids:
            raise ValueError(f"Task {cfg.run.task_id} is not in the fixed target split.")
        if cfg.run.budget not in cfg.data.goal.budgets:
            raise ValueError(f"Budget {cfg.run.budget} is not in the fixed target budgets.")
        return LiberoHDF5Dataset(
            cfg.data.root,
            cfg.data.goal.suite,
            [cfg.run.task_id],
            cfg.run.budget,
            cfg.policy.action_chunk_size,
        )

    raise ValueError(f"Unknown run.stage: {cfg.run.stage}")


def _output_dir(cfg: DictConfig) -> Path:
    if cfg.run.output_dir is not None:
        return Path(cfg.run.output_dir)
    if cfg.run.stage == "seen":
        name = f"seen_seed_{cfg.run.seed}"
    else:
        name = f"target_{cfg.run.task_id}_budget_{cfg.run.budget}_seed_{cfg.run.seed}"
    return Path("outputs") / name


def _training_schedule(cfg: DictConfig) -> tuple[int, int, int]:
    if cfg.run.stage == "seen":
        return (
            cfg.training.seen_steps,
            cfg.training.seen_warmup_steps,
            cfg.training.seen_checkpoint_every_steps,
        )
    return (
        cfg.training.target_steps,
        cfg.training.target_warmup_steps,
        cfg.training.target_checkpoint_every_steps,
    )


def _normalization_stats(cfg: DictConfig, dataset: LiberoHDF5Dataset) -> dict[str, dict[str, torch.Tensor]]:
    if cfg.run.stage == "seen":
        return dataset.seen_dataset_stats()

    stats_path = Path(cfg.run.init_checkpoint) / "dataset_stats.pt"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Target fine-tuning requires seen normalization statistics: {stats_path}"
        )
    return torch.load(stats_path, map_location="cpu", weights_only=True)


def _lr_multiplier(step: int, total_steps: int, warmup_steps: int, min_lr_ratio: float) -> float:
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr_ratio + (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2


def _write_json(path: Path, contents: dict) -> None:
    path.write_text(json.dumps(contents, indent=2, sort_keys=True) + "\n")


def _torch_dtype(name: str) -> torch.dtype:
    return getattr(torch, name)


def _save_checkpoint(
    output_dir: Path,
    step: int,
    policy,
    stats: dict[str, dict[str, torch.Tensor]],
    keep_last: int,
) -> Path:
    checkpoint_dir = output_dir / "checkpoints" / f"step_{step:06d}"
    policy_dir = checkpoint_dir / "policy"
    policy.save_pretrained(policy_dir)
    torch.save(stats, policy_dir / "dataset_stats.pt")

    checkpoints = sorted((output_dir / "checkpoints").glob("step_*"))
    for stale_checkpoint in checkpoints[:-keep_last]:
        shutil.rmtree(stale_checkpoint)
    return policy_dir


def _configure_logging(output_dir: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(output_dir / "train.log"), logging.StreamHandler()],
        force=True,
    )


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    if not cfg.runtime.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError(f"Configured device is unavailable: {cfg.runtime.device}")

    output_dir = _output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=False)
    _configure_logging(output_dir)
    OmegaConf.save(cfg, output_dir / "config.yaml", resolve=True)
    seed_everything(cfg.run.seed)

    steps, warmup_steps, checkpoint_every = _training_schedule(cfg)
    dataset = _dataset_from_config(cfg)
    stats = _normalization_stats(cfg, dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.micro_batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )
    policy = load_libero_policy(
        cfg.run.init_checkpoint,
        cfg.runtime.device,
        model_dtype=_torch_dtype(cfg.runtime.model_dtype),
        action_chunk_size=cfg.policy.action_chunk_size,
        action_execution_steps=cfg.policy.action_execution_steps,
        freeze_vision_encoder=cfg.policy.freeze_vision_encoder,
        train_expert_only=cfg.policy.train_expert_only,
        train_state_proj=cfg.policy.train_state_proj,
    )
    preprocessor, _ = make_libero_processors(policy.config, stats)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in policy.parameters() if parameter.requires_grad),
        lr=cfg.training.learning_rate,
        betas=tuple(cfg.training.betas),
        weight_decay=cfg.training.weight_decay,
    )
    min_lr_ratio = 2.5e-6 / cfg.training.learning_rate
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _lr_multiplier(step, steps, warmup_steps, min_lr_ratio),
    )
    amp_dtype = _torch_dtype(cfg.runtime.amp_dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    run_metadata = {
        "run": OmegaConf.to_container(cfg.run, resolve=True),
        "data_manifest": dataset.source_manifest(),
        "trainable_parameters": count_trainable_parameters(policy),
    }
    _write_json(output_dir / "run.json", run_metadata)
    logging.info("frames=%s episodes=%s trainable_parameters=%s", len(dataset), len(dataset.episodes), run_metadata["trainable_parameters"])

    wandb_run = None
    if cfg.wandb.enabled:
        import wandb

        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=output_dir.name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    data_iterator = itertools.cycle(dataloader)
    policy.train()
    optimizer.zero_grad(set_to_none=True)
    interval_loss = 0.0
    interval_start = time.perf_counter()

    for step in range(1, steps + 1):
        step_loss = 0.0
        for _ in range(cfg.training.gradient_accumulation_steps):
            raw_batch = next(data_iterator)
            batch = preprocessor(raw_batch)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                loss, loss_dict = policy(batch)
            scaler.scale(loss / cfg.training.gradient_accumulation_steps).backward()
            step_loss += loss.item()

        if step == 1:
            has_gradient = any(
                parameter.grad is not None for parameter in policy.parameters() if parameter.requires_grad
            )
            if not has_gradient:
                raise RuntimeError("No gradients reached the trainable SmolVLA parameters.")

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.training.grad_clip_norm)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"Non-finite gradient norm at step {step}: {grad_norm}")
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        interval_loss += step_loss / cfg.training.gradient_accumulation_steps

        if step % cfg.training.log_every_steps == 0:
            elapsed = time.perf_counter() - interval_start
            log_values = {
                "step": step,
                "loss": interval_loss / cfg.training.log_every_steps,
                "grad_norm": float(grad_norm),
                "lr": scheduler.get_last_lr()[0],
                "updates_per_second": cfg.training.log_every_steps / elapsed,
                **loss_dict,
            }
            logging.info("%s", json.dumps(log_values, sort_keys=True))
            if wandb_run is not None:
                wandb_run.log(log_values, step=step)
            interval_loss = 0.0
            interval_start = time.perf_counter()

        if step % checkpoint_every == 0 or step == steps:
            policy_dir = _save_checkpoint(
                output_dir,
                step,
                policy,
                stats,
                cfg.training.keep_last_checkpoints,
            )
            logging.info("saved checkpoint: %s", policy_dir)

    dataset.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
