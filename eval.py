"""Evaluate a trained SmolVLA checkpoint in the official LIBERO environment."""

import json
import os
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from tlab_data.experiment import load_libero_policy, make_libero_processors


def _write_json(path: Path, contents: dict) -> None:
    path.write_text(json.dumps(contents, indent=2, sort_keys=True) + "\n")


def _policy_dir(checkpoint: str) -> Path:
    path = Path(checkpoint)
    return path / "policy" if (path / "policy").exists() else path


def _wrong_instruction(task_id: int, task_ids: list[int], configured_id: int | None) -> str:
    wrong_task_id = configured_id if configured_id is not None else task_ids[(task_ids.index(task_id) + 1) % len(task_ids)]
    if wrong_task_id == task_id:
        raise ValueError("The language-control instruction must come from another target task.")
    from libero.libero import benchmark

    return benchmark.get_benchmark_dict()["libero_goal"]().get_task(wrong_task_id).language


def _evaluate_task(cfg: DictConfig, policy, preprocessor, postprocessor, task_id: int) -> dict:
    from gymnasium.vector import SyncVectorEnv
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.envs.libero import create_libero_envs
    from lerobot.scripts.lerobot_eval import eval_policy

    envs_by_suite = create_libero_envs(
        task=cfg.evaluation.suite,
        n_envs=cfg.evaluation.num_envs,
        env_cls=SyncVectorEnv,
        gym_kwargs={
            "task_ids": [task_id],
            "obs_type": "pixels_agent_pos",
            "observation_height": 128,
            "observation_width": 128,
        },
    )
    env = envs_by_suite[cfg.evaluation.suite][task_id]
    env_config = LiberoEnvConfig(task=cfg.evaluation.suite, task_ids=[task_id])
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_config, policy.config)

    if cfg.evaluation.language_control:
        instruction = _wrong_instruction(
            task_id,
            list(cfg.evaluation.task_ids),
            cfg.evaluation.language_control_task_id,
        )
        for single_env in env.envs:
            single_env.task_description = instruction

    videos_dir = None
    if cfg.evaluation.videos_per_task:
        videos_dir = Path(cfg.output_dir) / "videos" / f"task_{task_id}"
    result = eval_policy(
        env,
        policy,
        env_preprocessor,
        env_postprocessor,
        preprocessor,
        postprocessor,
        n_episodes=cfg.evaluation.episodes_per_task,
        max_episodes_rendered=cfg.evaluation.videos_per_task,
        videos_dir=videos_dir,
        start_seed=cfg.evaluation.first_rollout_seed,
    )
    env.close()
    return result


@hydra.main(version_base=None, config_path="configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    if cfg.checkpoint is None:
        raise ValueError("Pass checkpoint=<run>/checkpoints/step_XXXXXX or its policy directory.")
    if not cfg.runtime.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError(f"Configured device is unavailable: {cfg.runtime.device}")

    policy_dir = _policy_dir(cfg.checkpoint)
    output_dir = Path(cfg.output_dir) if cfg.output_dir is not None else policy_dir.parents[2]
    matplotlib_config_dir = output_dir / ".matplotlib"
    matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MUJOCO_GL"] = cfg.runtime.mujoco_gl
    os.environ["MPLCONFIGDIR"] = str(matplotlib_config_dir)
    stats = torch.load(policy_dir / "dataset_stats.pt", map_location="cpu", weights_only=True)
    policy = load_libero_policy(policy_dir, cfg.runtime.device)
    preprocessor, postprocessor = make_libero_processors(policy.config, stats)

    task_results = {}
    for task_id in cfg.evaluation.task_ids:
        task_results[str(task_id)] = _evaluate_task(cfg, policy, preprocessor, postprocessor, task_id)

    success_rates = [result["aggregated"]["pc_success"] for result in task_results.values()]
    evaluation = {
        "checkpoint": str(policy_dir),
        "config": OmegaConf.to_container(cfg.evaluation, resolve=True),
        "per_task": task_results,
        "mean_success_rate": sum(success_rates) / len(success_rates),
    }
    metrics_path = output_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {"evaluations": []}
    metrics.setdefault("evaluations", []).append(evaluation)
    _write_json(metrics_path, metrics)

    if cfg.wandb.enabled:
        import wandb

        run = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, job_type="evaluation")
        run.log({"mean_success_rate": evaluation["mean_success_rate"]})
        run.finish()


if __name__ == "__main__":
    main()
