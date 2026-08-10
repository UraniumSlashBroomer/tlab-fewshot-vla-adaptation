# LIBERO SmolVLA adaptation

This repository trains and evaluates the fixed protocol in `research/hypothesis.md`.
Raw HDF5 data is read on the fly; no LeRobot dataset copy is created.

## Setup

Use Python 3.11 and install the project requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

LIBERO is installed from the official source because its published PyPI packaging is not usable with current pip. The source checkout is intentionally ignored by Git, but its revision is fixed:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
git -C third_party/LIBERO checkout 8f1084e
pip install -e third_party/LIBERO --no-deps --no-build-isolation --config-settings editable_mode=compat
export LIBERO_CONFIG_PATH="$PWD/.libero"
export MUJOCO_GL=egl
python scripts/setup_libero_config.py
```

The script creates LIBERO's normally interactive path configuration without prompts.

Download only the fixed 40 seen tasks and three target tasks (rather than all 90 seen HDF5 files):

```bash
python scripts/download_libero_hdf5.py
```

## Runs

The baseline uses the fixed 40 seen tasks and target IDs 0--2 from `configs/data/libero.yaml`. SmolVLA predicts chunks of 50 actions but executes one action before reading the next simulation observation, matching its simulation recipe.

```bash
# Seen pretrain
python train.py run.stage=seen run.seed=0

# Target fine-tune from a seen checkpoint
python train.py run.stage=target run.task_id=0 run.budget=5 run.seed=0 \
  run.init_checkpoint=outputs/seen_seed_0/checkpoints/step_030000/policy

# 40 fixed online rollouts per target task
python eval.py checkpoint=outputs/target_0_budget_5_seed_0/checkpoints/step_005000
```

Set `wandb.enabled=true` to mirror metrics to Weights & Biases. It is disabled by default. Training writes diagnostics to `train.log`; evaluation appends structured rollout results to `metrics.json`.
