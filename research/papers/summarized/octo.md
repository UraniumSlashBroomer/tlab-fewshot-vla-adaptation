# Octo: An Open-Source Generalist Robot Policy

## Core idea

Octo is a transformer generalist robot policy designed around flexible observation and action interfaces. It shows that diverse robot pretraining can provide an initialization that is rapidly adapted to new tasks and modalities.

## Architecture

Images, language or goal-image task specification, and proprioception are converted to tokens and processed by a transformer. A readout predicts a multi-step action chunk. Octo accommodates missing modalities through token-level interface choices.

## Objective

The paper models each action dimension with discretized bins and cross-entropy, rather than SmolVLA's continuous flow-matching action expert.

## Important implementation details

- Training uses image augmentations and modality dropout to tolerate heterogeneous datasets.
- The model is explicitly designed for multi-task mixtures and fast downstream fine-tuning.
- The paper's action-space and framework choices are different from LeRobot SmolVLA.

## Relevant to our project

The useful contribution is the transfer hypothesis: target adaptation can be regularized by retaining exposure to a related multi-task mixture. In our allowed setting, this becomes replay/mixing of `libero_90` demonstrations while fine-tuning on 5/10/25 target trajectories. Mild visual augmentations are also a compatible, low-cost hypothesis because camera and scene semantics are shared across LIBERO tasks.

## Open questions

The paper does not isolate replay mixing in the exact LIBERO-90-to-goal transfer setting, so the mixture ratio and sampling schedule must be treated as an experiment rather than adopted as fact.
