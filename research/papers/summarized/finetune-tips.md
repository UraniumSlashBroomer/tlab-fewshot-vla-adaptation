# What to Ignore, What to React: Visually Robust RL Fine-Tuning of VLA Models

## Core idea

PAIR-VLA augments online PPO adaptation with paired visual perturbations: the policy should be invariant to task-irrelevant changes and sensitive to task-altering changes. It supervises the resulting action distributions, not merely visual representations.

## Architecture

The deployed VLA architecture is unchanged. During RL fine-tuning, the method creates task-preserving and task-altering versions of each observation and compares their action distributions.

## Objective

PPO is augmented with an action-distribution invariance loss for nuisance perturbations and a separation/sensitivity loss for target-relevant perturbations.

## Important implementation details

- The paper evaluates OpenVLA and pi0.5 on ManiSkill3, not LIBERO few-shot imitation.
- It needs online environment interaction, task rewards, and an RL implementation for flow-matching policies.

## Relevant to our project

Not a valid main method under this assignment: the environment is allowed for evaluation, not collection of new interaction data, and the task has no reward-function premise. The conceptual lesson supports cautious visual augmentation, but the actual PAIR-VLA objective should not be used.

## Open questions

Offline equivalents would be a different method and are not established by this paper.
