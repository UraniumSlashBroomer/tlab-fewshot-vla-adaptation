# FiberTune: Preserving Action-Fiber Visual Residuals in Vision-Language-Action Fine-Tuning

## Core idea

FiberTune argues that action loss can collapse visual information that does not change the immediate action but is important for future behavior or generalization. It preserves the action-orthogonal residual of intermediate visual tokens while fine-tuning.

## Architecture

At training time, an online action probe estimates action-predictive representation directions. The method removes these directions from visual-token features, aligns the residual to a frozen visual teacher through a small fixed adapter, and regularizes the residual's effective rank. The probe, teacher, and adapter are discarded at inference.

## Objective

`L = L_task + lambda_align L_residual-align + lambda_rank L_effective-rank`. The alignment is applied to probe-filtered residual tokens, not the full visual representation.

## Important implementation details

- The paper reports controlled fine-tuning improvements on pi0.5 and OpenVLA-OFT, including LIBERO.
- Its ablation finds full-token teacher alignment harmful in one setting; effective-rank regularization contributes much of its reported benefit.
- It needs access to intermediate visual token features and a suitable frozen teacher feature space.

## Relevant to our project

This is an appealing training-only regularizer because it preserves SmolVLA's deployed architecture and action head. It could be adapted to the SmolVLM visual token stream, but requires careful feature hooks, a frozen teacher copy, and probe calibration. It is a medium-risk candidate after simple LoRA/replay baselines, not before the core pipeline works.

## Open questions

The paper does not test SmolVLA or a strict 5-demo regime. Its small LIBERO gains from already adapted checkpoints do not predict a reliable cost-curve shift, and the correct feature layer/teacher for SmolVLA is unspecified.
