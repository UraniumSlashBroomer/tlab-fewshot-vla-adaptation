# PriorVLA: Prior-Preserving Adaptation for Vision-Language-Action Models

## Core idea

PriorVLA argues that scarce-data fine-tuning should preserve the pretrained policy as an active source of scene and motor priors, not merely use it for initialization. A frozen copy of the action expert supplies motor-prior features to a trainable adaptation expert.

## Architecture

The pretrained action expert is duplicated into a frozen Prior Expert and trainable Adaptation Expert. Learnable scene, motor, and action query tokens route VLM scene features and Prior Expert denoising features to the adaptation branch. Only the adaptation branch generates actions.

## Objective

It keeps the standard flow-matching MSE loss; the Prior Expert is never directly optimized or decoded into actions.

## Important implementation details

- PriorVLA is built on pi0.5 and executes both action experts at every denoising step, increasing memory and compute.
- It tunes the adaptation expert, queries, and vision encoder while freezing the Prior Expert and most VLM parameters.
- It reports LIBERO and 10-demo real-robot results, but not SmolVLA or our strict LIBERO split.

## Relevant to our project

Its central anti-forgetting insight supports seen replay and parameter-efficient adaptation. A literal port is expensive because it duplicates SmolVLA's action expert and changes attention paths; it conflicts with our preference for light overhead around the baseline. Treat it as motivation, not a first implementation.

## Open questions

The cost of a frozen duplicate expert on a P100 and the required SmolVLA attention-mask changes are unknown. There is no evidence that its pi0.5 results transfer to 5-demo LIBERO-Goal adaptation.
