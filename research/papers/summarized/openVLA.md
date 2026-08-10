# OpenVLA: An Open Vision-Language-Action Model

## Core idea

OpenVLA turns a pretrained VLM into a robot policy by predicting discretized action tokens from an image and language instruction. Its main relevance is evidence for broad multi-task robot pretraining and careful fine-tuning of a VLM backbone.

## Architecture

The paper uses a Prismatic VLM: fused SigLIP and DINOv2 image features, a projector, and a Llama-2 language model. A continuous action vector is discretized dimension-wise into 256 bins and emitted autoregressively as language-model tokens.

## Objective

Cross-entropy is computed only on the action-token sequence. Action bin boundaries use the 1st and 99th action quantiles to reduce the effect of outliers.

## Important implementation details

- The authors find 224px images sufficient in their VLA experiments, despite higher-resolution image costs.
- Contrary to common VLM practice, their VLA results require updating the vision encoder during large-scale robot training.
- The paper supports LoRA fine-tuning in its codebase, but the original architecture and action representation differ materially from SmolVLA.
- The paper stresses testing language grounding in scenes with distractors, rather than reporting only aggregate task success.

## Relevant to our project

The assignment's seen-pretraining phase follows the same high-level principle: multi-task robot data should produce a stronger target-task initialization than task-only learning. More importantly, the paper motivates the mandated wrong-instruction control. If SmolVLA ignores language, improving visual/action fitting alone is not enough for this task.

## Open questions

Its scale (7B parameters and nearly one million demonstrations) is far from our setting. Its evidence for vision-encoder fine-tuning does not resolve whether full, LoRA, or expert-only adaptation is best for a 450M SmolVLA with 5--25 target trajectories.
