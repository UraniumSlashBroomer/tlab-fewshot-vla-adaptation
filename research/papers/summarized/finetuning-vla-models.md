# Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success

## Core idea

OpenVLA-OFT argues that fine-tuning quality can improve by replacing autoregressive discrete actions with a parallel continuous action-chunk head trained with L1 loss. It reports strong LIBERO results for OpenVLA, as well as faster inference.

## Architecture

OFT changes OpenVLA by using multiple images and state tokens, a bidirectional parallel action decoder, a four-layer MLP continuous-action head, and an action chunk of length `K`. OFT+ further applies language-conditioned FiLM modulation inside the visual encoder.

## Objective

The main OFT action head is supervised with L1 regression on normalized continuous actions. The authors also compare a diffusion head; they report comparable task performance with more efficient L1 training and inference in their controlled OpenVLA setting.

## Important implementation details

- OFT is an architectural conversion of an autoregressive, discretized OpenVLA, not merely a hyperparameter change.
- On LIBERO, the authors train until normalized L1 loss is below 0.01, with checkpoint evaluation; they note the best checkpoint can differ by suite.
- FiLM is important for their demanding real-robot language-grounding tasks but has only a small gain in their LIBERO comparison.
- Multiple images, action chunking, and continuous actions are central in their recipe.

## Relevant to our project

SmolVLA already has continuous, chunked flow-matching actions and multiple image/state conditioning. Therefore the headline OFT conversion is not directly applicable. The transferable hypothesis is narrower: preserve continuous chunked control; test parameter-efficient tuning and perhaps language-conditioned visual modulation only after verifying that the wrong-instruction control exposes weak language use.

## Open questions

The paper studies OpenVLA, not SmolVLA, and uses much larger data budgets. It gives no evidence that adding FiLM or replacing SmolVLA's flow-matching expert with L1 improves a 5-demo target adaptation; replacing the native head would confound the assignment and should be a later, high-cost ablation.
