# VLA-GSE: Boosting Parameter-Efficient Fine-Tuning in VLA with Generalized and Specialized Experts

## Core idea

VLA-GSE is a parameter-efficient fine-tuning method intended to preserve VLM knowledge while providing more adaptation capacity than ordinary LoRA. It combines an always-on generalized low-rank expert with input-routed specialized low-rank experts.

## Architecture

For each frozen VLM weight, SVD components initialize the generalized and specialized adapters. A sparse router chooses top-k specialized experts; auxiliary load balancing, gradient-scale balancing, and backbone-weight adjustment stabilize routing and initialization. The paper fully tunes its separate action head.

## Objective

The main objective is the downstream action loss plus a router load-balancing term. It does not introduce a new action representation.

## Important implementation details

- The method is built for OpenVLA-OFT-style VLM weights and uses SVD decomposition across many backbone matrices.
- It reports results on LIBERO-Plus, not the strict LIBERO-90-to-goal protocol.
- It changes the PEFT mechanism rather than the action head and updates a small fraction of the full model.

## Relevant to our project

It supports the general hypothesis that limited-parameter adaptation may beat full fine-tuning with scarce data. But it is not an initial implementation candidate: porting its routed SVD adapters to SmolVLA is substantially more work and risk than ordinary LoRA, without direct evidence for 5--25 trajectory SmolVLA adaptation.

## Open questions

The paper provides no SmolVLA implementation or direct comparison to expert-only tuning in our setting. It would be appropriate only if ordinary LoRA shows a clear benefit and we have surplus compute.
