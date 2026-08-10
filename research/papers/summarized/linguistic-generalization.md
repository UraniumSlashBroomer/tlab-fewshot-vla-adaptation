# Enhancing Linguistic Generalization of VLA: Fine-Tuning OpenVLA via Synthetic Instruction Augmentation

## Core idea

The paper uses an LLM to generate several semantically varied instructions for each robot trajectory, then fine-tunes OpenVLA with LoRA on randomly paired trajectory-instruction variants.

## Architecture

No new policy architecture: LoRA is applied to OpenVLA attention projections while its original discrete action objective remains unchanged.

## Objective

The standard action prediction loss is used after replacing the single dataset instruction with one of several generated paraphrases.

## Important implementation details

- The reported experiment uses a manually curated 100-trajectory BridgeData V2 subset and action-token accuracy, not rollout success.
- It reports a slight decrease in exact token accuracy and a small increase in 5-bin tolerance; it does not provide a LIBERO few-shot result.

## Relevant to our project

The idea may help only if the mandatory wrong-instruction control reveals language insensitivity. But generating instructions from external LLMs and using paraphrases introduces a new language-data source, which conflicts with the strict "only LIBERO data" interpretation. We should not use it as a main method.

## Open questions

The study's small curated dataset, manual selection, and lack of rollout evaluation make its claimed generalization transfer uncertain.
