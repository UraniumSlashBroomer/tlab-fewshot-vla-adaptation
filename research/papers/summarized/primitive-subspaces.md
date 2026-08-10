# Primitive Subspaces as a Causal Mechanism for Few-Shot Task Transfer in VLAs

## Core idea

The paper compares flat task-level training with training on manually segmented primitive-level episodes and primitive-specific instructions. At test time it conditions the policy on embeddings of a few held-out demonstrations without updating weights.

## Architecture

It requires constructing primitive segments, primitive prompts, demonstration encodings, and an inference-time mechanism that prepends demonstration embeddings to the VLA context. It studies OpenVLA and pi0.5.

## Objective

The main training objective is ordinary policy supervision; the proposed difference is the primitive-segmented data view. Linear probes and subspace ablations are analyses rather than training losses.

## Important implementation details

- The paper reports an advantage for recombining known primitives and a failure mode for genuinely novel primitive types.
- It includes a LIBERO-Long replication, but the reported Figure 2 explicitly says its numbers are "illustrative pending verification".
- No official implementation is listed in repository metadata.

## Relevant to our project

The broad hypothesis is relevant: `libero_90` may teach reusable reach/grasp/transport/place primitives. But creating primitive segmentation and demonstration-conditioned inference is a large protocol and architecture departure, while our assignment measures gradient-based 5/10/25 fine-tuning. Do not use this paper's quantitative claims as evidence until verified.

## Open questions

Whether canonical LIBERO data exposes usable primitive boundaries and whether context demonstration conditioning fits SmolVLA remain unresolved. The pending-verification statement makes this a background idea, not an implementation candidate.
