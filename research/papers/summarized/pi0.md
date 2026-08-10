# pi0: A Vision-Language-Action Flow Model for General Robot Control

## Core idea

pi0 combines a pretrained VLM with a dedicated action expert trained by conditional flow matching on continuous action chunks. It separates broad, diverse pretraining from high-quality downstream post-training and reports stronger low-data transfer than several baselines on its tasks.

## Architecture

The model consumes multi-camera images, language, and proprioception. Image/language tokens use a pretrained VLM expert; state and noisy action-chunk tokens use a smaller action expert. The action horizon is 50 in the paper. Action tokens can attend to all conditioning inputs and one another.

## Objective

As in conditional flow matching, the model learns a vector field that maps a Gaussian-noised action chunk toward the demonstration action chunk. The paper samples the noise time from a beta distribution biased toward noisier inputs.

## Important implementation details

- The paper uses a large and heterogeneous pretraining mix, then fine-tunes on curated data for each downstream skill.
- Its claimed benefit of pretraining is greatest when downstream data are scarce or the task resembles pretraining skills.
- It uses a separate action expert so robotics-specific tokens need not use exactly the same MLP capacity as the VLM backbone.

## Relevant to our project

SmolVLA already shares the central VLM-plus-flow-action-expert structure. The most relevant adaptation idea is not to replace the architecture, but to reproduce pi0's two-stage logic within the rules: first `libero_90` seen training, then target adaptation mixed with a controlled amount of seen replay. This may preserve reusable reach/grasp/place skills while a small target set teaches task-specific object and goal binding.

## Open questions

pi0's data scale and robot mix are orders of magnitude larger, and its paper does not determine a replay ratio or update scope for SmolVLA on LIBERO. Its separate-expert design is already present conceptually but not necessarily identical in the released SmolVLA implementation.
