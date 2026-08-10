# SmolVLA: A vision-language-action model for affordable and efficient robotics

## Core idea

SmolVLA combines a compact pretrained VLM with a separate action expert that predicts a chunk of continuous robot actions by conditional flow matching. The paper targets efficient training and inference while retaining transfer from a VLM.

## Architecture

- Inputs: RGB image(s), language instruction, and robot state.
- State is linearly projected into the VLM token space; image, language, and state tokens condition the action expert.
- The action expert alternates cross-attention to VLM features and causal self-attention over action tokens.
- It predicts an action chunk; the released 450M model uses a 50-action horizon and 10 flow-integration steps at inference.

## Objective

For action chunk `A`, noise `eps`, and time `tau`, the noised action is `A_tau = tau A + (1 - tau) eps`. The expert predicts the conditional vector field using squared error. This is the native SmolVLA training objective.

## Important implementation details

- The VLM backbone is frozen in the paper's main pretraining and simulation fine-tuning recipe; the action expert is trainable.
- The paper uses 512x512 images, AdamW, and normalizes actions. Its published simulation runs use 100k steps, but this is not a suitable fixed budget for the 5/10/25-demo task without a small-data selection protocol.
- LIBERO ablations find: interleaved cross- and causal self-attention beats either alone; flow matching beats L1 action regression in their setup; action chunks around 10--50 outperform chunk size 1; querying observations more often improves success.
- Feeding state as a VLM-prefix token outperforms supplying it only to the action expert in their ablation.

## Relevant to our project

This is the direct architectural reference: we should preserve its action interface and flow-matching loss for the first baseline. The low-risk adaptation knobs are trainable-module choice (expert only vs selected VLM modules), replay mixing from `libero_90`, image augmentation, and chunk/replanning settings. Its LIBERO ablations justify testing chunk horizon and closed-loop action execution, but they do not establish their best values for our few-demo protocol.

## Open questions

The paper does not study the exact `0/5/10/25` per-task adaptation setting or report variance over two fine-tuning seeds there. We must validate whether expert-only adaptation has enough capacity at 5 demos and whether action normalization is consistent across the seen and held-out splits.
