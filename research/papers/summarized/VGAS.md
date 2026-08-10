# VGAS: Value-Guided Action-Chunk Selection for Few-Shot Vision-Language-Action Adaptation

## Core idea

VGAS treats few-shot control as proposal generation plus selection. A fine-tuned SmolVLA generates several candidate action chunks; a learned chunk critic ranks them, and the best candidate is executed. This targets geometrically plausible but slightly inaccurate "near-miss" actions.

## Architecture

The policy remains SmolVLA. VGAS adds a Q-Chunk-Former critic conditioned on visual, language, state, and candidate action-chunk tokens. The critic is initialized from early SmolVLM decoder layers and uses state-action fusion.

## Objective

The critic is trained offline with temporal-difference anchoring plus conservative and explicit geometric regularizers. The latter constrains values locally around demonstration action chunks so Best-of-N does not select an unsupported but overestimated action. At inference, it samples `N` policy proposals and ranks them with the critic.

## Important implementation details

- The paper reports a controlled 5-shot LIBERO comparison with SmolVLA-0.5B and uses Best-of-N with `N=8`; gains diminish beyond eight samples.
- Its critic and regularizers are substantially more complex than behavior cloning and increase inference latency.
- The paper also reports positive 10-shot and 5--30-shot scaling, but not our exact split or checkpoint.

## Relevant to our project

This is the most directly relevant new method: it uses a SmolVLA base, action chunks, and only offline LIBERO demonstrations. It preserves the action head, so it fits our stated constraint. It is a credible high-upside candidate for 5/10/25, especially at 5 where candidate variance is high. First verify its public code and dataset conversion; then run only a narrow 5-shot smoke/pilot before committing to the full curve.

## Open questions

It adds a separate critic, TD learning, candidate sampling, and a nontrivial action-space geometry design. Its cost may be too high for the current budget, and its reported backbone/data preprocessing must be reproduced exactly enough to make transfer to our SmolVLA checkpoint meaningful.
