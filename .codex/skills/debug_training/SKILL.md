---
name: debug-training
description: >
  Use when an ML/DL training run behaves unexpectedly:
  loss does not decrease, gradients are missing, training diverges,
  metrics collapse, NaNs appear, or a model cannot overfit simple data.
---

# Debugging workflow

Debug from cheapest checks to most expensive.

1. Reproduce the problem.

2. Inspect one batch:
   - shapes;
   - dtype;
   - ranges;
   - targets;
   - masking/alignment.

3. Inspect forward pass:
   - inputs;
   - outputs;
   - loss.

4. Check gradients:
   - required parameters have gradients;
   - frozen parameters do not;
   - gradient magnitudes are reasonable.

5. Check parameter updates.

6. Attempt to overfit one batch.

7. Attempt to overfit a tiny dataset.

8. Only after these checks investigate:
   - optimizer;
   - scheduler;
   - initialization;
   - architecture;
   - dataset scale.

Do not make multiple speculative fixes simultaneously.

Change one suspected cause at a time.
