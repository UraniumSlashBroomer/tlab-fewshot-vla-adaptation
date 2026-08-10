---
name: run-ml-experiment
description:
  Use this skill when running, modifying, evaluating, or comparing
  ML/DL experiments. Apply it for training runs, ablations,
  architecture changes, finetuning experiments, and baseline comparisons.
---

# ML Experiment Workflow

## Goal

Run controlled and reproducible ML experiments.

## Workflow

1. Identify the hypothesis.
2. Identify the baseline.
3. State:
   - what changes;
   - what stays fixed;
   - primary evaluation metric.
4. Inspect relevant implementation before modifying it.
5. Make the smallest necessary code/config change.
6. Run cheap sanity checks.
7. Run the experiment.
8. Evaluate using the standard evaluation pipeline.
9. Compare against the baseline.
10. Record observations separately from interpretation.

## Before expensive training

Check:
- input/output tensor shapes;
- trainable parameters;
- gradient flow; gradient blowing/vanishing;
- configuration;
- dataset;
- checkpoint loading.

## Coding rules

Follow repository coding rules from AGENTS.md.
Do not introduce unrelated refactors.

## Completion

Report:
- experiment configuration;
- result;
- comparison with baseline;
- important caveats;
- recommended next experiment.

for every heavy experiment create json file with metrics.
