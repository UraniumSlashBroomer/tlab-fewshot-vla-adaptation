# Research workflow

This is an ML research repository.

I need to solve test task for AI laboratory (task in research/task.md)
For research tasks, follow this loop:

1. Understand the hypothesis. (they are in research/hypothesis.md)
2. Inspect existing code and relevant previous experiments.
3. Run cheap sanity checks before expensive experiments.
4. Never launch a full training run if the smoke test fails.
5. Evaluate using the existing evaluation pipeline.
6. Compare against the relevant baseline.
7. Separate observations from interpretation.
8. Record the experiment and result.

# Before training

Always check:
- tensor shapes;
- trainable parameters;
- gradient flow; gradient exploding/vanishing;
- dataset/config correctness.

Prefer a short smoke run before full training.

# Experimental integrity

Do not silently change:
- dataset;
- evaluation protocol;
- seed set;
- training budget;
- optimizer;
- unrelated architecture components.

If one of these must change, explicitly report it.

# Commands

Training:
python train.py ...

Evaluation:
python eval.py ...

# Research behavior

Do not claim a hypothesis is supported based on training loss alone.

Do not interpret a single noisy run as strong evidence.

When uncertain about an implementation detail, inspect the code instead of guessing.

# Code quality

This is a research codebase. Optimize for readability,
simplicity, and ease of modification.

## General principles

Prefer simple, explicit code over defensive abstractions.

Avoid unnecessary:
- try/except blocks;
- input validation for internal code;
- fallback behavior;
- compatibility layers;
- wrapper classes;
- helper functions used only once.

Assume internal functions receive valid inputs unless
the boundary is explicitly user-facing or unsafe.

## Reuse

Before implementing new functionality:
1. search the repository for an existing implementation;
2. reuse or extend existing abstractions when appropriate;
3. do not duplicate logic.

If the same non-trivial logic appears more than once,
consider extracting it into a function.

## File organization

Do not grow unrelated functionality inside one file.

Prefer:
- models in model modules;
- datasets in dataset modules;
- evaluation in evaluation modules;
- visualization in visualization modules;
- experiment configs outside implementation code.

When a file becomes responsible for multiple unrelated concerns,
split it.

## Refactoring

When modifying existing code:
- preserve the existing architecture unless there is a reason to change it;
- avoid opportunistic large refactors;
- remove obsolete code created by your changes;
- do not leave duplicate implementations.

## Readability

Prefer straightforward PyTorch/Python.

Avoid clever abstractions unless they clearly reduce complexity.

Use descriptive names.

Comments should explain why, not restate what the code does.
