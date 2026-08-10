---
name: read-paper
description: >
  Use when reading, analyzing, comparing, or extracting implementation
  details from research papers relevant to this repository.
---

# Targeted paper reading

When the user asks about a paper:

1. Find the paper in `research/papers`.
   Use `research/papers/metadata.md` to locate it and understand its relevance.

2. Identify the user's concrete research question.

3. Read only what is needed to answer it:
   - abstract;
   - relevant method sections;
   - relevant equations;
   - architecture description;
   - relevant experiments or ablations.

4. Inspect the appendix only when necessary.

5. If an implementation detail is ambiguous:
   - inspect the official implementation, if available;
   - prefer official code over guessing from incomplete descriptions.

6. Report:
   - how the relevant method works;
   - tensor/interface requirements;
   - training objective;
   - implications for this repository;
   - required architectural changes, if applicable;
   - uncertainties or missing implementation details.

Clearly distinguish:
- claims made by the paper;
- observations from official code;
- your own interpretation.

# Additional context

When broader understanding of a paper is needed:

1. Read `research/papers/metadata.md`.

2. Check `research/papers/summarized/<paper-name>.md`.

3. If a summary exists:
   use it for orientation and general context.

   For exact equations, architecture details, experimental numbers,
   or implementation decisions, verify against the original paper
   or official code.

4. If no summary exists and deep understanding is actually required:
   read the paper in depth, covering:
   - motivation;
   - method;
   - architecture;
   - objective;
   - training setup;
   - experiments;
   - ablations;
   - limitations.

5. Write a concise reusable summary to:
   `research/papers/summarized/<paper-name>.md`

Do not create a full summary when targeted reading is sufficient.

Summary should be in paper-style:
```
# Paper name

## Core idea

...

## Architecture

...

## Objective

...

## Important implementation details

...

## Relevant to our project

...

## Open questions

...
```
