# Lab 0: Python, NumPy, PyTorch, and repository refresher

## Overview

Run a supplied baseline, implement small array/tensor operations, and identify a
reported accuracy whose number and units disagree. The baseline uses the built-in
Breast Cancer Wisconsin data purely as a small binary teaching example.

## Pre-lab preparation

Review Python functions, classes, dictionaries, slicing, array shapes, matrix
multiplication, broadcasting, Git, and pytest. Install uv and Python 3.12 before class.
In notes.md, state how an accuracy reported as a fraction should be interpreted.

## Readiness check

1. X has shape (4, 3), W has shape (2, 3), and b has shape (2,). What is the shape of X @ W.T + b?
2. A tensor has shape (2, 3, 4, 5) in NCHW order. What is its shape after permute(0, 2, 3, 1)?
3. If three of four predictions are correct. What is accuracy as a fraction and as a percentage?

Answer these questions before class and bring any questions to your TA.

## Before implementation

Your TA will introduce the repository commands, callable Python classes, NumPy arrays,
PyTorch tensors, data types, memory layout.
The supplied classifier is a demonstration. You do not implement its training loop.

## Already provided in the repository

`shared/data.py` returns fixed, stratified arrays, labels, and sample IDs, with
standardization fitted to the training split only. `run.py` supplies a complete tiny
classifier and loop. Its CSV contains sample_id, target, prediction, and class scores.
The shared plotting code consumes the loss list and labels to produce a curve and
confusion matrix. No plotting, downloading, or baseline model code is student work.

## Your implementation

Implement `select_columns`, `standardize_columns`, `linear_scores_numpy`,
`LinearTransform`, `torch_linear_scores`, `rearrange_nchw_to_nhwc`, and `accuracy_fraction`. Preserve batch dimensions.
Use the supplied statistics rather than fitting a scaler to validation examples.
The autograd task must use backward/grad, not a handwritten derivative.

## In-lab work

1. Run `uv run --locked python lab.py baseline 0` and inspect its CSV/report before implementing anything.
2. Complete and test the NumPy operations and the small Python class.
3. Complete equivalent tensor operations. Test a non-contiguous tensor and compare results.
4. Run `uv run --locked python lab.py run 0`. Recompute the metric from the actual labels/predictions.
5. Compare the reported units with your recomputed result. Record the inconsistency.
6. View the plots, commit code and notes, and package the evidence.

## Main experiment commands

Run these commands from the folder containing `lab.py`:

```bash
uv run --locked python lab.py check 0
uv run --locked python lab.py test 0
uv run --locked python lab.py run 0
uv run --locked python lab.py view 0
```

The starter functions intentionally raise `NotImplementedError`. Complete your
implementation before expecting the tests and main run to pass. Finish your
observations in `lab0/notes.md`, then package the selected main run:

```bash
uv run --locked python lab.py package 0 --run-id CORE_RUN_ID
```

Replace `CORE_RUN_ID` with the ID of your completed main run. Commit and push your
code, settings and final notes to your assigned private repository. Use
`git rev-parse HEAD` to obtain the full commit SHA. Submit the SHA, run ID and ZIP
in the LMS before starting the additional experiment. A Git push alone does not
submit the generated results.

Results are in `lab0/results/STAGE/RUN_ID/` and submission ZIPs are in
`lab0/submissions/`. Use `uv run --locked python lab.py view 0 --no-open` if the browser does
not open automatically. Keep results and existing submission ZIPs unchanged.
A revision to code or settings requires a new run and a new package before the
deadline. Smoke runs and supplied demonstrations cannot be submitted.

## Additional experiment

After collecting your main submission, the TA will provide `transfer.json` and
`transfer.sig` separately. They are intentionally absent from the initial ZIP.
Copy both into `lab0/`, beside `student.py`. Do not modify either signed file.
The `transfer` command verifies and displays the task. It does not create or
download the files. If either file is missing, ask the TA before continuing.

Repeat the accuracy audit with the assigned settings. Use the new prediction CSV
and settings_used.json to explain what changed and whether the reporting issue remains.

```bash
uv run --locked python lab.py transfer 0
```

Record and submit your expectation as directed before running the experiment.
The launcher applies the signed settings automatically. Do not copy the overrides
into `settings.json`.

```bash
uv run --locked python lab.py run 0 --stage transfer
uv run --locked python lab.py view 0 --stage transfer
```

Finish your observations in `notes.md`, then package the transfer run:

```bash
uv run --locked python lab.py package 0 --stage transfer --run-id TRANSFER_RUN_ID
```

Replace `TRANSFER_RUN_ID` with the ID of the completed transfer run. Commit and push
code, settings, final notes and both signed transfer files.
Submit the full commit SHA, run ID and transfer ZIP in the LMS. Keep the core
submission unchanged.

## Short post-lab explanation

Explain one NumPy/PyTorch difference, one shape or memory-layout issue, and the
accuracy-unit inconsistency. Refer to the exact reported and recomputed values.
Do not make clinical claims about the model.

## Evidence and tests

`validation_predictions.csv`, `raw.npz`, `loss.png`, `confusion.png`, `metrics.json`,
`details.json`, `run.json`, and `index.html`. The public tests cover array/tensor shapes and values, the autograd result and
accuracy units. The task contracts also require input non-mutation.
