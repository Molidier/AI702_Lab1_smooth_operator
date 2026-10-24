# AI702 labs

Eight repository-based sessions, Lab 0 to Lab 7. Each lab has its own folder.
Run the labs locally using the supplied Python environment and HTML reports.

## Start here

Read this page, then open the README in the lab folder your TA has released.
Use Python 3.12 and the supplied locked course environment. Run these commands
from the folder containing `lab.py`:

```bash
uv sync --locked
uv run --locked python lab.py check
uv run --locked python lab.py baseline 0
uv run --locked python lab.py view 0 --stage baseline
```

The supplied baseline works before your implementations are complete. It contains
one deliberately inconsistent reporting unit. This is a teaching example.

## Files in each lab

| File in each lab folder | What you do |
| --- | --- |
| `README.md` | Read the tasks and function contracts |
| `student.py` | Write your own assessed implementation |
| `settings.json` | Use the TA-approved experiment configuration |
| `notes.md` | Write your expectation, observations and explanation |
| `test_labN.py`, `run.py` | Supplied tests and runner, do not edit |
| `results/` | Automatically saved runs and reports, do not edit |
| `submissions/` | Your separate core and transfer ZIPs, do not edit |

Only `student.py`, `settings.json` and `notes.md` are normally editable.
A signed late task can introduce `lab4/repair.py`. Your TA explains that interface
when the task is released. All other root files, `shared/` and `.integrity/` are
supplied course infrastructure. The student copy has no `staff/` directory.

## The same four commands for each lab

For Lab 4, for example:

```bash
uv run --locked python lab.py test 4
uv run --locked python lab.py run 4
uv run --locked python lab.py view 4
uv run --locked python lab.py package 4 --run-id CORE_RUN_ID
```

Replace 4 with your lab number and CORE_RUN_ID with the completed main run ID. Tests initially fail because `student.py` is
intentionally unfinished. The launcher prints the exact report and submission paths.
Every completed run gets a new folder. The default view/package command selects the
latest completed run. Use `--run-id ID` to select an earlier one. A package cannot be
overwritten. Finish your in-session notes before packaging. Before the deadline, a revision
requires a new run and package. Any short post-lab explanation is submitted separately
in the LMS. Never replace a frozen core or transfer package after its deadline.

`--smoke` is a small diagnostic run, excluded from assessment. External data use
explicit synthetic fixtures in smoke mode. Normal runs never substitute synthetic
data. Use CPU for the course baseline. Other devices require a separately validated
installation. Data are cached outside the repository at `~/.cache/ai702`, or at the
staff-provided `AI702_DATA_DIR`.

## Data and learning sequence

| Lab | Focus | Data |
|---|---|---|
| 0 | Python, NumPy, tensors, basic autograd, reproducibility | Breast Cancer Wisconsin |
| 1 | Perceptrons, MLP forward computation, interpretable decision regions | AND/OR/XOR, polygons, Iris |
| 2 | Perceptron learning, empirical error, smooth mismatch | Iris, Raisin |
| 3 | Output representations, losses, supplied gradients, explicit updates | Digits |
| 4 | Complete manual backpropagation, numerical checks, PyTorch comparison | Tiny arrays and Wine |
| 5 | Learning rate, momentum, batching, sample order, adaptive updates | Analytic surfaces and Dry Bean |
| 6 | Batch normalization, dropout, controlled ablation, checkpoints | CIFAR-10, flattened MLP |
| 7 | Shared scanning, manual correlation, CNN versus MLP | Same CIFAR-10 split |

Prepare the datasets specified by your TA before class. Only run the commands
needed for your session:

```bash
uv run --locked python lab.py data raisin
uv run --locked python lab.py data banknote
uv run --locked python lab.py data dry_bean
uv run --locked python lab.py data penguins
uv run --locked python lab.py data cifar10
```

Iris, Digits, Wine, and Breast Cancer Wisconsin are bundled with scikit-learn.
UCI tables and Penguins are small downloads. CIFAR-10 is the larger, one-time
archive (roughly 170 MB decimal); a training subset does not reduce that initial
archive download. Do not commit datasets, checkpoints, or generated results to Git.
The optional pretrained ResNet example requires another weights download.

UCI files use the UCI repository's dataset-specific licensing; see each source page.
Penguins is CC0. Dataset source URLs are in `shared/data.py`. Cache checksums detect
changes after the first download; they do not independently authenticate a first
copy. Use an approved staff-prepared cache when offline operation is needed.

## Pre-lab, in-lab, post-lab

Use the Expectation-Experiment-Explanation cycle in `notes.md`. Record the expectation
before running the comparison. Complete the code and main experiments in the lab.
The TA introduces any new repository interface before it is used. A late condition
is distributed separately. Afterwards, explain the actual evidence rather than write
a second large application. Unsupervised pre-lab quizzes are readiness activities,
not proof of independent work.

The runner saves the code and configuration used by each run, notes at its start,
raw outputs and a local HTML report. These local timestamps and hashes are not proof
of authorship or timing. Submit the expectation to the LMS or TA before experimenting
when it is assessed. Core and transfer have separate deadlines, commits and ZIPs.
Commit and push your source and notes, then submit the exact commit SHA and the ZIP
shown by `package`. Generated results are ignored by Git, so a commit alone is incomplete.
The TA captures core before releasing the transfer. Retain both stage submissions.

Labs 6 and 7 use validation data for all student comparisons. Staff evaluate the final
test split after capturing the approved submission. Do not tune against the test set.
Lab 7 trains a fresh supplied MLP with the same split, seed and training budget as its
CNN. Its configuration and actual checkpoints are contained in that Lab 7 run.

## Late transfer

After collecting your core submission, the TA gives you `transfer.json` and
`transfer.sig` for the current lab and release. They are absent from the initial
ZIP. Lab 4 also receives `repair.py`. Put the files beside `student.py` in the
corresponding lab folder. Do not modify the signed JSON or signature. The transfer
command verifies/displays the files and does not download or create them.

For Lab 4, first display the task:

```bash
uv run --locked python lab.py transfer 4
```

Record your expectation before running the assigned experiment:

```bash
uv run --locked python lab.py run 4 --stage transfer
uv run --locked python lab.py view 4 --stage transfer
uv run --locked python lab.py package 4 --stage transfer --run-id TRANSFER_RUN_ID
```

Replace TRANSFER_RUN_ID with the completed transfer run ID. The launcher applies
the signed settings automatically. Finish your observations before packaging.

Commit the released transfer.json and transfer.sig together with your code (and
repair.py for Lab 4), so the submitted commit contains the complete signed condition.
The condition is applied to the same experiment rather than merely displayed.
Unreleased transfer material is absent from student distributions.

## AI use and individual assessment

Read `AI_POLICY.md`. During assessed core work, transfer work and individual checks,
use only the approved resources and TA assistance. Do not use conversational AI,
AI completion or autonomous agents to implement, debug, rewrite or explain assessed
answers. Disable AI coding extensions in the course environment. Conceptual AI study
before the assessment is permitted only within the boundary stated in the policy.

Signed files and trusted staff verification detect particular inconsistencies. They
cannot stop use of another device or prove who wrote a program. Staff capture the
stage submissions and run the authoritative checks on a dedicated grading machine.
Controlled individual exercises after Labs 4 and 7 assess your own understanding.
No personal chat histories or AI-detector scores are used to decide authorship.

## Course releases

Your TA provides the release and private-repository instructions for each session.
Use the issued student ZIP. Ask your TA for help with installation, release updates
or missing assignment files. Preserve your work before applying any course update.
