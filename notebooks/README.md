# Report notebooks

One notebook per question (`Q1_report.ipynb`, `Q2_report.ipynb`, `Q3_report.ipynb`), used **only**
to turn the CSV/TXT files already produced under `results/` into the figures, tables and
commentary that go into the PDF report. Splitting per question (instead of one shared notebook)
keeps the team from stepping on each other's cells/outputs at the same time.

## What goes in here

- Loading `results/<question>/*.csv` and `*_summary.txt`.
- Report-specific plot formatting (labels, combined subplots, styling) built on top of
  `src/plotting.py`.
- The written discussion/commentary for each sub-question, as markdown cells, so the whole
  team can see and edit the draft in one place before it's copied into the report.

## What does NOT go in here

- Model logic (variables, objective, constraints) -- that stays in `src/model.py`.
- Data loading/scenario logic -- that stays in `src/data_loader.py` / `src/scenarios.py`.
- Anything that needs to be reproducible with a single command for grading -- that's
  `main.py` (see README.md at the repo root, section 5).

If you find yourself writing model code in a notebook cell, it belongs in `src/` instead --
import it from there and call it, don't redefine it here.

## Workflow

1. Finish/update the model for a question in `src/`.
2. From the repo root: `python main.py --question <case> --scenarios` to (re)populate `results/`.
3. Open the matching notebook and re-run the loading cells.
4. Fill in the `# TODO` code cells (figures/tables) and the `_TODO_` markdown cells (discussion)
   for the sub-questions you just solved.
5. Before committing: `Kernel -> Restart & Run All` once to check it still runs top to bottom,
   then `Edit -> Clear Outputs` (outputs make git diffs noisy and can go stale vs. the latest
   `results/`).

## Section headers

Each notebook's sections follow the sub-question lettering from
`course_materials/Assignment_1_Instructions.html` (the grading table), e.g. `## (d) -- ...` for
Question 1.(d). Purely theoretical sub-questions (formulation, duality, KKT, qualitative
hypotheses) are left as a placeholder only -- they're answered as text in the report, not with
a figure.
