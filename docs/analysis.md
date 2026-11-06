# Analysis

## Local scoring

```bash
uv run python dev/score_local.py predictions.csv answers.csv
SPLIT_JSON=/absolute/path/split.json \
  uv run python dev/score_local.py predictions.csv answers.csv --split holdout --vs baseline.csv
```

Predictions use columns `id,pred`; `pred` is a JSON list. Gold uses `id,answer`;
`answer` is a Python-literal list, optionally with lists of accepted alternatives.
The scorer weights items uniformly, not by the official IOL point values.
`split.json` supplies search/holdout IDs and a `problem_of` map so queries from
the same problem are resampled together. Without that map, rows become bootstrap
units; do not describe such intervals as accounting for shared-problem dependence.

## Voting curves

`dev/local/e47_ncurve.py` expects user-supplied `transcript.jsonl` sample pools and
local `dev/mock-eval/fixtures/test.csv` and `answers.csv`. These are not released.

```bash
uv run python dev/local/e47_ncurve.py /absolute/path/saved-run --draws 8 --split search
```

Outputs go into the supplied run's `ncurve/` directory. The first sample must be
greedy and later samples stochastic for the stage-two interpretation to apply.
The first-N sequence and mean across sampled subsets are different estimators;
paired intervals for a particular selection are not intervals for the subset mean.
This tool reaggregates existing outputs; it does not run inference.

## Public/private leaderboard comparison

Place the independently obtained boards at `refs/leaderboard_final/public.tsv`
and `private.tsv`. Both TSVs require `rank`, `id` (team), `submission_datetime`, `score`, `chrf`,
`exact_match`, and `explanation_rate`. The private TSV also needs `submission`.

```bash
uv run python dev/leaderboard_delta.py
```

Rows match by submission timestamp. Public-best selection affects which systems
can be matched. Between-split variation mixes selection and split differences;
it is not a confidence interval for a component effect. No boards or personal
account metadata are redistributed here.

The voting tool uses aggregation from the later `solver/deploy/v2` tree, with
`ASSIGNMENT=0`, `DETLANES=0` and `VOTE_WEIGHT=none` forced for plain majority
voting. It does not reproduce the archived final-4B pipeline. Greedy-first sample
ordering is a caller-supplied requirement, not inferred from saved text. Sampling
temperature is determined by the run that produced the transcript.

Leaderboard analysis retains the study's named baseline and public-score threshold
of 0.18 for its descriptive selection comparison. That subsection is skipped if
the baseline or qualifying scores are absent. Equal score triples do not prove
that artifacts are identical or that the scorer is deterministic. No causal or
truncation conclusion is inferred automatically.

The voting loader preserves sample permutations and appends finalize events to
their recorded slots before selecting solve samples. Stage-two intervals concern
draw 0 only; they are labelled separately from the mean across random selections.
