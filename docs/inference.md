# Inference

`final_4b/script.py` sets the final 4B entry-point defaults. Its executable configuration sets `N_SAMPLES=5`. The pipeline
starts with a greedy answer and schedules four optional stochastic answers.
Stochastic temperature is 1.0. Budget exhaustion can skip optional passes.
These are configuration facts, not a guarantee of five completed answers per row.

Inputs are a local GGUF model and a CSV with columns `id`, `context`, `query`,
`work_lang`, `task_lang`, `task_type`, and `eval_type`. The output is
`submission.csv` in the current directory, with JSON answer lists in `pred`.

```bash
mkdir -p outputs/final-4b
cd outputs/final-4b
GGUF_PATH=/absolute/path/model.gguf TEST_CSV=/absolute/path/test.csv \
  ../../.venv/bin/python ../../final_4b/script.py
```

Before running this command, install a llama-cpp-python build that supports your
GGUF architecture and hardware into the environment. The release test environment
does not install GPU libraries. The historical package used vendored dependencies;
those binaries are not included here. `N_GPU_LAYERS=0` selects CPU offloading
behavior, but does not reproduce historical speed or the T4 time budget.

The entry point defaults to a 30-minute budget. Set `TIME_LIMIT_S`, `N_SAMPLES`,
or `MAX_NEW` only when intentionally changing the configuration. Keep output
runs in different directories to avoid overwriting submissions.

For experiments, `solver/deploy/v2` is a later tree, with additional aggregation
options. `solver/v2` contains retrieval. Their tests run independently so Python
cannot silently reuse the wrong imported `v2` package. The optional parallel
backend needs a separately built `llama-iolbatch`; its source is provided in
`dev/iolbatch/`, but it is not needed for the final 4B bindings path.

The experimental `solver/v2` tree also contains an optional numeral pre-pass.
Its separately vendored number solver is not part of this release and the
pre-pass therefore returns without changing answers. Set `NUMBRUTE=0` explicitly
when using this tree. The historical `dev/test_numbrute.py` requires that omitted
package and is not included in the default checks; running it here fails its
positive-solution assertion. Neither this feature nor its tests apply to
`final_4b/`.

## Sampling and dependency limits

The archived final bindings path explicitly sets temperature 0 for greedy and
1.0 for stochastic passes, and top-p 1.0 and 0.95 respectively. It does not set
`repeat_penalty`, `top_k`, or `min_p`; those come from llama-cpp-python defaults.
The inspected archived vendor directory identifies llama-cpp-python 0.3.33.
Its `create_completion` signature defaults to `repeat_penalty=1.0`, `top_k=40`
and `min_p=0.05`. These were checked in the archived library itself; an older
development comment claiming 1.1 does not describe this archive. The archive's
identity with the competition's frozen vendor files is unverified. A fresh PyPI
installation may differ from this vendored build.
The final code is preserved, so these defaults have not been silently overwritten.

Install hardware-specific bindings with `uv pip install` into the prepared
`.venv` after syncing. A later exact `uv sync --frozen` removes packages outside
the lockfile; reinstall the bindings afterward or use a separate inference
environment. The lockfile covers only CPU checks.

## Optional grammar retrieval

This applies only to `solver/v2`. Supply your own LanceDB grammar collection,
Glottolog checkout and metadata JSON. None is included here.

```bash
uv run --with lancedb python dev/gramquery_extract.py \
  --lancedb /path/to/db --glottolog /path/to/glottolog \
  --meta /path/to/glottocode_to_glottolog_metadata.json --out data/gramquery
.venv/bin/python dev/gramquery_embed.py \
  --chunks data/gramquery/chunks.jsonl --model /path/to/embedder.gguf \
  --out data/gramquery/vectors.f16.npy
```

Embedding requires separately installed llama-cpp-python. Enable retrieval with
`GRAMQUERY=1 GRAMQUERY_DIR=/absolute/path/data/gramquery` and
`GRAMQUERY_EMBED_MODEL=/absolute/path/embedder.gguf` when calling the experimental
solver. Missing inputs disable retrieval; check its logs. This is not a final-4B
configuration and has not been validated on a GPU in this prepared repository.
