# Training and trace preparation

The expert-review corpus and annotations are not distributed. Providing the
builders does not make the trace-supervision experiment independently runnable
without those inputs. Compression outputs were prepared separately; the builder
checks and assembles them, it does not generate expert corrections or compress
traces automatically.

## Trace data

Set `REVIEW_CORPUS` to your `reviewed_only.jsonl` input. `dev/sft/build_dataset.py`
reads user-supplied files under `dev/sft/work/` and writes `dev/sft/dataset/`.
Both locations are ignored by Git. See the subcommands and their schema in source:

```bash
uv run python dev/sft/build_dataset.py --help
```

The delete-only validator checks ordered token containment, tier length limits,
and the structure of the compression output. It does not verify linguistic
correctness. The test suite includes valid and invalid synthetic subsequences.

## Instruction-data builders

The historical E36/E39/E40 scripts assemble mixtures from direct-answer training
rows plus multilingual or Linguini data. Set `IOL_TRAIN_PATH` to a local JSONL
of direct-answer rows. They no longer default to an author-owned private dataset.
The multilingual builder downloads its source dataset when invoked; no data is
included here. Its source allowlist records historical choices, not a current
legal determination about every upstream dataset version.

```bash
MIST_REVISION=your-dataset-commit IOL_TRAIN_PATH=/absolute/path/direct.jsonl \
  uv run --with huggingface_hub python dev/sft/build_e36_mist.py --ratio 4
```

These mixture builders are not the final multilingual-only training-file selector.
For the final multilingual-only arm, supply the actual selected instruction file
to training. Do not infer the final training data from a builder's filename.
Linguini experiments require the original locally obtained fixture CSVs and
`dev/mock-eval/split.json`; never publish the fixture text in this repository.
Uploads occur only with `--upload` and an explicit `DATASET_REPO` destination.

## LoRA training

`train.py` retains its inline GPU dependencies. `TRAIN_PATH` accepts a local
training JSONL; otherwise `DATASET_REPO` and `TRAIN_FILE` identify a Hub dataset.
Rows use `messages`, `completion`, and `problem_id`; builders also record tier
metadata. GPU dependencies must match the model architecture. They have not been
installed or validated by this release's CPU tests.

```bash
MODEL_ID=unsloth/gemma-4-26B-A4B-it TRAIN_PATH=/absolute/path/train.jsonl OUT_REPO= \
  LORA_R=32 LR=1e-4 EPOCHS=6 MAX_LEN=16384 \
  uv run dev/sft/train.py
```

An empty `OUT_REPO` disables Hub checkpoint pushes. Weights, compute, and data
access are supplied by the user. The script's historical default model is the
26B arm; always set `MODEL_ID` explicitly. `merge_verified.py` checks adapter
application before exporting merged weights. Review its environment variables
before use; it is not run by the release checks. Historical training dependencies
were not frozen in this export, so exact training reproduction is not claimed.

The example above targets the historical Gemma arm. It is not a generic model
recipe. For another model, verify its chat template and EOS token; `EOS_OK=1`
bypasses the Gemma EOS guard, and `THINK_OFF_FINGERPRINT` must be set to the
actual nonempty no-thinking template marker. Do not guess this marker. This
export does not provide a validated Qwen training environment or recipe.

MIST downloads require an explicit `MIST_REVISION`. The historical dataset commit
was not recorded here; selecting a revision now does not establish that it was
used in the paper. Mixture uploads require `--upload`, `DATASET_REPO`, and an
existing private Hub dataset repository. No visibility change is performed.

External training logs are disabled by default. Set `REPORT_TO=wandb` explicitly
to enable W&B; possessing a `WANDB_API_KEY` alone does not enable it. Checkpoint
uploads require a nonempty `OUT_REPO` and a private destination.

`merge_verified.py` requires `ADAPTER_REPO` and uses `/data` for its checkout,
merged weights and GGUF outputs. It can download adapters and base weights.
Uploads are disabled by default; `UPLOAD=1 OUT_REPO=owner/repo` explicitly enables
them and requires a private destination. `UPLOAD_MERGED=1` selects the
weights-only path, even with uploads disabled. Otherwise it converts and quantizes
using the pinned `LLAMA_TAG`; a missing tag fails without falling back to HEAD.
GPU merging and conversion have not been exercised by the CPU suite.

Upload-enabled training/merging uses the adjacent `release_io.py` module. Run
from a checkout; a remote single-file job must also package that module. The
inline dependency list does not bundle repository-local Python files.
