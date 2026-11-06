# Small-Model Linguistic Reasoning under Compute Constraints

Code accompanying the paper **Small-Model Linguistic Reasoning under Compute Constraints**
(Da-Chen Lian, Chunki Lim, Kai-Yang Peng, Nai-Wei Lu, Pei-Yang Lin and Shu-Kai Hsieh; ROCLING 2026),
from the [LOPE Lab](https://lope.linguistics.ntu.edu.tw/) at National Taiwan University.
The paper will appear in the ROCLING 2026 proceedings.

The submitted 4B model is on Hugging Face:
[richardlian/iolai-2026-4b-mist-oldv2](https://huggingface.co/richardlian/iolai-2026-4b-mist-oldv2/tree/ba789db20d369f197cae6ac4383199cf0aac6c2f)
(the submitted revision). The dataset, expert annotations and generated solutions are not included.

This repository contains inference, supervised fine-tuning, trace-validation,
and analysis code. It includes the final 4B submission's Python files and two
separate development solver trees. These are different versions with different
purposes; they must not be substituted when attributing a reported result.

## Start here

Install Python 3.11 or newer and [uv](https://docs.astral.sh/uv/), then run:

```bash
uv sync --frozen
uv run python tests/run_checks.py
```

The checks use synthetic inputs. They need neither model weights nor a GPU and
are not a rerun of the paper's experiments. `uv.lock` pins the release test
environment, not the historical competition or GPU training environment.

## What is included

| Directory | Purpose |
| --- | --- |
| `final_4b/` | Final 4B submission entry point and solver modules; original file identities checked against the published model repository, then comments/docstrings cleaned without changing executable logic |
| `solver/deploy/v2/` | Later development code for voting, confidence weighting, and answer delivery |
| `solver/v2/` | Separate experimental solver, including grammar retrieval |
| `dev/sft/` | Training-data builders, delete-only trace validation, LoRA training, and verified adapter merging |
| `dev/score_local.py` | Local exact-match/chrF scoring and paired problem-level bootstrap |
| `dev/local/e47_ncurve.py` | Reaggregation of saved samples into voting curves |
| `dev/leaderboard_delta.py` | Matching public/private leaderboard rows and calculating score shifts |
| `dev/iolbatch/` | Optional C++ parallel-generation driver source |
| `dev/test_*.py`, `tests/` | CPU checks for parsing, aggregation, delivery, retrieval, and release interfaces; some historical tests need omitted assets |

See [inference](docs/inference.md), [training](docs/training.md), and
[analysis](docs/analysis.md) for inputs and commands. [provenance.json](provenance.json)
records the source and release SHA-256 of every copied file and lists modifications.

## Reproduction limits

This is a code release. It contains no expert-review corpus, annotations,
Linguini problem text, model weights, generated answers, private evaluation labels,
or saved experiment transcripts. Some scripts therefore require inputs that are
not distributed here. The historical private evaluation cannot be rerun through
this repository. CPU tests establish software behavior on synthetic examples;
they do not reproduce the reported scores or verify GPU compatibility.

The local scorer weights answer items uniformly because the official point
weights were unavailable locally. It is not numerically interchangeable with
the official weighted leaderboard score. Public leaderboard comparisons were
exploratory; a local component comparison is not a private-split ablation.

The original final 4B files were checked against the published submission code on
2026-10-02. The released files retain the same executable logic after comment cleanup.
The source check verifies original file contents, not the model weights or the
competition's frozen revision identity. See `final_4b/source.json`.

The code in this repository is released under the [MIT License](LICENSE). Dependencies,
model weights, and external datasets retain their own terms. The optional C++
driver derives from llama.cpp; its upstream license is included with the source and must accompany distributed
builds. See [third-party notices](THIRD_PARTY_NOTICES.md).

Historical checks outside the default suite are retained for reference:
`dev/test_n_items.py` needs the omitted `solver/script.py`;
`dev/test_assignment.py` needs omitted fixtures and exits when they are absent;
`dev/test_numbrute.py` needs the omitted numeral solver. These are not passing
release checks. Use `tests/run_checks.py` for the supported CPU suite.

To verify the archived final source separately, obtain the upstream Python files
and run `uv run python tests/verify_upstream.py /path/to/upstream`.
This checks original Git blob IDs and executable AST equality after removing
docstrings. It does not download files or check model weights.
