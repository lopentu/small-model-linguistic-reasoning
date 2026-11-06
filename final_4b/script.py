"""Final 4B submission entry point: direct JSON output and five candidate passes."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("GGUF_PATH", os.path.join(HERE, "model", "model-Q8_0.gguf"))
os.environ.setdefault("PROMPT_MODE", "direct")
os.environ.setdefault("N_SAMPLES", "5")
os.environ.setdefault("MAX_NEW", "700")
os.environ.setdefault("FEWSHOT", "0")
os.environ.setdefault("THINK", "0")
os.environ.setdefault("REFINE_ROUNDS", "0")
os.environ.setdefault("LLAMA_CTX", "6144")
os.environ.setdefault("TRANSCRIPT", "")

from v2.script import main  # noqa: E402

main()
