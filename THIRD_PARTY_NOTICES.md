# Third-party material

No model weights, dataset rows, annotated solutions, or vendored Python/CUDA
libraries are included. Obtain these from their original distributors and comply
with their terms.

`dev/iolbatch/iolbatch.cpp` was adapted from the llama.cpp parallel example.
llama.cpp is distributed under the MIT license. The upstream copyright/license notice from tag b9964 is included at
`dev/iolbatch/LICENSE` and must accompany distributed copies.
The project's original code is released under the MIT License (see `LICENSE`).

The refinement prompt was adapted from the LOBSTER multi-round solver
(Lian et al., ROCLING 2025). Prompt text is retained with the inference code;
cleanup changes comments and docstrings, not the model inputs.
