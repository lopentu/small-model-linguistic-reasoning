"""IOL-AI 2026 solver v2 — a stage-pipeline rewrite of solver/script.py.

Layering (each module readable alone, no cycles):

    textproc      pure string functions (parsing, normalization, voting)
    prompts       prompt text + message builders
    validators    deterministic answer checks (feed the refine stage)
    config        env-var configuration
    budget        wall-clock accounting and token-budget scaling
    transcript    append-only JSONL log per generation (debug + resume)
    backends      Backend protocol + mock / llama_cpp / transformers drivers
    pipeline      ProblemState, Pass protocol, concrete passes, scheduler
    script        entry point wiring the above together

Adding a step = writing one Pass class and inserting it into the list
returned by `pipeline.default_passes` (see dev/test_solver_v2.py for a
worked example). The submission repo root runs a thin script.py:

    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from v2.script import main
    main()
"""
