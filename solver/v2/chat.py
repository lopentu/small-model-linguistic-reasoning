r"""Model-agnostic chat-template rendering with explicit thinking control.

Both Gemma-4 and Qwen3.6 GGUFs carry a Jinja chat template that honors
`enable_thinking` (verified 12 Jul against the actual GGUF metadata):
  - enable_thinking=False -> the assistant turn opens with a CLOSED empty think
    block (Qwen `<think>\\n\\n</think>`, Gemma equivalent) => non-thinking
  - enable_thinking=True / omitted -> the block is left OPEN => the model reasons

The bindings' `create_chat_completion` cannot pass `enable_thinking` (no such
param, no **kwargs), and — critically — the templates DEFAULT to thinking-on
when the flag is absent. So we render the model's own template ourselves with an
explicit flag and feed the string to `create_completion`. This replaces the
Gemma-only `<|think|>`-prefix hack and makes Qwen controllable too.

NOTE: `--reasoning-budget` (capping thinking LENGTH) is a llama-cli feature, not
a bindings one; it lives in the CLI-driver backend, not here. This module only
turns thinking on/off.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .prompts import Messages


def _raise(msg: str = "") -> None:
    raise RuntimeError(msg)


def render_chat(template: str, messages: Messages, *, think: bool) -> str:
    """Render a GGUF Jinja chat template to a prompt string with thinking on/off.

    Pure and unit-testable (no model). `add_generation_prompt=True` so the output
    ends at the assistant turn ready for completion.
    """
    import jinja2  # vendored on the eval image

    env = jinja2.Environment(loader=jinja2.BaseLoader(), autoescape=False)  # noqa: S701 — prompt text, not HTML
    env.globals["raise_exception"] = _raise
    env.filters.setdefault("tojson", lambda v, **_k: __import__("json").dumps(v))
    return env.from_string(template).render(
        messages=messages,
        add_generation_prompt=True,
        enable_thinking=think,
    )
