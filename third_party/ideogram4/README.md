# Files from ideogram-oss/ideogram4 (Apache License 2.0)

Copied unmodified from https://github.com/ideogram-oss/ideogram4 at commit 990fe1c
(2026-11-01). The licence is in `LICENSE.md` next to this file. This licence covers
these source files only; the Ideogram 4 MODEL WEIGHTS are under the Ideogram 4
Non-Commercial licence (see the README of crispz-ideogram).

| File here | Upstream path | Used by |
|---|---|---|
| `caption_verifier.py` | `src/ideogram4/caption_verifier.py` | `cz_caption.validate` (the reference schema checks) |
| `magic_prompt_v1.txt` | `src/ideogram4/magic_prompt_system_prompts/v1.txt` | `cz_magic` (Ollama engine, optional "v1" system prompt) |

`cz_caption.reorder_keys` re-implements `reorder_caption_keys` from upstream
`src/ideogram4/magic_prompt.py` (same rules, same licence).
