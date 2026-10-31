"""The pruning of the text encoder, and the VRAM budget that depends on it.

FLUX.2 does not read the LLM's output: it stacks the hidden states of three INTERMEDIATE
layers (hence a context_embedder 3 x hidden wide). Everything that comes after the last
layer read -- eight blocks of a Qwen3-8B and a projection onto 152,000 tokens -- is
computed for every image then thrown away. Since hidden_states[k] is the output AFTER k
blocks, removing them is EXACT, not approximate: checked bit for bit on the real model
(15.3 -> 11.2 GB, encoding 5.0 -> 3.2 s, torch.equal true on the three layers read).

These tests lock down the two traps met while writing it:

1. THE METHOD'S NAME. The first version looked for `_get_qwen_prompt_embeds`;
   diffusers calls it `_get_qwen3_prompt_embeds` here. So the pruning never fired
   -- and it SAID so, but nobody reads a log when a lower figure is already there and
   wrong. We now look for the method by its PARAMETER.

2. A BUDGET ON AN INTENTION. The worst consequence of point 1: the VRAM budget deducted
   the pruning's 4 GB without checking that it had happened. Measured: 29.7 GB announced,
   32.3 GB really resident, 0.0 GB free -- and under Windows it does not even crash, it
   spills into shared memory and the render collapses in silence. So the budget now
   follows the _ENCODER_TRIMMED flag only, never TRIM_TEXT_ENCODER.

Run:  .venv/Scripts/python tests/test_encoder_trim.py

"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import cz_pipeline as P


class _FakeEncoder(torch.nn.Module):
    """The bare minimum: .model.layers, .lm_head, some parameters to count."""

    def __init__(self, n=36, width=8):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(
            [torch.nn.Linear(width, width) for _ in range(n)])
        self.lm_head = torch.nn.Linear(width, 512)


def _fake_pipe(layers=(9, 18, 27), n=36, method="_get_qwen3_prompt_embeds"):
    """A pipe whose CLASS carries the method, as at diffusers."""
    def _embeds(prompt, tokenizer, text_encoder, hidden_states_layers=layers):
        return None

    cls = type("FakePipe", (), {method: staticmethod(_embeds)})
    p = cls()
    p.text_encoder = _FakeEncoder(n=n)
    return p


def test_the_read_layers_are_found_by_parameter_not_by_name():
    """The exact trap: the method is called _get_qwen3_..., not _get_qwen_...."""
    for name in ("_get_qwen3_prompt_embeds", "_get_qwen_prompt_embeds",
                 "_get_some_future_name_embeds"):
        p = _fake_pipe(method=name)
        assert P._encoder_layers_used(p) == 27, name
    print("OK test_the_read_layers_are_found_by_parameter_not_by_name")


def test_an_unreadable_signature_trims_nothing():
    """With no reliable information, we keep EVERYTHING: silently wrong embeddings would
    be infinitely worse than a few wasted gigabytes."""
    cls = type("NoSuchMethod", (), {})
    p = cls()
    p.text_encoder = _FakeEncoder()
    assert P._encoder_layers_used(p) is None
    P._trim_text_encoder(p)
    assert len(p.text_encoder.model.layers) == 36
    assert P._ENCODER_TRIMMED is False, "the budget must deduct nothing"
    print("OK test_an_unreadable_signature_trims_nothing")


def test_trimming_keeps_exactly_the_blocks_that_are_read():
    p = _fake_pipe()
    kept = list(p.text_encoder.model.layers)[:28]
    P._trim_text_encoder(p)
    assert len(p.text_encoder.model.layers) == 28, len(p.text_encoder.model.layers)
    # they really are the SAME objects, in order: we cut, we do not rebuild
    assert all(a is b for a, b in zip(p.text_encoder.model.layers, kept))
    assert isinstance(p.text_encoder.lm_head, torch.nn.Identity)
    assert P._ENCODER_TRIMMED is True
    print("OK test_trimming_keeps_exactly_the_blocks_that_are_read")


def test_trimming_twice_changes_nothing():
    p = _fake_pipe()
    P._trim_text_encoder(p)
    P._trim_text_encoder(p)
    assert len(p.text_encoder.model.layers) == 28
    print("OK test_trimming_twice_changes_nothing")


def test_the_budget_follows_the_deed_not_the_intent():
    """The bug that cost a full card on klein: 4 GB deducted from a pruning never done.
    On FLUX.3 the pruning is never done (Qwen3-VL, see _trim_text_encoder), so the flag
    must change NOTHING in the budget, and the budget is the precision's figure."""
    old = (P.BASE_REPO, P._ENCODER_TRIMMED, P.ZIMAGE_TRANSFORMER, P.PRECISION)
    P.BASE_REPO = "black-forest-labs/flux-3-image-klein"
    P.ZIMAGE_TRANSFORMER = None
    P.PRECISION = "bf16"
    try:
        P._ENCODER_TRIMMED = False
        whole = P._base_vram_need_gb()
        P._ENCODER_TRIMMED = True
        trimmed = P._base_vram_need_gb()
        assert whole == trimmed, (whole, trimmed)
        assert abs(whole - (P._FLUX3_REST_GB + P._FLUX3_TRANSFORMER_GB["bf16"])) < 1e-6, whole
        P.PRECISION = "nvfp4"
        assert P._base_vram_need_gb() < whole
    finally:
        P.BASE_REPO, P._ENCODER_TRIMMED, P.ZIMAGE_TRANSFORMER, P.PRECISION = old
    print("OK test_the_budget_follows_the_deed_not_the_intent")


def test_the_trim_on_flux3_is_announced_not_silent():
    """On a Flux3Pipe the function has nothing it can prune yet; it must say so, and the
    flag that feeds the budget must stay False."""
    import cz_flux3
    lines = []
    real_log = P._log
    P._log = lambda m: lines.append(str(m))

    class _Pipe(cz_flux3.Flux3Pipe):          # no modular pipeline needed for this path
        def __init__(self):
            pass
    old = (P._ENCODER_TRIMMED, P.TRIM_TEXT_ENCODER)
    try:
        P.TRIM_TEXT_ENCODER = True
        P._ENCODER_TRIMMED = True
        P._trim_text_encoder(_Pipe())
        assert P._ENCODER_TRIMMED is False
        assert any("NOT trimmed" in m and "Qwen3-VL" in m for m in lines), lines
    finally:
        P._log = real_log
        P._ENCODER_TRIMMED, P.TRIM_TEXT_ENCODER = old
    print("OK test_the_trim_on_flux3_is_announced_not_silent")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("All text-encoder trim tests passed.")
