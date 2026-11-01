"""cz_ideogram.IdeogramPipe on a tiny random Ideogram 4 (CPU, ~30 s, no gated download).

Lot 1: text to image through crispz's calling convention, the guidance schedule kept in
shape for any step count, size fitting, callback and Stop, the named refusals of the
paths lot 3 / lot 4 will bring (img2img, inpaint, edit), the negative prompt dropped,
and the family detection the LoRA and checkpoint guards rely on.
Run: python -m pytest tests/test_ideogram_engine.py -q
"""
import os
import sys

import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cz_ideogram  # noqa: E402
from ideogram_tiny import tiny_pipe  # noqa: E402

_PIPE = None


def _pipe():
    global _PIPE
    if _PIPE is None:
        _PIPE = tiny_pipe()
    return _PIPE


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


def test_txt2img_size_and_callback():
    seen = []
    out = _pipe()(prompt="a poster that says HELLO", width=256, height=512,
                  num_inference_steps=3, generator=_gen(),
                  callback_on_step_end=lambda p, i, t, kw: seen.append(i) or kw)
    assert out.images[0].size == (256, 512)
    assert seen == [0, 1, 2]


def test_deterministic_seed():
    import numpy as np
    a = np.asarray(_pipe()(prompt="x", width=256, height=256, num_inference_steps=2,
                           generator=_gen(5)).images[0])
    b = np.asarray(_pipe()(prompt="x", width=256, height=256, num_inference_steps=2,
                           generator=_gen(5)).images[0])
    assert (a == b).all()


def test_guidance_schedule_shape():
    s = cz_ideogram.guidance_schedule(48)
    assert s == [7.0] * 45 + [3.0] * 3                # the official one, exactly
    s = cz_ideogram.guidance_schedule(24, 5.0)
    assert len(s) == 24 and s[0] == 5.0 and s[-1] == pytest.approx(5.0 * 3 / 7)
    assert cz_ideogram.guidance_schedule(1) == [7.0]
    assert cz_ideogram.guidance_schedule(10, 0) [0] == 7.0   # <= 0 -> the official peak


def test_any_step_count_reaches_the_pipeline():
    # the raw pipeline refuses a schedule whose length differs from the step count
    out = _pipe()(prompt="x", width=256, height=256, num_inference_steps=5,
                  generator=_gen(), guidance_scale=4.0)
    assert out.images[0].size == (256, 256)


def test_fit_size():
    assert cz_ideogram.fit_size(1000, 1000) == (992, 992)
    assert cz_ideogram.fit_size(4096, 4096) == (2048, 2048)
    assert cz_ideogram.fit_size(100, 100) == (256, 256)
    w, h = cz_ideogram.fit_size(2048, 256)
    assert w / h <= 6.0 + 1e-9


def test_interrupt_returns_an_image():
    def cb(p, i, t, kw):
        if i == 0:
            p._interrupt = True
        return kw
    out = _pipe()(prompt="x", width=256, height=256, num_inference_steps=4,
                  generator=_gen(), callback_on_step_end=cb)
    assert out.images[0].size == (256, 256)
    out = _pipe()(prompt="x", width=256, height=256, num_inference_steps=2, generator=_gen())
    assert out.images[0].size == (256, 256)          # the next call starts clean


@pytest.mark.parametrize("kw", [{"strength": 0.4}, {"mask_image": Image.new("L", (8, 8))},
                                {"image": Image.new("RGB", (8, 8))}])
def test_lot3_and_lot4_paths_are_refused_by_name(kw):
    with pytest.raises(NotImplementedError) as e:
        _pipe()(prompt="x", width=256, height=256, num_inference_steps=1, **kw)
    assert "lot" in str(e.value)


def test_negative_prompt_is_dropped_not_fatal():
    out = _pipe()(prompt="x", negative_prompt="blurry", width=256, height=256,
                  num_inference_steps=1, generator=_gen())
    assert out.images[0].size == (256, 256)


def test_no_encode_prompt_exposed():
    # cz_pipeline's embedding cache must find nothing to call (see the module docstring)
    assert not hasattr(_pipe(), "encode_prompt")


def test_family_detection():
    full = ["llm_cond_proj.weight", "layers.0.attention.to_q.weight", "embed_image_indicator.weight"]
    assert cz_ideogram.family_from_keys(full) == "ideogram"
    flux3 = ["early_stream_modulations.image.lin.weight"]
    assert cz_ideogram.family_from_keys(flux3) == "flux3"
    flux2 = ["double_blocks.0.img_attn.qkv.weight"]
    assert cz_ideogram.family_from_keys(flux2) == "flux2"
    # a LoRA: names shared with Z-Image, the width decides
    lora_ideo = {"transformer.layers.0.attention.to_q.lora_A.weight": {"shape": [16, 4608]}}
    lora_zimg = {"transformer.layers.0.attention.to_q.lora_A.weight": {"shape": [16, 3840]}}
    assert cz_ideogram.family_from_header(lora_ideo) == "ideogram"
    assert cz_ideogram.family_from_header(lora_zimg) is None


def test_repo_detection():
    assert cz_ideogram.is_engine_repo(cz_ideogram.DEFAULT_REPO)
    assert cz_ideogram.is_engine_repo("someone/ideogram-4-mirror")
    assert not cz_ideogram.is_engine_repo("black-forest-labs/flux-3-image-klein")


def test_single_file_transformer_is_refused_by_name():
    with pytest.raises(NotImplementedError):
        cz_ideogram.load_transformer("repo", torch.bfloat16, single_file="x.safetensors")


def test_live_preview_uses_the_ideogram_projection():
    """The app's real step-end callback, on the tiny pipe: a frame comes out and it went
    through the (p, p, c) projection derived from this pipe's VAE, not the FLUX.2 one."""
    import cz_pipeline as P
    P._IDEOGRAM_PROJ_CACHE.clear()
    was = P.LIVE_PREVIEW_ENABLED
    P.LIVE_PREVIEW_ENABLED = True
    P.preview_begin()
    try:
        kw = P._step_end_kwargs(3, 256, 256)
        _pipe()(prompt="GRAND OPENING", width=256, height=256, num_inference_steps=3,
                generator=_gen(), **kw)
        snap = P.preview_snapshot()
        assert snap["img"] is not None and snap["step"] == 3, snap
        vae = _pipe()._pipe.vae
        assert id(vae) in P._IDEOGRAM_PROJ_CACHE
    finally:
        P.preview_end()
        P.LIVE_PREVIEW_ENABLED = was


def test_fast_attention_gives_the_same_image():
    """No text padding + no all-True mask = the same image as the raw call (2048 text
    slots, a mask on every attention): padding sits in its own segment, text positions
    start at 0, image positions are fixed. Only the attention kernel changes."""
    import numpy as np
    was = cz_ideogram.FAST_ATTENTION
    seen = []
    pipe = _pipe()
    real = pipe._pipe.__class__.__call__

    def spy(self, *a, **kw):
        seen.append(kw.get("max_sequence_length"))
        return real(self, *a, **kw)
    try:
        pipe._pipe.__class__.__call__ = spy
        cz_ideogram.set_fast_attention(False)
        slow = pipe(prompt='a poster that says "GRAND OPENING"', width=256, height=256,
                    num_inference_steps=3, generator=_gen(7), output_type="np").images
        cz_ideogram.set_fast_attention(True)
        fast = pipe(prompt='a poster that says "GRAND OPENING"', width=256, height=256,
                    num_inference_steps=3, generator=_gen(7), output_type="np").images
    finally:
        pipe._pipe.__class__.__call__ = real
        cz_ideogram.set_fast_attention(was)
    n = cz_ideogram.prompt_token_count(pipe._pipe, 'a poster that says "GRAND OPENING"')
    assert seen == [None, n] and 0 < n < 100, (seen, n)
    procs = {type(m.processor).__name__ for m in pipe._pipe.transformer.modules()
             if hasattr(m, "processor") and m.processor is not None}
    assert procs == {"_AllTrueMaskSkip"}, procs
    diff = float(np.abs(np.asarray(fast) - np.asarray(slow)).max())
    print(f"fast vs raw: max pixel diff {diff:.2e}")
    assert diff < 2e-2, diff


def test_mask_skip_keeps_a_real_mask():
    """A mask that is not all True (a batch of prompts of different lengths) is passed
    through untouched."""
    cls = cz_ideogram._mask_skip_processor_class()
    got = []
    p = cls()
    import diffusers.models.transformers.transformer_ideogram4 as T
    orig = T.Ideogram4AttnProcessor.__call__
    try:
        T.Ideogram4AttnProcessor.__call__ = lambda self, a, h, m, r: got.append(m) or h
        full = torch.ones(1, 1, 4, 4, dtype=torch.bool)
        part = full.clone()
        part[..., 0] = False
        p(None, torch.zeros(1), full, None)
        p(None, torch.zeros(1), part, None)
    finally:
        T.Ideogram4AttnProcessor.__call__ = orig
    assert got[0] is None and got[1] is part


def test_model_offload_moves_only_the_text_encoder():
    """'model' offload on Ideogram = the encoder alone leaves the card: in before the
    call, back at the first transformer, never per step; no accelerate hook anywhere
    (diffusers' chain moved the conditional transformer back and forth every step, 2.8
    s/step against 1.7 on the 5090, and its encoder hook never fired, so the encoder ran
    on the CPU). Execution device = CPU here: what is checked is WHEN the encoder moves,
    that the transformers never do, and that the image does not change."""
    import numpy as np
    from ideogram_tiny import tiny_pipe
    pipe = tiny_pipe()
    ref = pipe(prompt="GRAND OPENING", width=256, height=256, num_inference_steps=3,
               generator=_gen(3), output_type="np").images
    raw = pipe._pipe
    pipe.enable_model_cpu_offload(device="cpu")
    assert pipe._all_hooks and not getattr(raw, "_all_hooks", None)
    for name in ("text_encoder", "transformer", "unconditional_transformer", "vae"):
        assert not hasattr(getattr(raw, name), "_hf_hook"), name

    moves = []
    te_to = raw.text_encoder.to
    tr_to = raw.transformer.to
    raw.text_encoder.to = lambda *a, **k: moves.append(("te", str(a[0]))) or te_to(*a, **k)
    raw.transformer.to = lambda *a, **k: moves.append(("tr", str(a[0]))) or tr_to(*a, **k)
    try:
        out = pipe(prompt="GRAND OPENING", width=256, height=256, num_inference_steps=3,
                   generator=_gen(3), output_type="np").images
    finally:
        del raw.text_encoder.to, raw.transformer.to
    assert moves == [("te", "cpu"), ("te", "cpu")], moves   # in (to dev=cpu), back once
    assert not pipe._te_on_gpu
    assert float(np.abs(np.asarray(out) - np.asarray(ref)).max()) < 1e-5

    # diffusers' own model offload comes back on the raw pipe (a LoRA load): replaced.
    raw.enable_model_cpu_offload(device="cpu")
    assert raw._all_hooks
    pipe(prompt="x", width=256, height=256, num_inference_steps=2, generator=_gen(0))
    assert not getattr(raw, "_all_hooks", None)
    assert not hasattr(raw.transformer, "_hf_hook")
    assert pipe._offload_device is not None

    # release_vram(offload=True) path and switching back to 'none'.
    pipe.maybe_free_model_hooks()
    pipe.remove_all_hooks()
    assert not pipe._all_hooks and pipe._offload_device is None and not pipe._tr_pre


def test_sparse_text_norm_is_exact_and_follows_the_flag():
    """llm_cond_norm on the non-zero rows only == the stock RMSNorm on every row (a zero
    row normalizes to zero), in bf16 like on the card, and the stock path when the flag
    is off."""
    from diffusers.models.normalization import RMSNorm
    g = torch.Generator().manual_seed(0)
    norm = RMSNorm(512, eps=1e-6, elementwise_affine=True).to(torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(512, generator=g))
    x = torch.zeros(2, 40, 512, dtype=torch.bfloat16)
    x[:, 3:9] = torch.randn(2, 6, 512, generator=g).to(torch.bfloat16)   # the text rows
    ref = norm(x)
    calls = []
    real = norm.forward
    norm.forward = lambda t: calls.append(tuple(t.shape)) or real(t)
    holder = type("P", (), {})()
    holder.transformer = type("T", (), {})()
    holder.transformer.llm_cond_norm = norm
    was = cz_ideogram.FAST_ATTENTION
    try:
        assert cz_ideogram.install_sparse_text_norm(holder) == 1
        assert cz_ideogram.install_sparse_text_norm(holder) == 0          # idempotent
        cz_ideogram.set_fast_attention(True)
        got = norm(x)
        assert calls[-1] == (12, 512), calls        # 2 x 6 text rows, not 2 x 40
        assert got.dtype == ref.dtype and torch.equal(got, ref)
        cz_ideogram.set_fast_attention(False)
        norm(x)
        assert calls[-1] == (2, 40, 512), calls     # stock path
    finally:
        cz_ideogram.set_fast_attention(was)
