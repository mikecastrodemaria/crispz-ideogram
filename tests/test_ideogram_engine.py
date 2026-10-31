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
