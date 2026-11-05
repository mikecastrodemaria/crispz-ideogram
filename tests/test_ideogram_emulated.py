"""img2img and inpaint EMULATED on Ideogram 4 (tiny random pipeline, CPU): the tail of
Ideogram's own schedule, the guidance rotated, the area outside the mask kept exactly,
a Stop that still returns the partial image, edit still refused."""
# scope: ideogram
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import cz_ideogram  # noqa: E402
from ideogram_tiny import tiny_pipe  # noqa: E402

_PIPE = None


def _pipe():
    global _PIPE
    if _PIPE is None:
        _PIPE = tiny_pipe()
    return _PIPE


def _gen(s=0):
    return torch.Generator().manual_seed(s)


def _init(w=256, h=256):
    img = Image.new("RGB", (w, h), (30, 60, 200))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w // 2, h], fill=(230, 40, 40))
    d.ellipse([w // 4, h // 4, 3 * w // 4, 3 * h // 4], fill=(250, 220, 30))
    return img


def _count_calls(pipe):
    calls = []
    tr = pipe._pipe.transformer
    h = tr.register_forward_pre_hook(lambda m, a: calls.append(1))
    return calls, h


def test_img2img_runs_the_tail_of_the_schedule():
    from diffusers.pipelines.ideogram4.pipeline_ideogram4 import (_logit_normal_sigmas,
                                                                  _resolution_aware_mu)
    pipe = _pipe()
    seen = {}
    real_call = pipe._pipe.__class__.__call__

    def spy(self, *a, **kw):
        seen.update(kw)
        out = real_call(self, *a, **kw)
        seen["timesteps"] = self.scheduler.timesteps.clone()
        return out
    calls, h = _count_calls(pipe)
    pipe._pipe.__class__.__call__ = spy
    try:
        out = pipe(prompt="a poster", image=_init(320, 256), strength=0.5,
                   num_inference_steps=8, generator=_gen(1)).images[0]
    finally:
        pipe._pipe.__class__.__call__ = real_call
        h.remove()
    assert out.size == (320, 256)                      # the image's own size
    assert len(calls) == 4                             # 0.5 x 8 steps
    full = _logit_normal_sigmas(8, _resolution_aware_mu(256, 320, 0.0), std=1.5)
    ts = seen["timesteps"].float() / 1000.0
    assert torch.allclose(ts, full[4:].float(), atol=1e-4), (ts, full[4:])
    sched = cz_ideogram.guidance_schedule(8, None)
    assert seen["guidance_schedule"] == list(sched[4:]) + list(sched[:4])
    assert seen["latents"].shape == (1, (256 // 16) * (320 // 16), 128)
    assert "set_timesteps" not in vars(pipe._pipe.scheduler)    # wrap removed


def test_strength_extremes():
    pipe = _pipe()
    calls, h = _count_calls(pipe)
    try:
        pipe(prompt="x", image=_init(), strength=0.0, num_inference_steps=8, generator=_gen())
        assert len(calls) == 1                         # a 0 denoise still touches it
        calls.clear()
        pipe(prompt="x", image=_init(), strength=1.0, num_inference_steps=8, generator=_gen())
        assert len(calls) == 8
    finally:
        h.remove()


def test_low_strength_stays_closer_to_the_init():
    pipe = _pipe()
    init = _init()
    a = np.asarray(init, dtype=np.float32)
    low = pipe(prompt="x", image=init, strength=0.2, num_inference_steps=10,
               generator=_gen(3)).images[0]
    high = pipe(prompt="x", image=init, strength=1.0, num_inference_steps=10,
                generator=_gen(3)).images[0]
    d_low = np.abs(np.asarray(low, dtype=np.float32) - a).mean()
    d_high = np.abs(np.asarray(high, dtype=np.float32) - a).mean()
    assert d_low < d_high, (d_low, d_high)


def test_inpaint_keeps_the_area_outside_the_mask():
    """Checked on the latent TOKENS the decoder receives: outside the mask, after the last
    step (sigma 0), they are the init image's own tokens, exactly; inside, they moved.
    (Pixels cannot be compared: the FLUX.2 VAE decoder has a mid-block self-attention,
    every decoded pixel depends on the whole latent.)"""
    pipe = _pipe()
    init = _init(256, 256)
    mask = Image.new("L", (256, 256), 0)
    ImageDraw.Draw(mask).rectangle([160, 0, 255, 255], fill=255)        # right side
    last = {}

    def grab(p, i, t, kw):
        last["lat"] = kw["latents"].clone()
        return kw
    pipe(prompt="x", image=init, mask_image=mask, strength=1.0, num_inference_steps=6,
         generator=_gen(5), callback_on_step_end=grab)
    x0 = pipe._encode_tokens(init, 256, 256, "cpu")
    m = cz_ideogram.IdeogramPipe._mask_tokens(mask, 256, 256, "cpu")[0, :, 0] > 0
    lat = last["lat"][0]
    assert m.sum() > 0 and (~m).sum() > 0
    assert torch.allclose(lat[~m], x0[0][~m], atol=1e-5)
    assert (lat[m] - x0[0][m]).abs().mean() > 1e-2


def test_full_white_mask_is_plain_img2img():
    pipe = _pipe()
    full = Image.new("L", (256, 256), 255)
    a = pipe(prompt="x", image=_init(), mask_image=full, strength=0.5, num_inference_steps=4,
             generator=_gen(7), output_type="np").images
    b = pipe(prompt="x", image=_init(), strength=0.5, num_inference_steps=4,
             generator=_gen(7), output_type="np").images
    assert np.abs(np.asarray(a) - np.asarray(b)).max() < 1e-5


def test_stop_returns_the_partial_image():
    pipe = _pipe()
    calls, h = _count_calls(pipe)

    def stop(p, i, t, kw):
        if i == 1:
            p._interrupt = True
        return kw
    mask = Image.new("L", (256, 256), 0)
    ImageDraw.Draw(mask).rectangle([0, 0, 128, 128], fill=255)
    try:
        out = pipe(prompt="x", image=_init(), mask_image=mask, strength=1.0,
                   num_inference_steps=8, generator=_gen(), callback_on_step_end=stop)
    finally:
        h.remove()
    assert len(calls) == 2 and out.images[0].size == (256, 256)


def test_mask_tokens_max_pool():
    m = Image.new("L", (64, 32), 0)
    m.putpixel((20, 5), 255)                         # one pixel in token (row 0, col 1)
    t = cz_ideogram.IdeogramPipe._mask_tokens(m, 64, 32, "cpu")
    assert t.shape == (1, 8, 1) and t[0, 1, 0] == 1.0 and t.sum() == 1.0


def test_an_image_without_strength_is_still_refused_by_name():
    with pytest.raises(NotImplementedError, match="crispz-flux3"):
        _pipe()(prompt="x", image=_init(), num_inference_steps=2)


def test_progress_and_preview_count_the_steps_that_run():
    """Lot 7: diffusers sized its bar on the full schedule (14/48, then stopped), and the
    live preview said "step 5 of 48". Both now count the steps the emulated img2img
    really runs; everything is put back after the call."""
    pipe = _pipe()
    inner = pipe._pipe
    totals, previews = [], []
    real_bar = type(inner).progress_bar

    def spy_bar(self, iterable=None, total=None):
        totals.append(total)
        return real_bar(self, iterable=iterable, total=total)

    def cb(p, i, t, kw):
        previews.append(getattr(p, "_cz_run_steps", None))
        return kw
    type(inner).progress_bar = spy_bar
    try:
        pipe(prompt="x", image=_init(), strength=0.5, width=256, height=256,
             num_inference_steps=8, generator=_gen(), callback_on_step_end=cb)
    finally:
        type(inner).progress_bar = real_bar
    assert totals == [4], totals
    assert previews and set(previews) == {4} and len(previews) == 4
    assert "progress_bar" not in vars(inner) and inner._cz_run_steps is None
    assert "set_timesteps" not in vars(inner.scheduler)
