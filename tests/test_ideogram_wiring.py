"""cz_pipeline / cz_protocol / cz_core on the Ideogram engine (lot 1, CPU).

A tiny IdeogramPipe is put in place of the loaded model and the real entry point runs:
generate(). Also: get_pipe hands the one object back, the guidance slider reaches the
adapter as the schedule's peak, no negative prompt, the sampler is left alone, the
protocol announces what lot 1 can and cannot do, the profile gives 48 steps / 7.0, and
a FLUX.3 LoRA is refused with the app it belongs to.
Run: python -m pytest tests/test_ideogram_wiring.py -q
"""
import json
import os
import struct
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cz_core  # noqa: E402
import cz_ideogram  # noqa: E402
import cz_pipeline as P  # noqa: E402
import cz_protocol  # noqa: E402
from ideogram_tiny import tiny_pipe  # noqa: E402

CALLS = []
_TINY = None


def _install():
    global _TINY
    if _TINY is None:
        _TINY = tiny_pipe()
    pipe = _TINY
    real = cz_ideogram.IdeogramPipe.__call__

    def spy(self, *a, **kw):
        CALLS.append(dict(kw))
        return real(self, *a, **kw)
    cz_ideogram.IdeogramPipe.__call__ = spy
    P._BASE_PIPE = pipe
    P._DERIVED = {"txt2img": pipe}
    P._LOADED_KEY = (P.BASE_REPO, P.ZIMAGE_TRANSFORMER, P.OFFLOAD_MODE, P.PRECISION)
    P._BASE_SCHED_CONFIG = dict(pipe.scheduler.config)
    P.LORAS = []
    P.EDIT_LORAS = []
    return real


def _run(fn):
    CALLS.clear()
    real = _install()
    try:
        return fn()
    finally:
        cz_ideogram.IdeogramPipe.__call__ = real
        P._BASE_PIPE, P._DERIVED, P._LOADED_KEY = None, {}, None


def test_defaults_point_at_ideogram():
    assert P.BASE_REPO == cz_ideogram.DEFAULT_REPO
    assert P.PRECISION == "nf4"
    assert P.CFG_MODEL_KEY == "ideogram_model"


def test_generate_reaches_the_adapter_with_the_guidance_peak():
    old = P.GUIDANCE
    P.GUIDANCE = 6.0
    try:
        img = _run(lambda: P.generate("a sign that says OPEN", 256, 512, 3, 1))
    finally:
        P.GUIDANCE = old
    assert img.size == (256, 512)
    assert CALLS[-1]["guidance_scale"] == 6.0
    assert "negative_prompt" not in CALLS[-1] or not CALLS[-1]["negative_prompt"]


def test_get_pipe_is_one_object():
    def go():
        base = P.get_pipe("txt2img")
        assert P.get_pipe("img2img") is base and P.get_pipe("inpaint") is base
    _run(go)


def test_no_negative_prompt():
    assert P._cfg("blurry") == {}


def test_sampler_left_alone():
    def go():
        pipe = P.get_pipe("txt2img")
        sched = pipe.scheduler
        old = (P.SAMPLER, P.SCHEDULE)
        P.SAMPLER, P.SCHEDULE = "unipc", "karras"
        try:
            P._apply_sampler(pipe)
        finally:
            P.SAMPLER, P.SCHEDULE = old
        assert pipe.scheduler is sched
    _run(go)


def test_protocol_caps():
    """Lot 3: img2img and inpaint are on, and announced as EMULATED."""
    caps = cz_protocol.caps_dict()["supports"]
    assert caps["negative"] is False
    assert caps["img2img"] is True and caps["inpaint"] is True
    assert sorted(caps["emulated"]) == ["img2img", "inpaint"]
    assert caps["edit"] is False and caps["refs"] is False
    assert cz_protocol.TOOL == "crispz-ideogram"


def test_profile():
    assert cz_core.profile_for_model("ideogram-ai/ideogram-4-nf4-diffusers") == (48, 7.0)


def _fake_safetensors(names_shapes):
    hdr, off = {}, 0
    for n, shape in names_shapes:
        size = 2
        for d in shape:
            size *= d
        hdr[n] = {"dtype": "BF16", "shape": shape, "data_offsets": [off, off + size]}
        off += size
    raw = json.dumps(hdr).encode()
    f = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False)
    f.write(struct.pack("<Q", len(raw)) + raw + b"\0" * off)
    f.close()
    return f.name


def test_a_flux3_lora_is_refused_and_named():
    path = _fake_safetensors([
        ("transformer.mode_blocks.image.0.attn.to_q.lora_A.weight", [4, 8]),
        ("transformer.mode_blocks.image.0.attn.to_q.lora_B.weight", [8, 4])])
    try:
        why = P._lora_unsupported(path)
    finally:
        os.unlink(path)
    assert why and "crispz-flux3" in why


def test_an_ideogram_lora_passes_the_family_guard():
    path = _fake_safetensors([
        ("transformer.layers.0.attention.to_q.lora_A.weight", [4, 4608]),
        ("transformer.layers.0.attention.to_q.lora_B.weight", [4608, 4])])
    try:
        why = P._lora_unsupported(path)
    finally:
        os.unlink(path)
    assert why is None


def test_a_flux3_checkpoint_is_refused_and_named():
    path = _fake_safetensors([("early_stream_modulations.image.lin.weight", [4, 4])])
    try:
        why = P._flux3_file_refusal(path)
    finally:
        os.unlink(path)
    assert why and "crispz-flux3" in why


def test_shipped_defaults_are_ideogram_not_flux3():
    """The app must START on the official recipe: config-sample.txt kept FLUX.3's
    16 steps / guidance 1.0 / "Distilled" preset, i.e. a first render at peak 1.0
    (almost no CFG) until a preset was clicked. The UI ranges must reach 48 steps and
    stop at Ideogram's 2048 px."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "config-sample.txt"), encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["default_performance"] in cfg["performance_presets"]
    assert cfg["default_gen_steps"] == 48
    assert cfg["default_guidance"] == cz_ideogram.SCHEDULE_PEAK
    import cz_ui
    assert all(max(v) <= cz_ideogram.MAX_SIDE for v in cz_ui.ASPECT_RATIOS.values())
    assert "2K  2048 x 2048  (1:1)" in cz_ui.ASPECT_RATIOS
    src = open(os.path.join(here, "cz_ui.py"), encoding="utf-8").read()
    assert "gen_steps = gr.Slider(2, 80," in src


def test_vram_downgrade_frees_the_old_pipe_before_reloading():
    """The guard fires at step 1 -> generate() reloads in 'model'. The old pipe must be
    unreachable BEFORE the reload: generate()'s own local kept it alive and the reload
    stacked a second 15 GB copy (5090, 2026-10-31: 29.9 GB, 90 s/step)."""
    import gc
    import types
    import weakref
    from PIL import Image

    class FakePipe:
        calls = 0

        def __call__(self, **kw):
            FakePipe.calls += 1
            if FakePipe.calls == 1:
                P._VRAM_DOWNGRADE = True          # what the step-1 guard does
            return types.SimpleNamespace(images=[Image.new("RGB", (64, 64))])

    first = FakePipe()
    ref = weakref.ref(first)
    alive_at_reload = []
    pipes = [first]
    del first

    def fake_get_pipe(kind="txt2img"):
        if not pipes:                             # the reload
            gc.collect()
            alive_at_reload.append(ref() is not None)
            return FakePipe()
        return pipes.pop()

    saved = (P.get_pipe, P.cz_hw.record_downgrade, P.OFFLOAD_MODE, P._AUTO_OFFLOAD,
             P._set_slicing)
    try:
        P.get_pipe = fake_get_pipe
        P.cz_hw.record_downgrade = lambda *a, **k: None
        P.OFFLOAD_MODE = "auto"
        P._set_slicing = lambda *a, **k: None
        img = P.generate.__wrapped__("x", 64, 64, 2, 1) if hasattr(P.generate, "__wrapped__") \
            else P.generate("x", 64, 64, 2, 1)
    finally:
        (P.get_pipe, P.cz_hw.record_downgrade, P.OFFLOAD_MODE, P._AUTO_OFFLOAD,
         P._set_slicing) = saved
        P._VRAM_DOWNGRADE = False
    assert img is not None and FakePipe.calls == 2
    assert alive_at_reload == [False], "the old pipe was still referenced at reload"


def test_auto_offload_prefers_encoder_only_under_40_gb():
    """'auto' -> 'model' (the encoder alone off the card) when the VRAM test says 'none'
    on a card under 40 GB: 2048 px peaks at 23.4 GB that way on the 5090, ~29 GB with the
    encoder resident. Bigger cards, and every other verdict, are left as they are."""
    f = P._ideogram_auto
    assert f("none", "fits", 31.8)[0] == "model"
    assert "explicit 'none'" in f("none", "fits", 31.8)[1]
    assert f("none", "fits", 48.0) == ("none", "fits")
    assert f("model", "tight", 24.0) == ("model", "tight")
    assert f("sequential", "tiny", 8.0) == ("sequential", "tiny")
    assert f("none", "fits", None) == ("none", "fits")


# --- Lot 3: the app's img2img / inpaint paths reach the emulated adapter --------------
def _img(w=256, h=256):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (w, h), (40, 90, 200))
    ImageDraw.Draw(im).ellipse([w // 4, h // 4, 3 * w // 4, 3 * h // 4], fill=(240, 200, 20))
    return im


def test_inpaint_run_goes_through_the_emulated_inpaint():
    from PIL import Image, ImageDraw
    mask = Image.new("L", (256, 256), 0)
    ImageDraw.Draw(mask).rectangle([128, 0, 255, 255], fill=255)
    out = _run(lambda: P.inpaint_run(_img(), mask, "a red door", 4, 0.8, 3))
    assert out.size == (256, 256)
    kw = CALLS[-1]
    assert kw["mask_image"] is not None and kw["strength"] == 0.8


def test_refine_whole_is_an_img2img():
    def go():
        return P._refine_whole(P.get_pipe("img2img"), _img(320, 256), 0.35, 8, "x", 1)
    out = _run(go)
    assert out.size == (320, 256)
    kw = CALLS[-1]
    assert kw["strength"] == 0.35 and kw.get("mask_image") is None


def test_reframe_and_directional_outpaint_run():
    out = _run(lambda: P.reframe(_img(256, 256), 16, 9, "contain", "x", 3, 1))
    assert out.width > out.height
    assert CALLS[-1]["mask_image"] is not None
    out2 = _run(lambda: P.outpaint_directions(_img(256, 256), None, ["right"], "x", 3, 1))
    assert out2.width > 256


def test_extra_checkpoint_folder_and_subfolders_are_scanned(tmp_path, monkeypatch):
    """Mike's setup (2026-11-01): single files under the EXTRA folder, in subfolders. They
    must be found (recursive walk of both folders), and an Ideogram single file must be
    skipped WITH its reason in the log, never silently."""
    import torch
    from safetensors.torch import save_file
    main, extra = tmp_path / "main", tmp_path / "extra"
    (extra / "ideogram" / "comfy").mkdir(parents=True)
    main.mkdir()
    t = {"llm_cond_proj.weight": torch.zeros(2, 2), "embed_image_indicator.weight": torch.zeros(2, 2)}
    save_file(t, str(extra / "ideogram" / "comfy" / "ideogram4_fp8Scaled.safetensors"))
    save_file({"x": torch.zeros(1)}, str(main / "not_a_model.safetensors"))
    logs = []
    monkeypatch.setattr(P, "CHECKPOINTS_DIR", str(main))
    monkeypatch.setattr(P, "CHECKPOINTS_EXTRA_DIR", str(extra))
    monkeypatch.setattr(P, "_log", lambda m, *a, **k: logs.append(m))
    P._FAMILY_CACHE.clear() if hasattr(P, "_FAMILY_CACHE") else None
    assert P._checkpoint_dirs() == [str(main), str(extra)]
    assert P._walk_checkpoints(str(extra)) == ["ideogram/comfy/ideogram4_fp8Scaled.safetensors"]
    listed = P.list_checkpoints()
    assert listed == []
    said = [m for m in logs if "ideogram4_fp8Scaled.safetensors" in m]
    # found in the sub-folder, read, and skipped for ITS reason (lot 8b: a 2-tensor stub
    # is not the published transformer), never silently
    assert said and "not an Ideogram 4 transformer of the published size" in said[0], logs
    assert P.resolve_checkpoint("ideogram4_fp8Scaled.safetensors").endswith(
        os.path.join("ideogram", "comfy", "ideogram4_fp8Scaled.safetensors"))


def test_refine_whole_downgrade_frees_the_weights_the_caller_still_holds(monkeypatch):
    """Lot 7 (lot 3's open point): _refine_whole gets its pipe from the CALLER, which keeps
    it during the VRAM fallback. free_vram now releases the IdeogramPipe wrapper, so the
    real pipeline is unreachable at reload time although the caller still holds the
    wrapper; the stale wrapper is then swapped for the fresh pipe."""
    import gc
    import types
    import weakref
    import cz_ideogram
    from PIL import Image

    class Inner:
        pass
    inner = Inner()
    ref = weakref.ref(inner)
    held = cz_ideogram.IdeogramPipe(inner)          # what the caller keeps
    del inner
    calls, alive_at_reload, fresh = [], [], []

    def fake_call(pipe, **kw):
        calls.append(pipe)
        if len(calls) == 1:
            P._VRAM_DOWNGRADE = True
        return types.SimpleNamespace(images=[Image.new("RGB", (64, 64))])

    def fake_get_pipe(kind="txt2img"):
        if P._BASE_PIPE is not None:
            return P._BASE_PIPE
        gc.collect()
        alive_at_reload.append(ref() is not None)
        new = types.SimpleNamespace(released=False)
        fresh.append(new)
        P._BASE_PIPE = new
        return new

    monkeypatch.setattr(P, "_BASE_PIPE", held)
    monkeypatch.setattr(P, "_DERIVED", {"txt2img": held})
    monkeypatch.setattr(P, "_qwen_call", fake_call)
    monkeypatch.setattr(P, "get_pipe", fake_get_pipe)
    monkeypatch.setattr(P, "_set_slicing", lambda *a, **k: None)
    monkeypatch.setattr(P.cz_hw, "record_downgrade", lambda *a, **k: None)
    monkeypatch.setattr(P, "OFFLOAD_MODE", "auto")
    monkeypatch.setattr(P, "_AUTO_OFFLOAD", "none")
    try:
        out = P._refine_whole(held, Image.new("RGB", (64, 64)), 0.3, 4, "x", 1)
    finally:
        P._VRAM_DOWNGRADE = False
    assert out is not None and len(calls) == 2
    assert alive_at_reload == [False], "the caller's reference pinned the old weights"
    assert held.released and calls[1] is fresh[0]
    import pytest
    with pytest.raises(RuntimeError, match="released"):
        held.transformer
    assert P._fresh(held) is fresh[0], "a stale wrapper is swapped for the current pipe"
