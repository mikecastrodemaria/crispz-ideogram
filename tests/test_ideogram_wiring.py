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


def test_protocol_caps_for_lot1():
    caps = cz_protocol.caps_dict()["supports"]
    assert caps["negative"] is False
    assert caps["img2img"] is False and caps["inpaint"] is False
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
