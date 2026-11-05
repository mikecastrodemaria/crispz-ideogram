"""Lot 8a: Ideogram 4 Instant (fal, distilled, one conditional transformer) on a tiny
random Ideogram 4 (CPU, no gated download).

What the first validation asks:
  - numerical equivalence at guidance 1: the full two-branch call and the call with the
    unconditional computation removed give the SAME latents;
  - the profile, not a file name or a precision, decides the recipe (steps, guidance,
    mu / std, the unconditional branch), and a guidance other than the recipe's is
    refused, never applied;
  - a transformer source is checked strictly before loading (architecture, tensor
    names, shapes, duplicates, quantized dtypes);
  - the emulated img2img / inpaint follow the profile's schedule.
Run: python -m pytest tests/test_ideogram_instant.py -q
"""
import json
import os
import sys

import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cz_ideogram as E  # noqa: E402
from ideogram_tiny import tiny_components  # noqa: E402

INSTANT = E.PROFILES["instant"]
TINY_ARCH = None


def _gen(s=0):
    return torch.Generator().manual_seed(s)


def _raw(comps):
    from diffusers import Ideogram4Pipeline
    return Ideogram4Pipeline(**comps)


def _distilled_pipe(seed=0):
    """The tiny pipeline with a zero unconditional branch, under the Instant profile."""
    comps = tiny_components(seed)
    comps["unconditional_transformer"] = E.ZeroUnconditionalTransformer(torch.float32)
    return E.IdeogramPipe(_raw(comps), repo="tiny", profile=INSTANT, source="tiny")


def _latents(pipe, **kw):
    args = dict(prompt="a poster that says HELLO", width=256, height=256,
                num_inference_steps=4, guidance_scale=1.0, guidance_schedule=None,
                mu=0.0, std=1.75, generator=_gen(3), output_type="latent")
    args.update(kw)
    return pipe(**args).images


# --- 1. Equivalence ------------------------------------------------------------------
def test_dropping_the_unconditional_branch_is_exact_at_guidance_one():
    comps = tiny_components(0)
    full = _latents(_raw(dict(comps)))              # real second transformer, weight 0
    comps["unconditional_transformer"] = E.ZeroUnconditionalTransformer(torch.float32)
    zero = _latents(_raw(comps))
    assert torch.equal(full, zero), (full - zero).abs().max()


def test_the_adapter_runs_the_published_recipe_and_one_transformer_call_per_step():
    pipe = _distilled_pipe()
    seen, calls = {}, []
    real = pipe._pipe.__class__.__call__

    def spy(self, *a, **kw):
        seen.update(kw)
        return real(self, *a, **kw)
    h = pipe._pipe.transformer.register_forward_pre_hook(lambda m, a: calls.append(1))
    pipe._pipe.__class__.__call__ = spy
    try:
        out = pipe(prompt="x", width=256, height=256, generator=_gen(1)).images[0]
    finally:
        pipe._pipe.__class__.__call__ = real
        h.remove()
    assert out.size == (256, 256)
    assert seen["num_inference_steps"] == 8 and len(calls) == 8
    assert seen["guidance_scale"] == 1.0 and seen["guidance_schedule"] is None
    assert (seen["mu"], seen["std"]) == (0.0, 1.75)


def test_adapter_output_equals_the_raw_pipeline_at_the_recipe():
    comps = tiny_components(0)
    raw = _latents(_raw(dict(comps)), num_inference_steps=8)
    comps["unconditional_transformer"] = E.ZeroUnconditionalTransformer(torch.float32)
    pipe = E.IdeogramPipe(_raw(comps), repo="tiny", profile=INSTANT)
    E.set_fast_attention(False)        # the raw call above has no fast path either
    try:
        ours = pipe(prompt="a poster that says HELLO", width=256, height=256,
                    generator=_gen(3), output_type="latent").images
    finally:
        E.set_fast_attention(True)
    assert torch.equal(raw, ours)


def test_another_guidance_is_refused_not_applied():
    pipe = _distilled_pipe()
    with pytest.raises(ValueError, match="guidance 1"):
        pipe(prompt="x", width=256, height=256, num_inference_steps=2,
             guidance_scale=7.0, generator=_gen())
    # 1.0 or nothing is fine
    pipe(prompt="x", width=256, height=256, num_inference_steps=1, guidance_scale=1.0,
         generator=_gen())


def test_steps_outside_the_recipe_are_said(monkeypatch):
    lines = []
    monkeypatch.setattr(E, "_log", lambda m: lines.append(str(m)))
    _distilled_pipe()(prompt="x", width=256, height=256, num_inference_steps=3,
                      generator=_gen())
    assert any("published recipe is 8" in m for m in lines)


def test_the_base_profile_is_unchanged():
    comps = tiny_components(0)
    pipe = E.IdeogramPipe(_raw(comps), repo="tiny")
    assert pipe.profile.key == "base"
    seen = {}
    real = pipe._pipe.__class__.__call__

    def spy(self, *a, **kw):
        seen.update(kw)
        return real(self, *a, **kw)
    pipe._pipe.__class__.__call__ = spy
    try:
        pipe(prompt="x", width=256, height=256, num_inference_steps=4, generator=_gen())
    finally:
        pipe._pipe.__class__.__call__ = real
    assert seen["guidance_schedule"] == E.guidance_schedule(4, None)
    assert (seen["mu"], seen["std"]) == (0.0, 1.5)


# --- 2. Profiles ---------------------------------------------------------------------
def test_profiles_come_from_documented_sources_never_from_a_name(tmp_path):
    assert E.resolve_profile("fal/ideogram-v4-instant") is INSTANT
    with pytest.raises(E.SourceError, match="NVFP4"):
        E.resolve_profile("fal/ideogram-v4-fast")
    # a local folder named "instant" says nothing by itself
    d = tmp_path / "my-ideogram-instant"
    (d / "transformer").mkdir(parents=True)
    (d / "transformer" / "config.json").write_text("{}")
    assert E.is_transformer_source(str(d))
    with pytest.raises(E.SourceError, match="no inference profile"):
        E.resolve_profile(str(d))
    assert E.resolve_profile(str(d), "instant") is INSTANT
    (d / E.PROFILE_SIDECAR).write_text(json.dumps({"profile": "instant"}))
    assert E.resolve_profile(str(d)) is INSTANT


def test_contradictions_are_refused(tmp_path):
    with pytest.raises(E.SourceError, match="documented as the 'instant'"):
        E.resolve_profile("fal/ideogram-v4-instant", "base")
    with pytest.raises(E.SourceError, match="distilled recipe"):
        E.resolve_profile(None, "instant")            # 8 steps on the base weights
    assert E.resolve_profile(None).key == "base"
    with pytest.raises(E.SourceError, match="unknown Ideogram profile"):
        E.resolve_profile(None, "turbo")


def test_profile_metadata():
    m = INSTANT.as_meta()
    assert m == {"profile": "instant", "steps_advised": 8, "guidance": 1.0,
                 "unconditional_branch": False, "mu": 0.0, "std": 1.75}
    assert E.PROFILES["base"].as_meta()["guidance"] == "schedule"


# --- 3. Strict source check ----------------------------------------------------------
def _tiny_cfg():
    tr = tiny_components(0)["transformer"]
    return dict(tr.config), tr


def _expected_of(cfg):
    return {k: cfg[k] for k in ("in_channels", "num_layers", "num_attention_heads",
                                "attention_head_dim", "intermediate_size", "adaln_dim",
                                "llm_features_dim")}


def _save_source(tmp_path, tr, profile="instant", shard="20KB"):
    root = tmp_path / "src"
    tr.save_pretrained(str(root / "transformer"), max_shard_size=shard)
    if profile:
        (root / E.PROFILE_SIDECAR).write_text(json.dumps({"profile": profile}))
    return root


def test_the_published_architecture_is_checked_field_by_field():
    E.check_architecture(dict(E.ARCH))
    bad = dict(E.ARCH, num_layers=36, attention_head_dim=128)
    with pytest.raises(E.SourceError) as e:
        E.check_architecture(bad)
    assert "num_layers=36" in str(e.value) and "attention_head_dim=128" in str(e.value)


def test_a_clean_sharded_folder_passes_and_loads(tmp_path):
    cfg, tr = _tiny_cfg()
    root = _save_source(tmp_path, tr)
    tdir = root / "transformer"
    assert len(list(tdir.glob("*.safetensors"))) > 1, "the fixture must be sharded"
    info = E.check_tensors(str(tdir), cfg, want_dtype="F32")
    assert info["tensors"] == len(tr.state_dict())
    loaded = E.load_source(str(root), dtype=torch.float32, expected=_expected_of(cfg))
    assert loaded.profile is INSTANT
    for k, v in tr.state_dict().items():
        assert torch.equal(loaded.transformer.state_dict()[k], v), k


def _rewrite(tdir, edit):
    """Rewrites every shard through `edit(name, tensor) -> [(name, tensor), ...]`."""
    from safetensors.torch import load_file, save_file
    for fp in sorted(tdir.glob("*.safetensors")):
        sd = load_file(str(fp))
        out = {}
        for k, v in sd.items():
            for nk, nv in edit(k, v):
                out[nk] = nv
        save_file(out, str(fp))


def test_missing_unknown_shape_and_quantized_tensors_are_named(tmp_path):
    cfg, tr = _tiny_cfg()
    names = sorted(tr.state_dict())
    victim, renamed, reshaped, quant = names[0], names[1], names[2], names[3]
    root = _save_source(tmp_path, tr)
    tdir = root / "transformer"

    def edit(k, v):
        if k == victim:
            return []
        if k == renamed:
            return [(k + "_fused", v)]
        if k == reshaped:
            return [(k, v.reshape(-1)[:-1].clone() if v.numel() > 1 else torch.zeros(2))]
        if k == quant:
            return [(k, v.to(torch.float8_e4m3fn))]
        return [(k, v)]
    _rewrite(tdir, edit)
    with pytest.raises(E.SourceError) as e:
        E.check_tensors(str(tdir), cfg)
    msg = str(e.value)
    assert "missing" in msg and victim in msg
    assert "unknown" in msg and renamed + "_fused" in msg
    assert "shape mismatch" in msg and reshaped in msg
    assert "quantized tensors (F8_E4M3)" in msg


def test_a_tensor_present_in_two_shards_is_refused(tmp_path):
    from safetensors.torch import load_file, save_file
    cfg, tr = _tiny_cfg()
    root = _save_source(tmp_path, tr)
    tdir = root / "transformer"
    shards = sorted(tdir.glob("*.safetensors"))
    first = load_file(str(shards[0]))
    k = next(iter(first))
    last = load_file(str(shards[-1]))
    last[k] = first[k]
    save_file(last, str(shards[-1]))
    with pytest.raises(E.SourceError, match="duplicated"):
        E.check_tensors(str(tdir), cfg, want_dtype="F32")


def test_a_refused_source_is_refused_before_any_download(monkeypatch):
    monkeypatch.setattr(E, "fetch_source", lambda *a, **k: pytest.fail("downloaded"))
    with pytest.raises(E.SourceError, match="NVFP4"):
        E.load_source("fal/ideogram-v4-fast")


def test_a_quantized_config_is_refused(tmp_path):
    cfg, tr = _tiny_cfg()
    root = _save_source(tmp_path, tr)
    cpath = root / "transformer" / "config.json"
    c = json.loads(cpath.read_text())
    c["quantization_config"] = {"quant_method": "bitsandbytes"}
    cpath.write_text(json.dumps(c))
    with pytest.raises(E.SourceError, match="bitsandbytes"):
        E.load_source(str(root), dtype=torch.float32, expected=_expected_of(cfg))


# --- 4. Assembly and the emulated paths under the profile ------------------------------
def test_wrap_distilled_then_img2img_and_inpaint_follow_the_profile(tmp_path):
    from diffusers.pipelines.ideogram4.pipeline_ideogram4 import (_logit_normal_sigmas,
                                                                  _resolution_aware_mu)
    cfg, tr = _tiny_cfg()
    loaded = E.load_source(str(_save_source(tmp_path, tr)), dtype=torch.float32,
                           expected=_expected_of(cfg))
    comps = tiny_components(0)
    pipe = E.wrap_distilled(_raw(comps), loaded)
    assert isinstance(pipe._pipe.unconditional_transformer, E.ZeroUnconditionalTransformer)
    seen = {}
    real = pipe._pipe.__class__.__call__

    def spy(self, *a, **kw):
        seen.update(kw)
        out = real(self, *a, **kw)
        seen["timesteps"] = self.scheduler.timesteps.clone()
        return out
    init = Image.new("RGB", (256, 256), (200, 30, 30))
    pipe._pipe.__class__.__call__ = spy
    try:
        pipe(prompt="x", image=init, strength=0.5, generator=_gen(2))
    finally:
        pipe._pipe.__class__.__call__ = real
    full = _logit_normal_sigmas(8, _resolution_aware_mu(256, 256, 0.0), std=1.75)
    assert torch.allclose(seen["timesteps"].float() / 1000.0, full[4:].float(), atol=1e-4)
    assert seen["guidance_schedule"] is None and seen["guidance_scale"] == 1.0
    # inpaint: tokens outside the mask are exactly the init's
    mask = Image.new("L", (256, 256), 0)
    mask.paste(255, (0, 0, 128, 256))
    keep = pipe._encode_tokens(init, 256, 256, "cpu")
    out = pipe(prompt="x", image=init, mask_image=mask, strength=1.0, generator=_gen(2),
               output_type="latent").images
    m = pipe._mask_tokens(mask, 256, 256, "cpu")
    outside = (m[0, :, 0] == 0)
    assert torch.allclose(out[0][outside].float(), keep[0][outside], atol=1e-5)


def test_stop_returns_the_partial_image():
    pipe = _distilled_pipe()

    def stop(p, i, t, kw):
        if i == 2:
            p._interrupt = True
        return kw
    out = pipe(prompt="x", width=256, height=256, generator=_gen(),
               callback_on_step_end=stop).images[0]
    assert out.size == (256, 256)
