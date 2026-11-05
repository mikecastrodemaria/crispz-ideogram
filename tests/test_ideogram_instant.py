"""Offline Instant tests: complete conversion, one denoiser, sampling and switching."""
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest
import torch
from safetensors.torch import save_file

import cz_ideogram
from cz_ideogram_instant import (
    IdeogramInstantPipeline, checkpoint_plan, load_checkpoint, simple_sigmas,
)
from ideogram_tiny import tiny_components


def original_layout(model):
    state = dict(model.state_dict())
    for i in range(model.config.num_layers):
        prefix = f"layers.{i}.attention."
        state[prefix + "qkv.weight"] = torch.cat([
            state.pop(prefix + f"to_{p}.weight") for p in "qkv"], dim=0)
        state[prefix + "o.weight"] = state.pop(prefix + "to_out.0.weight")
    return {"model.diffusion_model." + k: v.contiguous() for k, v in state.items()}


@pytest.fixture(scope="module")
def components():
    return tiny_components()


def test_converter_roundtrip_and_strict_rejection(tmp_path, components):
    model = components["transformer"]
    state = original_layout(model)
    path = tmp_path / "Fal_Ideo_4_NSFW_BF16.safetensors"
    save_file(state, str(path))
    loaded = load_checkpoint(str(path), torch.float32, config=dict(model.config))
    assert loaded._cz_ideogram_instant
    for name, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name]), name
    state.pop(next(iter(state)))
    save_file(state, str(path))
    with pytest.raises(ValueError, match="Incomplete"):
        checkpoint_plan(str(path), config=dict(model.config))
    state = original_layout(model)
    state["unknown.weight"] = torch.zeros(2)
    save_file(state, str(path))
    with pytest.raises(ValueError, match="Unsupported checkpoint tensor"):
        checkpoint_plan(str(path), config=dict(model.config))
    state = original_layout(model)
    key = next(iter(state))
    state[key] = state[key].to(torch.int8)
    save_file(state, str(path))
    with pytest.raises(ValueError, match="ConvRot"):
        checkpoint_plan(str(path), config=dict(model.config))


def test_simple_schedule_reference():
    # 8 simple steps select 1000,875,...,125 from the sigma table.
    source = torch.arange(1000, 0, -125) / 1000
    expected = 5.25 * source / (1 + 4.25 * source)
    assert torch.equal(simple_sigmas(8), expected)
    assert simple_sigmas(1).tolist() == [1.0]
    assert torch.all(simple_sigmas(10).diff() < 0)
    with pytest.raises(ValueError):
        simple_sigmas(0)


def test_poster_smoke_check_rejects_noise():
    from PIL import Image, ImageDraw
    import numpy as np
    from tools.ideogram_instant import check_test_poster
    poster = Image.new("RGB", (256, 256), "white")
    ImageDraw.Draw(poster).rectangle((40, 100, 216, 150), fill="red")
    assert check_test_poster(poster)["passed"]
    noise = Image.fromarray(np.random.default_rng(42).integers(0, 256, (256, 256, 3), dtype=np.uint8))
    assert not check_test_poster(noise)["passed"]


def test_single_denoiser_determinism_decode_callback(components):
    parts = dict(components, unconditional_transformer=None)
    raw = IdeogramInstantPipeline(**parts)
    raw.transformer._cz_ideogram_instant = True
    wrapped = cz_ideogram.IdeogramPipe(raw, repo="tiny-instant")
    assert wrapped.variant == "ideogram_instant"
    assert raw.unconditional_transformer is None
    seen, forwards = [], []
    handle = raw.transformer.register_forward_pre_hook(lambda *a: forwards.append(1))
    def callback(pipe, i, t, kw):
        seen.append(i)
        return kw
    def run():
        return wrapped(prompt="A red poster that says HELLO", width=256, height=256,
                       num_inference_steps=2, guidance_scale=7,
                       generator=torch.Generator().manual_seed(42), output_type="latent",
                       callback_on_step_end=callback).images
    try:
        first, second = run(), run()
        assert torch.equal(first, second)
        assert seen == [0, 1, 0, 1]
        assert len(forwards) == 4  # exactly ONE transformer forward per step
        image = wrapped(prompt="HELLO", width=256, height=256, num_inference_steps=1,
                        generator=torch.Generator().manual_seed(1)).images[0]
        assert image.size == (256, 256)
        def stop(pipe, i, t, kw):
            pipe._interrupt = True
            return kw
        forwards.clear()
        wrapped(prompt="x", width=256, height=256, num_inference_steps=3,
                output_type="latent", callback_on_step_end=stop)
        assert len(forwards) == 1
        with pytest.raises(NotImplementedError, match="text to image"):
            wrapped(prompt="x", image=image, strength=.5)
    finally:
        handle.remove()
        del raw.transformer._cz_ideogram_instant


def test_profile_and_pipeline_switching(monkeypatch):
    import cz_core
    import cz_pipeline as p
    assert cz_core.profile_for_model("Fal_Ideo_4_NSFW_BF16.safetensors") == (8, 1.0)
    assert cz_core.profile_for_model("ideogram-ai/ideogram-4-nf4-diffusers") == (48, 7.0)
    monkeypatch.setattr(p, "ZIMAGE_TRANSFORMER", "Fal_Ideo_4_NSFW_BF16.safetensors")
    assert p._swap_transformer(None) is False
    monkeypatch.setattr(p, "ZIMAGE_TRANSFORMER", None)
    monkeypatch.setattr(p, "_LOADED_KEY", (p.BASE_REPO, "Fal_Ideo_4_NSFW_BF16.safetensors", "model", "nf4"))
    assert p._swap_transformer(None) is False


def test_metadata_records_actual_instant_sampling(monkeypatch):
    import cz_pipeline as p
    monkeypatch.setattr(p, "ZIMAGE_TRANSFORMER", "Fal_Ideo_4_NSFW_BF16.safetensors")
    meta = p._gen_meta("txt2img", "HELLO", guidance=7)
    assert meta["guidance"] == 1
    assert meta["sampler"] == "euler/simple"
    assert meta["flow_shift"] == 5.25


def test_offload_stages_encoder_and_transformer():
    """Never place both large models on the GPU together during prompt encoding."""
    from types import SimpleNamespace
    calls = []
    class Tracked(torch.nn.Linear):
        def __init__(self, name):
            super().__init__(1, 1)
            self.label = name
        def to(self, device):
            calls.append((self.label, str(device)))
            return self
    tr, enc = Tracked("transformer"), Tracked("encoder")
    tr._cz_ideogram_instant = True
    inner = SimpleNamespace(transformer=tr, text_encoder=enc, vae=None,
                            remove_all_hooks=lambda: None)
    wrapped = cz_ideogram.IdeogramPipe(inner)
    wrapped.enable_model_cpu_offload(device="cuda")
    calls.clear()
    wrapped._encoder_in()
    wrapped._encoder_out()
    assert calls == [("transformer", "cpu"), ("encoder", "cuda"),
                     ("encoder", "cpu"), ("transformer", "cuda")]
    wrapped.remove_all_hooks()


def test_load_skips_unconditional_weights(monkeypatch, components):
    transformer = components["transformer"]
    transformer._cz_ideogram_instant = True
    captured = {}
    def fake_load(repo, **kw):
        captured.update(kw)
        return IdeogramInstantPipeline(**dict(components, transformer=kw["transformer"],
                                              scheduler=kw["scheduler"], unconditional_transformer=None))
    monkeypatch.setattr(IdeogramInstantPipeline, "from_pretrained", fake_load)
    try:
        wrapped = cz_ideogram.load(transformer=transformer)
        assert "unconditional_transformer" in captured
        assert captured["unconditional_transformer"] is None
        assert wrapped.variant == "ideogram_instant"
        assert captured["scheduler"].config.shift == 1
    finally:
        del transformer._cz_ideogram_instant


def test_diffusers_component_loader_does_not_load_either_denoiser(tmp_path, monkeypatch, components):
    """Use the real from_pretrained path, not just a mock of engine.load."""
    from diffusers import Ideogram4Pipeline, Ideogram4Transformer2DModel
    Ideogram4Pipeline(**components).save_pretrained(str(tmp_path))
    def forbidden(*args, **kwargs):
        raise AssertionError("The checkpoint override must skip both repository denoisers")
    monkeypatch.setattr(Ideogram4Transformer2DModel, "from_pretrained", forbidden)
    raw = IdeogramInstantPipeline.from_pretrained(
        str(tmp_path), transformer=components["transformer"], unconditional_transformer=None)
    assert raw.transformer is components["transformer"]
    assert raw.unconditional_transformer is None
