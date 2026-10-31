"""cz_flux3.Flux3Pipe, the adapter between the FLUX.3 modular pipeline and the Flux2Klein
calling convention the rest of crispz uses. CPU, tiny random components (tests/flux3_tiny.py),
no download except the optional Qwen3-VL processor test, which skips offline.
"""
import os
import sys

import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cz_flux3  # noqa: E402
from flux3_tiny import TEXT_HIDDEN, SEQ, tiny_pipe, tiny_qwen3_vl, tiny_transformer  # noqa: E402

TMP = os.path.join(os.environ.get("TEMP") or "/tmp", "cz_flux3_adapter")
os.makedirs(TMP, exist_ok=True)


def _gen(seed=1):
    return torch.Generator().manual_seed(seed)


# --- family, repos, schedule (pure functions) ---------------------------------------
def test_family_is_read_from_tensor_names_in_both_layouts():
    bfl3 = ["early_stream_modulations.0.lin.weight", "mode_blocks.3.img_attn.qkv.weight",
            "x_embedders.0.weight"]
    diffusers3 = ["transformer.content_mode_blocks.0.attn.to_q.weight",
                  "txt_mode_blocks.1.norm.weight", "target_timestep_embedder.linear_1.weight"]
    bfl2 = ["double_blocks.0.img_attn.qkv.weight", "single_stream_modulation.lin.weight",
            "double_stream_modulation_img.lin.weight"]
    diffusers2 = ["transformer_blocks.0.attn.to_q.weight",
                  "single_transformer_blocks.0.attn.to_q.weight"]
    assert cz_flux3.family_from_keys(bfl3) == "flux3"
    assert cz_flux3.family_from_keys(diffusers3) == "flux3"
    assert cz_flux3.family_from_keys(bfl2) == "flux2"
    assert cz_flux3.family_from_keys(diffusers2) == "flux2"
    # a ComfyUI prefix changes nothing
    assert cz_flux3.family_from_keys(["model.diffusion_model." + k for k in bfl2]) == "flux2"
    # no marker at all (a VAE, a text encoder) -> unknown, never a guess
    assert cz_flux3.family_from_keys(["encoder.conv_in.weight", "__metadata__"]) is None
    # FLUX.2's 'single_stream_modulation.' must not match inside FLUX.3's
    # 'single_stream_modulations.' (the plural is the FLUX.3 one)
    assert cz_flux3.family_from_keys(["single_stream_modulations.0.lin.weight"]) == "flux3"


def test_repo_names_decide_the_variant_and_the_family():
    assert cz_flux3.is_flux3_repo(cz_flux3.FLUX3_DISTILLED)
    assert cz_flux3.is_flux3_repo("unsloth/flux-3-image-klein-FP8")
    assert not cz_flux3.is_flux3_repo("black-forest-labs/FLUX.2-klein-4B")
    assert not cz_flux3.is_flux3_repo("")
    assert cz_flux3.variant_of(cz_flux3.FLUX3_DISTILLED) == "distilled"
    assert cz_flux3.variant_of(cz_flux3.FLUX3_BASE) == "base"
    assert cz_flux3.DEFAULTS["distilled"] == {"steps": 16, "guidance": 1.0}
    assert cz_flux3.DEFAULTS["base"] == {"steps": 50, "guidance": 3.5}


def test_img2img_schedule_is_a_fraction_of_the_steps_and_always_one_step():
    """diffusers' rule (FluxImg2Img.get_timesteps): strength 0.5 runs half the steps from
    the shifted sigma of that index. Reading the strength as a sigma ran ONE step at 0.35
    (every shifted sigma but the last is above 0.5), measured on the 5090 e2e."""
    n, shift = 16, 6.93
    full, s0, run = cz_flux3.img2img_schedule(n, 1.0, shift)
    assert run == n and len(full) == n and s0 == pytest.approx(1.0)
    tail, s0, run = cz_flux3.img2img_schedule(n, 0.5, shift)
    assert run == 8 and len(tail) == 8
    assert s0 == pytest.approx(cz_flux3.shifted_sigmas(n, shift)[8])
    # the tail is the UNSHIFTED linspace: the scheduler applies the shift itself
    assert tail[0] == pytest.approx(0.5) and tail[-1] == pytest.approx(1.0 / n)
    _, _, run = cz_flux3.img2img_schedule(n, 0.35, shift)
    assert run == 6                       # int(16 - 5.6) = 10 skipped
    _, _, run = cz_flux3.img2img_schedule(4, 0.05, shift)
    assert run == 1                       # a 0.05 denoise still touches the image
    assert cz_flux3.shifted_sigmas(2, shift)[0] == pytest.approx(1.0)


def test_build_blocks_drops_the_upsampler_and_the_encoder_and_hooks_both_loops():
    for base in (False, True):
        blocks = cz_flux3.build_blocks(base=base)
        assert "prompt_upsampler" not in blocks.sub_blocks
        assert "text_encoder" not in blocks.sub_blocks
        hooks = cz_flux3._find_hooks(blocks)
        assert len(hooks) == 2, base       # the text-to-image AND the image-conditioned loop
        assert all(type(h).__name__ == "CrispzStepHook" for h in hooks)


# --- the adapter on tiny components ------------------------------------------------
def test_txt2img_gives_the_asked_size_and_calls_back_every_step():
    pipe = tiny_pipe()
    calls = []

    def cb(p, i, t, kw):
        calls.append((i, float(t), tuple(kw["latents"].shape)))
        assert p is pipe
        return kw

    out = pipe(prompt="a cube", width=64, height=48, num_inference_steps=3, generator=_gen(),
               callback_on_step_end=cb).images
    assert len(out) == 1 and isinstance(out[0], Image.Image) and out[0].size == (64, 48)
    assert [c[0] for c in calls] == [0, 1, 2]
    assert calls[0][1] == pytest.approx(1000.0)          # the first timestep is sigma 1
    assert calls[0][2] == (1, (64 // 16) * (48 // 16), 128)   # packed tokens, 128 channels
    # sizes off the 16 grid are floored, never padded
    out = pipe(prompt="a cube", width=70, height=50, num_inference_steps=1, generator=_gen()).images
    assert out[0].size == (64, 48)


def test_multi_reference_edit_takes_a_list_of_images():
    pipe = tiny_pipe()
    r1 = Image.new("RGB", (64, 48), (200, 30, 30))
    r2 = Image.new("RGB", (48, 64), (30, 200, 30))
    out = pipe(prompt="put <ref_image_1> on <ref_image_0>", image=[r1, r2],
               num_inference_steps=2, generator=_gen()).images
    assert len(out) == 1 and out[0].size[0] % 16 == 0 and out[0].size[1] % 16 == 0
    out1 = pipe(prompt="x", image=r1, num_inference_steps=1, generator=_gen()).images
    assert out1[0].size == (64, 48)       # one reference: the output follows its size


def test_img2img_runs_the_tail_of_the_schedule_only():
    pipe = tiny_pipe()
    init = Image.new("RGB", (64, 48), (120, 120, 200))
    calls = []
    out = pipe(prompt="x", image=init, strength=0.6, num_inference_steps=8, generator=_gen(),
               callback_on_step_end=lambda p, i, t, kw: calls.append(i)).images
    assert out[0].size == (64, 48)
    _, _, expected = cz_flux3.img2img_schedule(8, 0.6, pipe.scheduler.config.shift)
    assert len(calls) == expected and 0 < expected < 8
    # the init image is the start point, NOT a reference: no image conditioning tokens
    # went in (a reference would have produced a different latent count in the hook)
    full = pipe(prompt="x", image=init, strength=1.0, num_inference_steps=2, generator=_gen(),
                callback_on_step_end=lambda p, i, t, kw: calls.append(i)).images
    assert full[0].size == (64, 48)


def test_inpaint_keeps_the_unmasked_tokens_and_a_white_mask_is_img2img():
    pipe = tiny_pipe()
    init = Image.new("RGB", (64, 48), (120, 120, 200))
    mask = Image.new("L", (64, 48), 0)
    mask.paste(255, (32, 0, 64, 48))      # repaint the right half
    seen = {}

    def cb(p, i, t, kw):
        seen["latents"] = kw["latents"].clone()
        seen["ctx"] = dict(p._step_ctx)
        return kw

    pipe(prompt="x", image=init, mask_image=mask, strength=1.0, num_inference_steps=3,
         generator=_gen(), callback_on_step_end=cb)
    ctx = seen["ctx"]
    assert ctx["mask"] is not None and ctx["x0"] is not None
    m = ctx["mask"][0, :, 0]
    assert m.sum() == 2 * 3 and m.numel() == 4 * 3        # 2 of 4 columns on 3 rows
    # after the LAST step the next sigma is 0: outside the mask the tokens ARE x0
    keep = (m == 0)
    assert torch.allclose(seen["latents"][0][keep], ctx["x0"][0][keep].to(seen["latents"].dtype))
    # a thin stroke: one pixel repaints its whole 16 px token (max-pool, never lost)
    thin = Image.new("L", (64, 48), 0)
    thin.putpixel((5, 5), 255)
    assert cz_flux3.Flux3Pipe._mask_tokens(thin, 64, 48, "cpu").sum() == 1
    # an all-white mask is a plain img2img: no blending context at all
    pipe(prompt="x", image=init, mask_image=Image.new("L", (64, 48), 255), strength=0.8,
         num_inference_steps=2, generator=_gen(), callback_on_step_end=cb)
    assert "mask" not in seen["ctx"]


def test_an_interrupt_stops_the_loop_and_still_returns_an_image(caplog):
    import logging
    caplog.set_level(logging.ERROR, logger="diffusers.modular_pipelines.modular_pipeline")
    pipe = tiny_pipe()
    steps = []

    def cb(p, i, t, kw):
        steps.append(i)
        if i == 1:
            p._interrupt = True
        return kw

    out = pipe(prompt="x", width=64, height=48, num_inference_steps=6, generator=_gen(),
               callback_on_step_end=cb).images
    assert steps == [0, 1]
    assert len(out) == 1 and out[0].size == (64, 48)   # decoded from the last latents
    assert pipe._step_ctx is None                       # the context never leaks
    # diffusers logs every exception crossing a block as an ERROR with a traceback, once
    # per nesting level: a wanted Stop must not print four of them
    assert not [r for r in caplog.records if "Error in block" in r.getMessage()], caplog.text
    assert cz_flux3._QuietInterrupt.active is False
    # the flag is reset by the next call
    out = pipe(prompt="x", width=64, height=48, num_inference_steps=2, generator=_gen()).images
    assert len(out) == 1


def test_a_list_of_prompts_gives_one_image_each():
    pipe = tiny_pipe()
    out = pipe(prompt=["a", "b", "c"], width=32, height=32, num_inference_steps=1,
               generator=[_gen(1), _gen(2), _gen(3)]).images
    assert len(out) == 3 and all(im.size == (32, 32) for im in out)


def test_register_modules_swaps_the_transformer_inside_the_modular_pipeline():
    pipe = tiny_pipe()
    old = pipe.transformer
    new = tiny_transformer()
    pipe.register_modules(transformer=new)
    assert pipe.transformer is new and pipe._mp.transformer is new and old is not new
    assert "transformer" not in pipe.__dict__          # no shadow copy on the adapter
    # the swapped model is the one that computes
    out = pipe(prompt="x", width=32, height=32, num_inference_steps=1, generator=_gen()).images
    assert len(out) == 1
    # and the other DiffusionPipeline names still answer
    assert pipe.config.is_distilled is True
    assert "transformer" in pipe.components and pipe.dtype == torch.float32


def test_the_base_variant_runs_a_real_cfg_with_a_negative():
    pipe = tiny_pipe(base=True)
    assert pipe.config.is_distilled is False
    type(pipe).encode_calls = 0
    out = pipe(prompt="x", negative_prompt="ugly", guidance_scale=3.5, width=32, height=32,
               num_inference_steps=2, generator=_gen()).images
    assert len(out) == 1
    assert type(pipe).encode_calls == 2                # the prompt AND the negative
    assert pipe._mp.guider.guidance_scale == pytest.approx(3.5)
    # guidance 1.0: no unconditional branch, the negative is not even encoded
    type(pipe).encode_calls = 0
    pipe(prompt="x", negative_prompt="ugly", guidance_scale=1.0, width=32, height=32,
         num_inference_steps=1, generator=_gen())
    assert type(pipe).encode_calls == 1
    assert pipe._mp.guider.guidance_scale == pytest.approx(1.0)   # per call, never sticky


def test_load_lora_weights_lifts_the_offload_hooks_and_puts_them_back(monkeypatch):
    """diffusers' LoRA loader re-enables the offload through `_pipeline.enable_model_cpu_
    offload()`, which the modular pipeline lacks (the app's first Outpaint-LoRA run,
    2026-10-29): the adapter removes its hooks before the call and restores them after."""
    pipe = tiny_pipe()
    pipe.enable_model_cpu_offload(device="cpu")
    seen = {}

    def fake_load(*a, **kw):
        seen["hooks_during"] = list(pipe._all_hooks)
        seen["args"] = (a, kw)
    monkeypatch.setattr(pipe._mp, "load_lora_weights", fake_load, raising=False)
    pipe.load_lora_weights("some/dir", weight_name="x.safetensors", adapter_name="cz_lora_0")
    assert seen["hooks_during"] == []                       # nothing for diffusers to touch
    assert seen["args"][1]["adapter_name"] == "cz_lora_0"
    assert pipe._all_hooks and pipe._execution_device.type == "cpu"   # offload back in place
    # a failure inside the loader still restores the offload (no pipe left on the CPU)
    monkeypatch.setattr(pipe._mp, "load_lora_weights",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("bad lora")), raising=False)
    with pytest.raises(ValueError):
        pipe.load_lora_weights("some/dir", weight_name="y.safetensors")
    assert pipe._all_hooks
    # without offload: a plain pass-through
    pipe.remove_all_hooks()
    monkeypatch.setattr(pipe._mp, "load_lora_weights", fake_load, raising=False)
    pipe.load_lora_weights("some/dir", weight_name="z.safetensors")
    assert pipe._all_hooks == []


def test_cpu_offload_names_exist_and_the_hooks_free_cleanly():
    pipe = tiny_pipe()
    pipe.enable_model_cpu_offload(device="cpu")        # accelerate hooks, CPU to CPU here
    assert pipe._all_hooks and pipe._execution_device.type == "cpu"
    out = pipe(prompt="x", width=32, height=32, num_inference_steps=1, generator=_gen()).images
    assert len(out) == 1
    pipe.remove_all_hooks()
    assert pipe._all_hooks == [] and pipe._offload_device is None
    pipe.to("cpu")
    pipe.maybe_free_model_hooks()                       # a no-op without hooks, never an error


# --- text encoding against the real Qwen3-VL recipe ---------------------------------
def test_encode_prompt_stacks_eight_hidden_states_padded_to_80():
    try:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
    except Exception as e:          # offline, no cache: the recipe is diffusers', skip
        pytest.skip(f"Qwen3-VL processor unavailable: {type(e).__name__}")
    pipe = cz_flux3.Flux3Pipe(tiny_pipe()._mp, tiny_qwen3_vl(), proc, repo="tiny")
    (emb,) = pipe.encode_prompt("a red cube on a table")
    assert tuple(emb.shape) == (1, SEQ, 8 * TEXT_HIDDEN)
    # the same prompt encodes the same way (the cache in cz_pipeline relies on it)
    (emb2,) = pipe.encode_prompt("a red cube on a table")
    assert torch.equal(emb, emb2)
    # a long prompt grows by whole blocks of 80
    (long,) = pipe.encode_prompt("word " * 120)
    assert long.shape[1] % 80 == 0 and long.shape[1] > SEQ


# --- loading plumbing (no weights: the loader is intercepted) ------------------------
def _fake_single_file(monkeypatch):
    from diffusers import Flux3Transformer2DModel
    seen = {}

    def fake(cls, path, **kw):
        seen["path"], seen["kw"] = path, kw
        return "loaded"
    monkeypatch.setattr(Flux3Transformer2DModel, "from_single_file", classmethod(fake))
    return seen


def test_load_transformer_keeps_a_gguf_quantized(monkeypatch):
    from diffusers import GGUFQuantizationConfig
    seen = _fake_single_file(monkeypatch)
    out = cz_flux3.load_transformer("repo/x", torch.bfloat16, single_file="C:/m/flux3-Q4_K_M.gguf")
    assert out == "loaded" and seen["path"].endswith(".gguf")
    q = seen["kw"]["quantization_config"]
    assert isinstance(q, GGUFQuantizationConfig) and q.compute_dtype == torch.bfloat16
    # the architecture always comes from the base repo
    assert seen["kw"]["config"] == "repo/x" and seen["kw"]["subfolder"] == "transformer"
    assert cz_flux3.is_gguf("a.GGUF") and not cz_flux3.is_gguf("a.safetensors")
    assert cz_flux3._bfl_quant_of("C:/m/flux3-Q4_K_M.gguf") is None    # never opened


def test_load_transformer_tells_bfl_fp8r_and_nvfp4_from_a_plain_file(monkeypatch):
    from safetensors.torch import save_file
    from diffusers import BFLQuantizationConfig
    seen = _fake_single_file(monkeypatch)
    plain = os.path.join(TMP, "plain.safetensors")
    save_file({"mode_blocks.0.img_attn.qkv.weight": torch.zeros(2, 2)}, plain)
    # BFL's fp8r: float8 weights next to the scale; nvfp4: uint8-packed + weight_scale_2
    fp8r = os.path.join(TMP, "fp8r.safetensors")
    save_file({"mode_blocks.0.img_attn.qkv.weight": torch.zeros(2, 2).to(torch.float8_e4m3fn),
               "mode_blocks.0.img_attn.qkv.weight_scale": torch.zeros(1)}, fp8r)
    nvfp4 = os.path.join(TMP, "nvfp4.safetensors")
    save_file({"mode_blocks.0.img_attn.qkv.weight": torch.zeros(2, 2, dtype=torch.uint8),
               "mode_blocks.0.img_attn.qkv.weight_scale": torch.zeros(1),
               "mode_blocks.0.img_attn.qkv.weight_scale_2": torch.zeros(1)}, nvfp4)
    # ComfyUI's comfy-quant INT8 names its scales the same way: int8 weights + blobs,
    # and it is NOT BFL's layout (it goes through the dequant loader instead)
    comfy = os.path.join(TMP, "comfy_int8.safetensors")
    save_file({"mode_blocks.0.img_attn.qkv.weight": torch.zeros(2, 2, dtype=torch.int8),
               "mode_blocks.0.img_attn.qkv.weight_scale": torch.ones(1),
               "mode_blocks.0.img_attn.qkv.comfy_quant": torch.zeros(4, dtype=torch.uint8)}, comfy)
    assert cz_flux3._bfl_quant_of(plain) is None
    assert cz_flux3._bfl_quant_of(fp8r) == "fp8r"
    assert cz_flux3._bfl_quant_of(nvfp4) == "nvfp4"
    assert cz_flux3._bfl_quant_of(comfy) is None
    cz_flux3.load_transformer("repo/x", torch.bfloat16, single_file=plain)
    assert "quantization_config" not in seen["kw"]
    cz_flux3.load_transformer("repo/x", torch.bfloat16, single_file=fp8r)
    assert isinstance(seen["kw"]["quantization_config"], BFLQuantizationConfig)


def test_load_transformer_keeps_the_qk_norms_in_fp32_on_a_single_file():
    """from_pretrained keeps norm_q / norm_k in float32 (diffusers' _keep_in_fp32_modules);
    diffusers' from_single_file casts them to bf16. The loader puts them back, so a
    CivitAI bf16 file renders through the same kernels as the repo (5090, 2026-10-28)."""
    from safetensors.torch import save_file
    d = os.path.join(TMP, "fp32_norms")
    os.makedirs(os.path.join(d, "transformer"), exist_ok=True)
    m = tiny_transformer()
    m.save_pretrained(os.path.join(d, "transformer"))
    single = os.path.join(d, "single.safetensors")
    save_file({k: v.contiguous() for k, v in m.state_dict().items()}, single)
    out = cz_flux3.load_transformer(d, torch.bfloat16, single_file=single)
    norms = {n: p.dtype for n, p in out.named_parameters() if "norm_q" in n or "norm_k" in n}
    assert norms and set(norms.values()) == {torch.float32}, norms
    others = {p.dtype for n, p in out.named_parameters() if n not in norms}
    assert others == {torch.bfloat16}, others
