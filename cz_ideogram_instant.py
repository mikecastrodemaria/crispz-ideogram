"""Experimental BF16 Ideogram Instant support, without a second denoiser.

The single-file converter is deliberately strict: unknown/quantized tensors fail
before allocating the model. Sampling follows the author's ComfyUI workflow:
Euler, simple sigmas, rational flow shift 5.25, CFG 1 (one conditional pass).
Real checkpoint image quality must be validated separately from CPU unit tests.
"""

import math
import os

import torch
from diffusers import Ideogram4Pipeline, Ideogram4Transformer2DModel
from diffusers.pipelines.ideogram4.pipeline_ideogram4 import (
    _expand_tensor_to_effective_batch,
)
from diffusers.pipelines.ideogram4.pipeline_output import Ideogram4PipelineOutput

REPO = "Winnougan/Ideogram_Instant_NSFW"
FILENAME = "Fal_Ideo_4_NSFW_BF16.safetensors"
STEPS = 8
SHIFT = 5.25


def is_instant_name(path):
    name = str(path or "").lower().replace("_", "-")
    return "ideogram-instant" in name or "fal-ideo-4-nsfw" in name


def tensor_destinations(name, shape):
    """Original/ComfyUI fused QKV -> diffusers split projections (views, no copy)."""
    for prefix in ("model.diffusion_model.", "diffusion_model.", "transformer."):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    if name.endswith(".attention.qkv.weight"):
        if len(shape) != 2 or shape[0] != 3 * shape[1]:
            raise ValueError(f"Invalid fused QKV shape: {name}: {shape}")
        return [name.replace(".qkv.weight", f".to_{p}.weight") for p in "qkv"]
    return [name.replace(".attention.o.weight", ".attention.to_out.0.weight")]


def checkpoint_plan(path, config=None):
    """Validate names, shapes, dtype and complete coverage without reading weights."""
    from accelerate import init_empty_weights
    from safetensors import safe_open

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    with init_empty_weights():
        model = Ideogram4Transformer2DModel(**(config or {}))
    expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    plan, seen = {}, set()
    with safe_open(path, framework="pt", device="cpu") as f:
        for name in f.keys():
            tensor = f.get_slice(name)
            shape = tuple(tensor.get_shape())
            if tensor.get_dtype() not in ("BF16", "F16", "F32"):
                raise ValueError(f"{name}: {tensor.get_dtype()} is not supported. "
                                 "Use the BF16 file, not INT8 ConvRot / FP8 / GGUF.")
            dests = tensor_destinations(name, shape)
            target_shape = (shape[0] // 3, *shape[1:]) if len(dests) == 3 else shape
            for dest in dests:
                if dest in seen:
                    raise ValueError(f"Duplicate checkpoint tensor: {dest}")
                if dest not in expected:
                    raise ValueError(f"Unsupported checkpoint tensor: {name} -> {dest}")
                if target_shape != expected[dest]:
                    raise ValueError(f"Shape mismatch for {dest}: "
                                     f"{target_shape}, expected {expected[dest]}")
                seen.add(dest)
            plan[name] = dests
    missing = sorted(set(expected) - seen)
    if missing:
        raise ValueError(f"Incomplete Ideogram checkpoint: missing {missing[:8]}")
    return plan


def load_checkpoint(path, dtype=torch.bfloat16, config=None):
    from accelerate import init_empty_weights
    from safetensors import safe_open

    plan = checkpoint_plan(path, config)
    with init_empty_weights():
        model = Ideogram4Transformer2DModel(**(config or {}))
    state = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for source, destinations in plan.items():
            tensor = f.get_tensor(source).to(dtype=dtype)
            values = tensor.chunk(3, dim=0) if len(destinations) == 3 else (tensor,)
            state.update(zip(destinations, values))
    model.load_state_dict(state, strict=True, assign=True)
    model.eval()
    model._cz_ideogram_instant = is_instant_name(path)
    return model


def simple_sigmas(steps, shift=SHIFT, device=None):
    if int(steps) < 1 or not math.isfinite(float(shift)) or shift <= 0:
        raise ValueError("Instant needs positive steps and a finite positive flow shift")
    # ComfyUI's simple schedule selects evenly spaced entries of the 1000-step
    # model sigma table, from largest to smallest. Terminal zero is added by Euler.
    indices = torch.floor(torch.arange(int(steps), device=device) * 1000 / int(steps))
    sigmas = (1000 - indices) / 1000
    return shift * sigmas / (1 + (shift - 1) * sigmas)


class IdeogramInstantPipeline(Ideogram4Pipeline):
    """Reuse Ideogram conditioning/decoding, with a single-model Euler loop.

    The inherited constructor keeps diffusers' component discovery intact. Passing
    unconditional_transformer=None avoids downloading/loading that component.
    """

    _optional_components = ["prompt_enhancer_head", "unconditional_transformer"]
    model_cpu_offload_seq = "prompt_enhancer_head->text_encoder->transformer->vae"

    @property
    def _execution_device(self):
        device = getattr(self, "_cz_execution_device", None)
        return device if device is not None else super()._execution_device

    @torch.no_grad()
    def __call__(self, prompt, height=1024, width=1024, num_inference_steps=STEPS,
                 guidance_scale=None, guidance_schedule=None, num_images_per_prompt=1,
                 generator=None, latents=None, output_type="pil", return_dict=True,
                 callback_on_step_end=None, callback_on_step_end_tensor_inputs=None,
                 max_sequence_length=2048, prompt_upsampling=False,
                 attention_kwargs=None, **kwargs):
        if kwargs:
            raise TypeError(f"Unsupported Instant arguments: {sorted(kwargs)}")
        if guidance_scale is not None and float(guidance_scale) != 1.0:
            raise ValueError("Instant currently supports CFG 1 only")
        if guidance_schedule is not None and any(float(g) != 1.0 for g in guidance_schedule):
            raise ValueError("Instant requires a constant guidance schedule of 1")
        if height % 16 or width % 16 or min(height, width) < 256 or max(height, width) > 2048:
            raise ValueError("Instant dimensions must be multiples of 16 in 256..2048")
        if max(height, width) / min(height, width) > 6:
            raise ValueError("Instant supports aspect ratios up to 6:1")
        if not isinstance(prompt, (str, list)) or not prompt or int(num_images_per_prompt) < 1:
            raise ValueError("Provide a prompt and at least one image per prompt")
        if callback_on_step_end_tensor_inputs not in (None, ["latents"]):
            raise ValueError("Only latents are available to the Instant callback")
        device = self._execution_device
        self._interrupt = False
        self._guidance_scale = 1.0
        self._attention_kwargs = attention_kwargs
        if prompt_upsampling:
            prompt = self.upsample_prompt(prompt, height=height, width=width,
                                         generator=generator, device=device)
        batch = 1 if isinstance(prompt, str) else len(prompt)
        count = batch * int(num_images_per_prompt)
        gh, gw = height // 16, width // 16
        features, positions, segments, indicator = self.encode_prompt(
            prompt=prompt, grid_h=gh, grid_w=gw,
            max_sequence_length=max_sequence_length, device=device)
        features, positions, segments, indicator = [
            _expand_tensor_to_effective_batch(x, batch, num_images_per_prompt)
            for x in (features, positions, segments, indicator)]
        features = features.to(self.transformer.dtype)
        sigmas = simple_sigmas(num_inference_steps, device=device)
        self.scheduler.set_timesteps(sigmas=sigmas.tolist(), device=device)
        self._num_timesteps = len(self.scheduler.timesteps)
        latents = self.prepare_latents(
            batch_size=count, num_image_tokens=gh * gw,
            latent_dim=self.transformer.config.in_channels, dtype=torch.float32,
            device=device, generator=generator, latents=latents)
        padding = torch.zeros(count, max_sequence_length, latents.shape[-1],
                              device=device, dtype=self.transformer.dtype)
        with self.progress_bar(total=self._num_timesteps) as progress:
            for i, t in enumerate(self.scheduler.timesteps):
                if self.interrupt:
                    break
                model_time = (1 - t.float() / self.scheduler.config.num_train_timesteps)
                model_time = model_time.expand(count).to(self.transformer.dtype)
                velocity = self.transformer(
                    hidden_states=torch.cat((padding, latents.to(padding.dtype)), dim=1),
                    timestep=model_time, encoder_hidden_states=features,
                    position_ids=positions, segment_ids=segments, indicator=indicator,
                    attention_kwargs=self.attention_kwargs, return_dict=False)[0]
                velocity = velocity[:, max_sequence_length:].float()
                latents = self.scheduler.step(-velocity, t, latents, return_dict=False)[0]
                if callback_on_step_end:
                    result = callback_on_step_end(self, i, t, {"latents": latents})
                    if result:
                        latents = result.get("latents", latents)
                progress.update()
        if output_type == "latent":
            images = latents
        else:
            mean = self.vae.bn.running_mean.to(latents).view(1, 1, -1)
            std = (self.vae.bn.running_var + self.vae.config.batch_norm_eps).sqrt().to(latents)
            z = latents * std.view(1, 1, -1) + mean
            channels = z.shape[-1] // 4
            z = z.view(count, gh, gw, 2, 2, channels).permute(0, 5, 1, 3, 2, 4)
            z = z.reshape(count, channels, gh * 2, gw * 2)
            decoded = self.vae.decode(z.to(self.vae.dtype), return_dict=False)[0]
            images = self.image_processor.postprocess(decoded.float(), output_type=output_type)
        self.maybe_free_model_hooks()
        return Ideogram4PipelineOutput(images=images) if return_dict else (images,)
