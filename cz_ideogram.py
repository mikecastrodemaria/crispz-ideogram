"""crispz-ideogram - the Ideogram 4.0 engine (diffusers Ideogram4Pipeline).

Ideogram 4.0 (open weights, 03/06/2026, Ideogram 4 Non-Commercial licence) is served by
diffusers 0.42.1 as a classic DiffusionPipeline, `Ideogram4Pipeline`:

  - TWO transformers: `transformer` reads the text, `unconditional_transformer` denoises
    with zeroed text features, and the two velocities are blended every step
    (asymmetric CFG). So there is NO negative prompt: the unconditional branch has no
    text input at all.
  - text encoder Qwen3-VL-8B, 13 hidden states concatenated (llm_features_dim 53248);
  - VAE AutoencoderKLFlux2 (the FLUX.2 / FLUX.3 one), packed tokens (B, h*w, 128);
  - logit-normal sigma schedule, resolution-aware, 48 steps by default, and a PER-STEP
    guidance schedule: 7.0 for 45 steps then 3.0 for the last 3;
  - native 2K (multiples of 16 from 256 to 2048 a side, ratio up to 6:1);
  - trained on structured JSON captions only (plain text works, less well).

`IdeogramPipe` puts it behind the calling convention cz_pipeline was written against
(the one cz_flux3.Flux3Pipe also follows):

    pipe(prompt=, image=, mask_image=, strength=, width=, height=, num_inference_steps=,
         generator=, guidance_scale=, callback_on_step_end=, output_type=).images

Lot 1 (this file) covers text-to-image. `strength` / `mask_image` / `image` raise a
named NotImplementedError: img2img and inpaint are emulated in lot 3 (SDEdit + latent
blending, the cz_flux3 techniques), editing goes through the cloud in lot 4.

What the adapter adds over the raw pipeline:
  - the guidance slider keeps the SHAPE of the official schedule (a 6.25 % tail at 3/7
    of the peak) for any number of steps; the raw pipeline refuses a schedule whose
    length differs from num_inference_steps;
  - no `encode_prompt` exposed: cz_pipeline's embedding cache would otherwise call the
    pipeline's (which needs the image grid) and hand back embeddings __call__ cannot
    take;
  - negative_prompt / prompt_embeds accepted and dropped, with one log line.

Depends on torch / diffusers / transformers only (never on cz_pipeline or gradio).
"""

import logging
import os
import types

import torch

DEFAULT_REPO = "ideogram-ai/ideogram-4-nf4-diffusers"
IDEOGRAM_REPOS = (DEFAULT_REPO, "ideogram-ai/ideogram-4-nf4")
FAMILY = "ideogram"
MODEL_LABEL = "Ideogram 4.0"

# The only variant published so far. Kept as a dict so cz_pipeline's "steps advised"
# lines and the profiles read one place.
DEFAULTS = {"ideogram": {"steps": 48, "guidance": 7.0, "size": 2048}}

# Precisions of the official repos. nf4 is the only public build (2026-10-31); an fp8
# build exists for ComfyUI only and needs 32 GB and more.
PRECISIONS = ("nf4",)

# Transformer width (18 heads x 256). Used only to NAME things: the family check reads
# tensor names, never a width.
HIDDEN_SIZE = 4608

# Official guidance schedule for 48 steps: 7.0 x 45 then 3.0 x 3.
SCHEDULE_PEAK = 7.0
SCHEDULE_TAIL = 3.0
SCHEDULE_TAIL_FRACTION = 3 / 48

MIN_SIDE, MAX_SIDE, GRID = 256, 2048, 16
MAX_RATIO = 6.0

_log_handler = logging.getLogger("crispz.ideogram")

# --- Attention: no padding, no mask when there is nothing to mask ----------------------
# Measured on the 5090 (2026-10-31, nf4, offload none): 1.7 s/step at 1024 and a VRAM
# saturation at the first 2048 step. Two causes, both in how the raw pipeline is called:
#   - max_sequence_length defaults to 2048: the conditional sequence is 2048 text slots
#     (left padding for a ~100-token prompt) + the image tokens, i.e. 6144 tokens at
#     1024 px and 18432 at 2048 px, and the text encoder also reads 2048 tokens;
#   - a boolean segment mask is passed to every attention call, even when it is all
#     True (always on the unconditional branch, and on the conditional one once the
#     padding is gone). With a mask, SDPA cannot take the flash kernel and may fall back
#     to the math one, whose attention matrix is heads x L^2: 18 x 18432^2 in bf16 is
#     12 GB per call at 2048 px, before the softmax copy.
# FAST_ATTENTION sizes the text region to the prompt's exact token count and drops a
# mask that is all True. Same result: padding tokens sit in their own segment (masked
# out), text positions start at 0 whatever the padding, the image positions are fixed.
# Checked bit-near on the tiny pipeline (tests/test_ideogram_engine.py). Off: env
# IDEOGRAM_FAST_ATTENTION=0 (or set_fast_attention(False)).
FAST_ATTENTION = os.environ.get("IDEOGRAM_FAST_ATTENTION", "1").strip() not in ("0", "false", "no")


def set_fast_attention(v):
    global FAST_ATTENTION
    FAST_ATTENTION = bool(v)


def _mask_skip_processor_class():
    from diffusers.models.transformers.transformer_ideogram4 import Ideogram4AttnProcessor

    class _AllTrueMaskSkip(Ideogram4AttnProcessor):
        """Ideogram4AttnProcessor that passes no mask when the mask is all True. One
        instance per transformer: the mask is built once per forward and shared by the
        34 blocks, so the all() check (one sync) runs once per forward, not per block."""
        _seen = None
        _all_true = False

        def __call__(self, attn, hidden_states, attention_mask, image_rotary_emb):
            if attention_mask is not None:
                if attention_mask is not self._seen:
                    self._seen = attention_mask
                    self._all_true = bool(attention_mask.all())
                if self._all_true:
                    attention_mask = None
            return super().__call__(attn, hidden_states, attention_mask, image_rotary_emb)
    return _AllTrueMaskSkip


def install_mask_skip(pipe):
    """Puts the mask-skipping processor on both transformers (idempotent). Returns the
    number of attention modules switched."""
    try:
        from diffusers.models.transformers.transformer_ideogram4 import (
            Ideogram4AttnProcessor, Ideogram4Attention)
    except Exception:
        return 0
    cls = None
    n = 0
    for name in ("transformer", "unconditional_transformer"):
        tr = getattr(pipe, name, None)
        if tr is None:
            continue
        proc = None
        for m in tr.modules():
            if not isinstance(m, Ideogram4Attention):
                continue
            cur = getattr(m, "processor", None)
            if cur is None or type(cur).__name__ == "_AllTrueMaskSkip":
                continue
            if type(cur) is not Ideogram4AttnProcessor:
                continue                     # someone else's processor: leave it alone
            if proc is None:
                cls = cls or _mask_skip_processor_class()
                proc = cls()
                proc.__dict__.update(cur.__dict__)     # keeps the attention backend
            m.processor = proc
            n += 1
    return n


def prompt_token_count(pipe, prompt):
    """The token count Ideogram4Pipeline.encode_prompt will see for `prompt` (chat
    template, no special tokens added), or None when it cannot be computed."""
    tok = getattr(pipe, "tokenizer", None)
    if tok is None or not isinstance(prompt, str):
        return None
    try:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return int(tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[-1])
    except Exception:
        return None


def _log(msg):
    """cz_core's logger without importing cz_core (keeps this module standalone)."""
    try:
        from cz_core import _log as core_log
        core_log(msg)
    except Exception:
        _log_handler.info(msg)


# --- Family detection ----------------------------------------------------------------
# Ideogram 4's transformer has heads no other family in crispz carries: llm_cond_proj /
# llm_cond_norm (the 53248-wide text projection), embed_image_indicator, adaln_proj, and
# a pipeline-level 'unconditional_transformer'. Those names identify a full transformer.
# A LoRA only touches layers.N.attention / feed_forward, and Z-Image (crispz-studio) uses
# the SAME names: there the width decides (Ideogram 4 is 4608 wide, Z-Image 3840), so
# LoRAs go through family_from_header, which reads shapes too.
_IDEOGRAM_MARKERS = ("llm_cond_proj.", "llm_cond_norm.", "embed_image_indicator.",
                     "adaln_proj.", "unconditional_transformer.")
_IDEOGRAM_LAYER_MARKERS = (".attention.to_q.", ".attention.to_k.", ".attention.to_v.",
                           ".attention.qkv.", ".attention.o.", ".attention.to_out.",
                           ".feed_forward.w1.", ".feed_forward.w2.", ".feed_forward.w3.")


def _has_marker(key, markers):
    k = "." + key
    return any("." + m in k for m in markers)


def family_from_keys(keys):
    """'ideogram', 'flux3', 'flux2' or None from tensor names alone. Only the strong
    Ideogram markers count here (see above); FLUX is cz_flux3's reading (one source)."""
    keys = [k for k in keys if isinstance(k, str) and k != "__metadata__"]
    if not keys:
        return None
    if any(_has_marker(k, _IDEOGRAM_MARKERS) for k in keys):
        return FAMILY
    try:
        import cz_flux3
        return cz_flux3.family_from_keys(keys)
    except Exception:
        return None


def family_from_header(hdr):
    """The same from a safetensors header {name: {"shape": [...]}}, plus the LoRA case:
    layers.N.attention / feed_forward matrices whose input or output width is 4608."""
    if not isinstance(hdr, dict):
        return None
    fam = family_from_keys(hdr.keys())
    if fam:
        return fam
    for k, v in hdr.items():
        if k == "__metadata__" or not isinstance(v, dict):
            continue
        if not any(m in k for m in _IDEOGRAM_LAYER_MARKERS):
            continue
        shape = v.get("shape") or []
        if len(shape) == 2 and HIDDEN_SIZE in (int(shape[0]), int(shape[1])):
            return FAMILY
    return None


def is_engine_repo(repo):
    """True for the official repos, a mirror carrying 'ideogram' in its name, or a local
    folder whose model_index declares Ideogram4Pipeline."""
    r = str(repo or "").strip()
    if not r:
        return False
    if r in IDEOGRAM_REPOS or "ideogram" in r.lower():
        return True
    idx = os.path.join(r, "model_index.json")
    if os.path.isfile(idx):
        try:
            import json
            with open(idx, encoding="utf-8") as f:
                return json.load(f).get("_class_name") == "Ideogram4Pipeline"
        except Exception:
            return False
    return False


def is_base_repo(repo):
    """No distilled / base split on Ideogram 4: one checkpoint, real (asymmetric) CFG."""
    return False


def variant_of(repo):
    return "ideogram"


def variant_from_keys(keys):
    return None


# --- Guidance --------------------------------------------------------------------------
def guidance_schedule(steps, peak=SCHEDULE_PEAK):
    """The official schedule's shape for any step count: `peak` for the first ~94 % of
    the steps, then a tail at 3/7 of the peak (3.0 when peak is the official 7.0), at
    least one step long. A peak <= 0 means "the official one"."""
    n = max(1, int(steps))
    p = float(peak) if peak and float(peak) > 0 else SCHEDULE_PEAK
    tail = max(1, round(n * SCHEDULE_TAIL_FRACTION)) if n > 1 else 0
    t = p * SCHEDULE_TAIL / SCHEDULE_PEAK
    return [p] * (n - tail) + [t] * tail


def fit_size(width, height):
    """(w, h) on the 16 grid within 256..2048 and a 6:1 ratio, the closest to the ask.
    The pipeline refuses anything else with a ValueError at call time."""
    w = int(width or DEFAULTS["ideogram"]["size"])
    h = int(height or DEFAULTS["ideogram"]["size"])
    w = min(MAX_SIDE, max(MIN_SIDE, w // GRID * GRID))
    h = min(MAX_SIDE, max(MIN_SIDE, h // GRID * GRID))
    # The short side is rounded UP to the grid: rounded down, 2048 / 6 = 341 -> 336
    # gives 6.1:1, which the pipeline refuses.
    if w / h > MAX_RATIO:
        h = min(MAX_SIDE, max(MIN_SIDE, -(-int(w / MAX_RATIO) // GRID) * GRID))
    elif h / w > MAX_RATIO:
        w = min(MAX_SIDE, max(MIN_SIDE, -(-int(h / MAX_RATIO) // GRID) * GRID))
    return w, h


# --- The adapter -------------------------------------------------------------------------
class IdeogramPipe:
    """An Ideogram4Pipeline behind crispz's calling convention. Unknown attributes are
    forwarded to the real pipeline (transformer, vae, scheduler, load_lora_weights,
    enable_model_cpu_offload, maybe_free_model_hooks...)."""

    _OWN = {"_pipe", "repo", "variant", "_said", "_cfg"}

    def __init__(self, pipe, repo=DEFAULT_REPO):
        object.__setattr__(self, "_pipe", pipe)
        object.__setattr__(self, "repo", repo)
        object.__setattr__(self, "variant", "ideogram")
        object.__setattr__(self, "_said", set())
        object.__setattr__(self, "_cfg", {"is_distilled": False})

    def __getattr__(self, name):
        if name == "encode_prompt":
            # Hidden on purpose, see the module docstring (embedding cache).
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "_pipe"), name)

    def __setattr__(self, name, value):
        if name in self._OWN:
            object.__setattr__(self, name, value)
        elif name == "_interrupt":
            self._pipe._interrupt = value
        else:
            setattr(self._pipe, name, value)

    @property
    def config(self):
        return types.SimpleNamespace(**self._cfg)

    def register_to_config(self, **kw):
        self._cfg.update(kw)

    def register_modules(self, **kw):
        self._pipe.register_modules(**kw)

    def to(self, *args, **kwargs):
        self._pipe.to(*args, **kwargs)
        return self

    def _say_once(self, key, msg):
        if key not in self._said:
            self._said.add(key)
            _log(msg)

    def __call__(self, prompt=None, image=None, mask_image=None, strength=None,
                 width=None, height=None, num_inference_steps=None, generator=None,
                 guidance_scale=None, num_images_per_prompt=1, negative_prompt=None,
                 prompt_embeds=None, negative_prompt_embeds=None,
                 callback_on_step_end=None, callback_on_step_end_tensor_inputs=None,
                 output_type="pil", prompt_upsampling=False, return_dict=True, **_ignored):
        if strength is not None or mask_image is not None:
            raise NotImplementedError(
                "Ideogram 4.0 has no img2img / inpaint: crispz-ideogram emulates them in "
                "lot 3 (not delivered yet). Until then refine, upscale-refine, inpaint, "
                "outpaint and the detailer are off on this model.")
        if image is not None:
            raise NotImplementedError(
                "Ideogram 4.0 does not edit images (text to image only). Local editing: "
                "send the image to crispz-flux3; cloud editing (Ideogram 4.5) comes in lot 4.")
        if negative_prompt or negative_prompt_embeds is not None:
            self._say_once("neg", "negative prompt ignored: Ideogram 4's unconditional "
                                  "branch takes no text at all (asymmetric CFG).")
        if prompt_embeds is not None and prompt is None:
            raise ValueError("Ideogram 4: pass the prompt as text (embeddings depend on the "
                             "image grid and are rebuilt by the pipeline).")
        steps = int(num_inference_steps or DEFAULTS["ideogram"]["steps"])
        w, h = fit_size(width, height)
        if (width and height) and (w, h) != (int(width), int(height)):
            self._say_once(("size", int(width), int(height)),
                           f"size {width}x{height} -> {w}x{h} (Ideogram 4: multiples of 16, "
                           f"{MIN_SIDE} to {MAX_SIDE} a side, ratio up to 6:1)")
        sched = guidance_schedule(steps, guidance_scale)
        kw = {"prompt": prompt if prompt is not None else "", "width": w, "height": h,
              "num_inference_steps": steps, "guidance_schedule": sched,
              "num_images_per_prompt": int(num_images_per_prompt or 1),
              "generator": generator, "output_type": output_type,
              "prompt_upsampling": bool(prompt_upsampling)}
        if FAST_ATTENTION and not prompt_upsampling:
            # (prompt_upsampling reuses max_sequence_length as its max_new_tokens.)
            n = prompt_token_count(self._pipe, kw["prompt"])
            if n and n <= 2048:
                kw["max_sequence_length"] = n
            if install_mask_skip(self._pipe):
                self._say_once("fast", "attention: text region sized to the prompt, "
                                       "all-True masks dropped (flash / efficient SDPA)")
        if callback_on_step_end is not None:
            kw["callback_on_step_end"] = callback_on_step_end
            kw["callback_on_step_end_tensor_inputs"] = ["latents"]
        out = self._pipe(**kw)
        images = out.images if hasattr(out, "images") else out[0]
        return types.SimpleNamespace(images=images) if return_dict else (images,)


# --- Loading -------------------------------------------------------------------------------
def load(repo=DEFAULT_REPO, dtype=torch.bfloat16, precision="nf4", transformer=None,
         text_encoder=None, token=None, prompt_enhancer=False):
    """A ready IdeogramPipe on the CPU (the caller places it). The nf4 repo carries its
    own quantization config; `precision` is accepted for symmetry with cz_flux3.
    `prompt_enhancer=True` also loads the head that turns the text encoder into the
    local magic prompt (lot 2)."""
    from diffusers import Ideogram4Pipeline
    kw = {"torch_dtype": dtype}
    if token:
        kw["token"] = token
    if transformer is not None:
        kw["transformer"] = transformer
    if prompt_enhancer:
        from diffusers import Ideogram4PromptEnhancerHead
        kw["prompt_enhancer_head"] = Ideogram4PromptEnhancerHead.from_pretrained(
            "diffusers/qwen3-vl-8b-instruct-lm-head", torch_dtype=dtype)
    pipe = Ideogram4Pipeline.from_pretrained(repo, **kw)
    return IdeogramPipe(pipe, repo=repo)


def load_transformer(repo, dtype, precision="nf4", single_file=None, token=None):
    """The conditional transformer alone (hot swap). Single files are not supported:
    diffusers 0.42.1 has no Ideogram 4 entry in its single-file converter."""
    if single_file:
        raise NotImplementedError(
            f"{os.path.basename(str(single_file))}: single-file Ideogram 4 transformers "
            "are not supported (diffusers has no converter for them yet). Use a diffusers "
            "folder or repo as the base model.")
    from diffusers import Ideogram4Transformer2DModel
    kw = {"subfolder": "transformer", "torch_dtype": dtype}
    if token:
        kw["token"] = token
    return Ideogram4Transformer2DModel.from_pretrained(repo, **kw)


def _bfl_quant_of(path):
    """No BFL fp8r / nvfp4 builds for Ideogram: kept for cz_pipeline's call sites."""
    return None
