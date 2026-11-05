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

Text to image (lot 1), and img2img / inpaint EMULATED (lot 3): `image` + `strength`
runs SDEdit on the tail of Ideogram's own schedule, `mask_image` adds latent blending
(see _prepare_emulated). `image` alone (a reference to edit) raises a named
NotImplementedError: editing goes through crispz-flux3 or the cloud (lot 4).

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

# Offload 'auto': below this much VRAM, 'none' is turned into 'model' (= the text encoder
# alone off the card, see IdeogramPipe.enable_model_cpu_offload). Measured on the 5090
# (32 GB): 'model' renders 1024 in 61 s with a 23.4 GB peak at 2048; 'none' keeps 5.5 GB
# more resident for the encoder, which brings 2048 back to ~29 GB of 31.8. The cost of
# 'model' is one encoder round trip per image, already inside the 61 s.
AUTO_NONE_MIN_VRAM_GB = 40.0

# The LM head of the local magic prompt (diffusers' Ideogram4PromptEnhancerHead), and the
# token budget of one caption: 25-55 words per element desc, a handful of elements.
PROMPT_ENHANCER_REPO = "diffusers/qwen3-vl-8b-instruct-lm-head"
MAGIC_MAX_NEW_TOKENS = 1536

# Describe (lot 2): the text encoder is the WHOLE Qwen3-VL-8B, vision tower included (699
# visual.* tensors in the nf4 file), so with the LM head it can look at an image. Only
# the image preprocessor config comes from the public Qwen repo (a few KB).
VISION_PROCESSOR_REPO = "Qwen/Qwen3-VL-8B-Instruct"
DESCRIBE_MAX_SIDE = 1024
# Boxes are asked in Qwen3-VL's own grounding format (bbox_2d = [x1, y1, x2, y2] in
# 0..1000, what it was trained to emit) and converted to Ideogram's [y0, x0, y1, x1]
# afterwards (cz_magic.clean): asking a model to emit the swapped order invites mistakes.
DESCRIBE_SYSTEM = (
    "You describe an image as a structured JSON caption that an image renderer will use "
    "to recreate it. Output ONE minified single-line JSON object and NOTHING else.\n\n"
    "SCHEMA, keys in this order:\n"
    '{"high_level_description":"...","style_description":{"aesthetics":"...",'
    '"lighting":"...","medium":"...","art_style":"...","color_palette":["#RRGGBB"]},'
    '"compositional_deconstruction":{"background":"...","elements":[ ... ]}}\n'
    "- for a photograph use \"photo\" (camera, lens, framing) instead of \"art_style\", "
    "placed before \"medium\", and medium \"photograph\";\n"
    "- object element: {\"type\":\"obj\",\"bbox_2d\":[x1,y1,x2,y2],\"desc\":\"...\"}\n"
    "- text element: {\"type\":\"text\",\"bbox_2d\":[x1,y1,x2,y2],\"text\":\"VERBATIM\","
    "\"desc\":\"...\"}\n\n"
    "RULES: bbox_2d in relative 0..1000 coordinates of the image. Transcribe EVERY visible "
    "text exactly, one text element per block, line breaks as \\n. medium is one of "
    "graphic_design, illustration, photograph, 3d_render, painting. color_palette: the 3 to "
    "8 dominant colours as uppercase #RRGGBB. Each desc is 15 to 50 words, concrete and "
    "standalone (shape, colour, material, position). The background is the surroundings "
    "only, never an element. Valid JSON: double quotes, no trailing commas.")

# Official guidance schedule for 48 steps: 7.0 x 45 then 3.0 x 3.
SCHEDULE_PEAK = 7.0
SCHEDULE_TAIL = 3.0
SCHEDULE_TAIL_FRACTION = 3 / 48

MIN_SIDE, MAX_SIDE, GRID = 256, 2048, 16

# Ideogram4Pipeline.__call__'s own schedule defaults (mu before the resolution shift,
# std of the logit-normal): the emulated img2img recomputes the same sigmas.
PIPE_MU, PIPE_STD = 0.0, 1.5
MAX_RATIO = 6.0

_log_handler = logging.getLogger("crispz.ideogram")


# --- Lot 8: architecture, weights format and inference profile, kept apart -------------
# Three different questions, three different answers:
#   - ARCHITECTURE: is this an Ideogram 4 transformer of the published size? Read from its
#     config.json and checked tensor by tensor (names, shapes, dtypes) before loading;
#   - WEIGHTS FORMAT: a diffusers `transformer/` folder in bf16 (fal's release). Quantized
#     exports (ComfyUI int8-convrot, nvfp4) need their own runtime and are refused here;
#   - INFERENCE PROFILE: steps, guidance, the unconditional branch, mu / std of the
#     logit-normal schedule. Never guessed from a file name or a precision: it comes from
#     the documented sources below, a `crispz_profile.json` next to a local folder, or an
#     explicit setting (`ideogram_profile`, env IDEOGRAM_PROFILE).
class Profile:
    """An inference recipe. `guidance` None = the official per-step schedule (the slider
    is its peak); a number = that constant scale, fixed (distilled checkpoints)."""

    def __init__(self, key, label, steps, guidance, unconditional, mu, std, note=""):
        self.key, self.label, self.steps = key, label, int(steps)
        self.guidance, self.unconditional = guidance, bool(unconditional)
        self.mu, self.std, self.note = float(mu), float(std), note

    @property
    def distilled(self):
        return not self.unconditional

    def as_meta(self):
        return {"profile": self.key, "steps_advised": self.steps,
                "guidance": "schedule" if self.guidance is None else self.guidance,
                "unconditional_branch": self.unconditional, "mu": self.mu, "std": self.std}

    def __repr__(self):
        return f"Profile({self.key})"


PROFILES = {
    # Ideogram4Pipeline's own defaults: 48 steps, 7.0 x 45 + 3.0 x 3, mu 0, std 1.5.
    "base": Profile("base", "Ideogram 4.0 (48 steps)", 48, None, True, 0.0, 1.5),
    # fal's model card (fal/ideogram-v4-instant, read 2026-11-05): 8 steps, ONE
    # conditional transformer, guidance_scale 1.0, guidance_schedule None, mu 0.0,
    # std 1.75. fal says stock diffusers is not bit-exact with their runtime (terminal
    # timestep and frequency-table corrections, not published).
    "instant": Profile("instant", "Ideogram 4 Instant by fal (8 steps)", 8, 1.0, False,
                       0.0, 1.75, note="distilled, no runtime CFG"),
}
DEFAULT_PROFILE = "base"

# The published Ideogram 4 transformer (read from the nf4 repo's transformer/config.json,
# public mirror cocktailpeanut/ideogram-4-nf4, 2026-11-05). A transformer source must
# match it field by field: any other size is not this model.
ARCH = {"_class_name": "Ideogram4Transformer2DModel", "in_channels": 128, "num_layers": 34,
        "num_attention_heads": 18, "attention_head_dim": 256, "intermediate_size": 12288,
        "adaln_dim": 512, "llm_features_dim": 53248, "mrope_section": [24, 20, 20],
        "rope_theta": 5000000, "norm_eps": 1e-05}

# Shared components (tokenizer, text encoder, VAE, scheduler) for a transformer-only
# source, at the revision fal's card pins. Neither nf4 transformer is downloaded.
COMPONENTS_REPO = DEFAULT_REPO
COMPONENTS_REVISION = "1874bc70267ba2c823a7239e1d70dd308c8d64dc"

# Documented transformer sources. `refuse` = known, not loadable here yet, with why.
TRANSFORMER_SOURCES = {
    "fal/ideogram-v4-instant": {"profile": "instant", "format": "diffusers", "dtype": "BF16",
                                "size_gb": 18.6},
    "fal/ideogram-v4-fast": {"refuse": (
        "fal/ideogram-v4-fast is trained with quantization-aware distillation for an "
        "NVFP4 runtime; fal states FP4 is required for its intended quality, and its "
        "BF16 tensors are pre-pack weights, not a BF16 release. It needs its own step "
        "(an FP4 runtime and quality tests), not a plain BF16 load.")},
}
PROFILE_SIDECAR = "crispz_profile.json"

# A diffusers checkpoint dtype tag -> fine for a bf16 load. Anything else (F8_E4M3, I8,
# U8, F4...) is a quantized export that needs its own runtime: refused before loading.
_FLOAT_DTYPES = {"BF16", "F16", "F32"}


class SourceError(ValueError):
    """A transformer source that cannot be used, with the reason in the message."""


def profile(key):
    k = str(key or "").strip().lower()
    if k not in PROFILES:
        raise SourceError(f"unknown Ideogram profile '{key}' "
                          f"(known: {', '.join(PROFILES)})")
    return PROFILES[k]


def is_transformer_source(src):
    """True for a transformer-only source: a documented repo id, or a local folder holding
    transformer/config.json and no model_index.json (a full pipeline is a base repo)."""
    s = str(src or "").strip()
    if not s:
        return False
    if s in TRANSFORMER_SOURCES:
        return True
    return (os.path.isfile(os.path.join(s, "transformer", "config.json"))
            and not os.path.isfile(os.path.join(s, "model_index.json")))


def resolve_profile(src, explicit=None):
    """The profile for a transformer source (None = the base pipeline). Order: the
    documented sources, then a crispz_profile.json in a local folder, then the explicit
    setting. An explicit setting that contradicts a documented one is refused; a source
    with no profile at all is refused (a recipe is never guessed)."""
    import json
    if not src:
        p = profile(explicit or DEFAULT_PROFILE)
        if p.distilled:
            raise SourceError(f"profile '{p.key}' is a distilled recipe: it needs the "
                              "matching distilled transformer, never the base weights "
                              "(their unconditional branch would be dropped).")
        return p
    s = str(src).strip()
    info = TRANSFORMER_SOURCES.get(s)
    if info and info.get("refuse"):
        raise SourceError(info["refuse"])
    documented = info.get("profile") if info else None
    side = os.path.join(s, PROFILE_SIDECAR)
    if documented is None and os.path.isfile(side):
        try:
            with open(side, encoding="utf-8") as f:
                documented = json.load(f).get("profile")
        except Exception as e:
            raise SourceError(f"{side}: unreadable ({e})")
    if explicit and documented and str(explicit).lower() != documented:
        raise SourceError(f"{s} is documented as the '{documented}' profile; "
                          f"'{explicit}' was asked. Remove the setting or fix it.")
    key = documented or explicit
    if not key:
        raise SourceError(
            f"{s}: no inference profile. Name it: ideogram_profile = "
            f"{' / '.join(PROFILES)} (config or env IDEOGRAM_PROFILE), or a "
            f"{PROFILE_SIDECAR} {{\"profile\": \"...\"}} in the folder.")
    p = profile(key)
    if p.unconditional:
        raise SourceError(f"{s}: profile '{p.key}' needs the unconditional transformer, "
                          "which a transformer-only source does not carry.")
    return p


def check_architecture(cfg, expected=None):
    """Raises SourceError naming every field of `cfg` (a transformer config dict) that
    differs from the published architecture."""
    exp = ARCH if expected is None else expected
    bad = []
    for k, v in exp.items():
        got = cfg.get(k)
        if isinstance(v, (list, tuple)):
            same = isinstance(got, (list, tuple)) and list(got) == list(v)
        elif isinstance(v, float):
            same = isinstance(got, (int, float)) and abs(float(got) - v) <= 1e-12
        else:
            same = got == v
        if not same:
            bad.append(f"{k}={got!r} (expected {v!r})")
    if bad:
        raise SourceError("not the Ideogram 4 transformer: " + ", ".join(bad))


def _expected_tensors(cfg):
    """{name: shape} the model built from `cfg` holds, without allocating it."""
    from diffusers import Ideogram4Transformer2DModel
    keep = {k: v for k, v in cfg.items() if not k.startswith("_")
            and k != "quantization_config"}
    with torch.device("meta"):
        model = Ideogram4Transformer2DModel.from_config(keep)
    return {k: tuple(t.shape) for k, t in model.state_dict().items()}


def _safetensors_header(path):
    import json
    import struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def check_tensors(folder, cfg, want_dtype="BF16"):
    """Strict pre-load check of a diffusers transformer folder: every tensor the model
    needs is there once, with its shape, in a float dtype; nothing unknown is present.
    Returns a summary dict; raises SourceError with the first problems listed."""
    import glob
    files = sorted(glob.glob(os.path.join(folder, "*.safetensors")))
    if not files:
        raise SourceError(f"{folder}: no .safetensors file")
    seen, dtypes, dup = {}, {}, []
    for fp in files:
        for name, meta in _safetensors_header(fp).items():
            if name == "__metadata__":
                continue
            if name in seen:
                dup.append(name)
            seen[name] = tuple(meta.get("shape") or ())
            dtypes[meta.get("dtype")] = dtypes.get(meta.get("dtype"), 0) + 1
    want = _expected_tensors(cfg)
    missing = sorted(set(want) - set(seen))
    unknown = sorted(set(seen) - set(want))
    shapes = sorted(k for k in set(want) & set(seen) if want[k] != seen[k])
    quant = sorted(d for d in dtypes if d not in _FLOAT_DTYPES)
    probs = []
    for label, items in (("missing", missing), ("unknown", unknown),
                         ("duplicated", sorted(set(dup))),
                         ("shape mismatch", [f"{k} {seen[k]} vs {want[k]}" for k in shapes])):
        if items:
            more = f" (+{len(items) - 3} more)" if len(items) > 3 else ""
            probs.append(f"{len(items)} {label}: {', '.join(items[:3])}{more}")
    if quant:
        probs.append(f"quantized tensors ({', '.join(quant)}): a quantized export needs its "
                     "own runtime, it is not loaded as bf16")
    if probs:
        raise SourceError(f"{folder}: " + "; ".join(probs))
    odd = sorted(d for d in dtypes if d != want_dtype)
    return {"tensors": len(seen), "files": len(files), "dtypes": dtypes,
            "note": (f"dtypes {odd} besides {want_dtype}: cast to bf16 at load") if odd
            else ""}


def fetch_source(src, token=None):
    """Local folder of a transformer source (its transformer/ downloaded once for a repo
    id, nothing else). Returns the path of the folder holding transformer/."""
    s = str(src)
    if os.path.isdir(s):
        return s
    from huggingface_hub import snapshot_download
    kw = {"allow_patterns": ["transformer/*", PROFILE_SIDECAR]}
    if token:
        kw["token"] = token
    return snapshot_download(s, **kw)


class ZeroUnconditionalTransformer(torch.nn.Module):
    """The unconditional branch of a distilled (single-branch) checkpoint: no parameter,
    returns zeros. With guidance 1.0 the pipeline's blend is 1.0 * cond + 0.0 * this, i.e.
    exactly the conditional velocity (fal's own wiring for diffusers). Never a second
    9B transformer in memory, and the call costs one zeros_like."""

    def __init__(self, dtype=torch.bfloat16):
        super().__init__()
        self.register_buffer("_dtype_anchor", torch.empty(0, dtype=dtype), persistent=False)

    @property
    def dtype(self):
        return self._dtype_anchor.dtype

    @property
    def device(self):
        # DiffusionPipeline.device / _execution_device read `.device` of every module.
        return self._dtype_anchor.device

    def forward(self, *, hidden_states, **kwargs):
        return (torch.zeros_like(hidden_states),)


class LoadedSource:
    """A transformer loaded from a source, with the profile it runs under."""

    def __init__(self, transformer, profile, source, info):
        self.transformer, self.profile, self.source, self.info = transformer, profile, source, info


def load_source(src, dtype=torch.bfloat16, explicit_profile=None, token=None, expected=None):
    """Loads a transformer-only source after the full check: profile first (no download
    for a refused or unprofiled source), then architecture, then tensors, then the load."""
    import json
    prof = resolve_profile(src, explicit_profile)
    root = fetch_source(src, token=token)
    tdir = os.path.join(root, "transformer")
    with open(os.path.join(tdir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    check_architecture(cfg, expected)
    if cfg.get("quantization_config"):
        raise SourceError(f"{src}: the config declares a quantization "
                          f"({cfg['quantization_config'].get('quant_method')}); this route "
                          "loads bf16 transformers only")
    info = check_tensors(tdir, cfg)
    if info["note"]:
        _log(f"{src}: {info['note']}")
    from diffusers import Ideogram4Transformer2DModel
    tr = Ideogram4Transformer2DModel.from_pretrained(tdir, torch_dtype=dtype)
    _log(f"transformer {src}: {info['tensors']} tensors checked in {info['files']} file(s), "
         f"profile '{prof.key}' ({prof.label})")
    return LoadedSource(tr, prof, str(src), info)

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
            if attention_mask is not None and FAST_ATTENTION:
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


# --- The text-conditioning norm, on the text rows only -----------------------------------
# The transformer injects the text through llm_cond_norm (RMSNorm over 53248 features:
# 13 encoder layers side by side) then llm_cond_proj, on the WHOLE sequence: ~150 text
# rows and, at 2048 px, 16384 image rows that are zeros (and are zeroed again right
# after). diffusers' RMSNorm upcasts to float32: x.float(), its square, the scaled copy,
# about 9 to 10 GB of temporaries per transformer call at 2048 px, two calls a step
# (5090, 2026-11-01: denoise peak 29.9 GB of 31.8, the VAE decode needed nothing more).
# A zero row normalizes to a zero row (0 * rsqrt(eps) * w = 0), so normalizing the
# non-zero rows only and leaving zeros elsewhere is the same result, minus the float32
# copies of the zeros. Only the norm is patched: llm_cond_proj, which LoRAs may target,
# is left alone. Follows FAST_ATTENTION (off = the stock path).
def _sparse_rms_forward(norm, orig):
    def forward(x):
        if not FAST_ATTENTION or x.dim() != 3:
            return orig(x)
        rows = (x.amax(-1) != 0) | (x.amin(-1) != 0)       # no full-size temporary
        n = int(rows.sum())
        if n == rows.numel():
            return orig(x)
        sub = orig(x[rows])                                   # (n, D)
        out = torch.zeros(x.shape, dtype=sub.dtype, device=x.device)
        out[rows] = sub
        return out
    return forward


def install_sparse_text_norm(pipe):
    """Patches llm_cond_norm of both transformers (idempotent). Returns how many."""
    n = 0
    for name in ("transformer", "unconditional_transformer"):
        norm = getattr(getattr(pipe, name, None), "llm_cond_norm", None)
        if norm is None or getattr(norm, "_cz_sparse", False):
            continue
        norm.forward = _sparse_rms_forward(norm, norm.forward)
        norm._cz_sparse = True
        n += 1
    return n


def _no_compile_without_triton():
    """outlines-core wraps its logits-mask kernel in torch.compile, which needs Triton;
    on Windows there is none, so every magic prompt printed two pages of TritonMissing
    traceback before PyTorch fell back to eager (5090, 2026-11-01: harmless, 43.7 s
    anyway). Without Triton, dynamo is switched off for the call: the kernel runs eagerly
    from the start, silently. crispz uses torch.compile nowhere else."""
    import contextlib
    import importlib.util
    if importlib.util.find_spec("triton") is not None:
        return contextlib.nullcontext()
    import torch._dynamo
    return torch._dynamo.config.patch(disable=True)


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

    _OWN = {"_pipe", "repo", "variant", "_said", "_cfg", "_all_hooks", "_tr_pre",
            "_te_on_gpu", "_offload_device", "_vl_processor", "released", "profile",
            "source"}

    def __init__(self, pipe, repo=DEFAULT_REPO, profile=None, source=None):
        object.__setattr__(self, "_pipe", pipe)
        object.__setattr__(self, "repo", repo)
        object.__setattr__(self, "profile", profile or PROFILES[DEFAULT_PROFILE])
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "variant", "ideogram")
        object.__setattr__(self, "_said", set())
        object.__setattr__(self, "_cfg", {"is_distilled": False})
        for k in ("_tr_pre", "_offload_device"):
            object.__setattr__(self, k, None)
        object.__setattr__(self, "_all_hooks", [])
        object.__setattr__(self, "_te_on_gpu", False)
        object.__setattr__(self, "released", False)

    def release(self):
        """Lets go of the real pipeline, so that a reference to THIS wrapper kept by some
        caller no longer pins 15 GB of weights on the card (cz_pipeline.free_vram calls
        it). A released wrapper refuses any use; cz_pipeline swaps it for a fresh pipe."""
        object.__setattr__(self, "_pipe", None)
        object.__setattr__(self, "released", True)
        object.__setattr__(self, "_all_hooks", [])
        object.__setattr__(self, "_vl_processor", None)

    def __getattr__(self, name):
        if name == "encode_prompt":
            # Hidden on purpose, see the module docstring (embedding cache).
            raise AttributeError(name)
        inner = object.__getattribute__(self, "_pipe")
        if inner is None:
            raise RuntimeError("this Ideogram pipeline was released (VRAM freed): "
                               "get a fresh one from cz_pipeline.get_pipe()")
        return getattr(inner, name)

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

    # --- 'model' offload = the TEXT ENCODER only -----------------------------------------
    # diffusers' enable_model_cpu_offload is a poor fit for Ideogram 4, twice over:
    #   - it chains text_encoder -> transformer -> unconditional_transformer -> vae, each
    #     model sending the previous one back to the CPU when it runs. Both transformers
    #     run at every step, so the conditional one went to the CPU and back every step:
    #     5.2 GB each way (5090, 2026-10-31: 2.8 s/step at 1024 against 1.7 in 'none');
    #   - its hook on the text encoder never fires: the pipeline calls the encoder's
    #     layers one by one (_get_text_encoder_hidden_states), never its forward(), so
    #     the 8B encoder simply ran ON THE CPU.
    # Here the two transformers and the VAE stay on the card; the encoder is moved in
    # explicitly before a call and sent back as soon as the first transformer runs (or at
    # the end of the call). Same method name, so every enable_model_cpu_offload() of
    # cz_pipeline gets it. No accelerate hook on any component, so diffusers' LoRA loader
    # does not see an offload to "restore" its own way.
    def enable_model_cpu_offload(self, gpu_id=None, device=None):
        p = self._pipe
        self.remove_all_hooks()
        dev = torch.device(device) if device is not None else torch.device(
            "cuda" if gpu_id is None else f"cuda:{int(gpu_id)}")
        for name in ("transformer", "unconditional_transformer", "vae",
                     "prompt_enhancer_head"):
            m = getattr(p, name, None)
            if isinstance(m, torch.nn.Module):
                m.to(dev)
        te = getattr(p, "text_encoder", None)
        if te is None:
            return
        te.to("cpu")
        object.__setattr__(self, "_offload_device", dev)
        object.__setattr__(self, "_all_hooks", ["text_encoder"])   # cz_pipeline counts it
        object.__setattr__(self, "_te_on_gpu", False)

        def tr_pre(module, args):
            if self._te_on_gpu:
                self._encoder_out()
        handles = []
        for name in ("transformer", "unconditional_transformer"):
            m = getattr(p, name, None)
            if m is not None:
                handles.append(m.register_forward_pre_hook(tr_pre))
        object.__setattr__(self, "_tr_pre", handles)

    def _encoder_in(self):
        if self._offload_device is not None and not self._te_on_gpu:
            self._pipe.text_encoder.to(self._offload_device)
            object.__setattr__(self, "_te_on_gpu", True)

    def _encoder_out(self):
        if self._te_on_gpu:
            self._pipe.text_encoder.to("cpu")
            object.__setattr__(self, "_te_on_gpu", False)

    def remove_all_hooks(self):
        for h in list(self._tr_pre or []):
            try:
                h.remove()
            except Exception:
                pass
        object.__setattr__(self, "_tr_pre", None)
        object.__setattr__(self, "_all_hooks", [])
        object.__setattr__(self, "_te_on_gpu", False)
        object.__setattr__(self, "_offload_device", None)
        self._pipe.remove_all_hooks()

    def maybe_free_model_hooks(self):
        """End of a call / release_vram(offload=True): the encoder back on the CPU (the
        transformers stay, that is the point)."""
        if self._offload_device is not None:
            self._encoder_out()
        else:
            self._pipe.maybe_free_model_hooks()

    def _keep_encoder_offload(self):
        """Should diffusers' own model offload come back on the raw pipeline (a LoRA
        loader, a merge from crispz-flux3), it is replaced, said once."""
        if self._offload_device is not None and getattr(self._pipe, "_all_hooks", None):
            self._say_once("reoffload", "offload: diffusers' model offload had come back; "
                                        "encoder-only offload reapplied")
            self.enable_model_cpu_offload(device=self._offload_device)

    # --- Local magic prompt (lot 2) -------------------------------------------------------
    def ensure_prompt_enhancer(self, token=None, dtype=torch.bfloat16):
        """The LM head that makes the text encoder generative (diffusers'
        Ideogram4PromptEnhancerHead), loaded once on first use: ~1.2 GB, downloaded the
        first time. The encoder body is shared, nothing else is loaded."""
        p = self._pipe
        head = getattr(p, "prompt_enhancer_head", None)
        if head is None:
            from diffusers import Ideogram4PromptEnhancerHead
            _log(f"magic prompt: loading the LM head ({PROMPT_ENHANCER_REPO}, ~1.2 GB, "
                 "downloaded the first time)")
            kw = {"torch_dtype": dtype}
            if token:
                kw["token"] = token
            head = Ideogram4PromptEnhancerHead.from_pretrained(PROMPT_ENHANCER_REPO, **kw)
            p.register_modules(prompt_enhancer_head=head)
        return head

    def magic_prompt(self, prompt, width, height, seed=None, temperature=None,
                     max_new_tokens=MAGIC_MAX_NEW_TOKENS, device=None):
        """Plain prompt -> JSON caption text, locally, with the text encoder Qwen3-VL-8B
        plus its LM head (diffusers' upsample_prompt; schema-constrained when `outlines`
        is installed, said once when it is not). Under the encoder-only offload the
        encoder and the head come onto the card for the call and leave after."""
        from diffusers.utils import is_outlines_available
        if not is_outlines_available():
            self._say_once("outlines", "magic prompt: `outlines` is not installed, the "
                                       "local magic prompt runs unconstrained and may "
                                       "return invalid JSON (pip install outlines)")
        head = self.ensure_prompt_enhancer()
        dev = self._offload_device or device or self._pipe._execution_device
        w, h = fit_size(width or DEFAULTS["ideogram"]["size"], height or DEFAULTS["ideogram"]["size"])
        gen = torch.Generator().manual_seed(int(seed)) if seed is not None and int(seed) >= 0 else None
        kw = {"height": h, "width": w, "max_new_tokens": int(max_new_tokens),
              "generator": gen, "device": dev}
        if temperature is not None:
            kw["temperature"] = float(temperature)
        offloaded = self._offload_device is not None
        self._encoder_in()
        head.to(dev)
        try:
            with torch.no_grad(), _no_compile_without_triton():
                return self._pipe.upsample_prompt(prompt, **kw)[0]
        finally:
            if offloaded:
                head.to("cpu")
                self._encoder_out()

    def _enhancer(self):
        """The generative Qwen3-VL (encoder body + LM head), built once, shared with
        diffusers' upsample_prompt (same attribute)."""
        p = self._pipe
        if getattr(p, "_prompt_enhancer", None) is None:
            from diffusers.pipelines.ideogram4.prompt_enhancer import build_prompt_enhancer
            p._prompt_enhancer = build_prompt_enhancer(p.text_encoder,
                                                      self.ensure_prompt_enhancer())
        return p._prompt_enhancer

    def _processor(self):
        proc = getattr(self, "_vl_processor", None)
        if proc is None:
            from transformers import AutoProcessor
            proc = AutoProcessor.from_pretrained(VISION_PROCESSOR_REPO)
            object.__setattr__(self, "_vl_processor", proc)
        return proc

    def describe_image(self, image, max_new_tokens=MAGIC_MAX_NEW_TOKENS, device=None):
        """An image -> a JSON caption text (boxes as bbox_2d, see DESCRIBE_SYSTEM), with the
        Qwen3-VL-8B already loaded as the text encoder, its vision tower and the LM head.
        Greedy decoding: the same image gives the same caption. Local only."""
        head = self.ensure_prompt_enhancer()
        enhancer = self._enhancer()
        proc = self._processor()
        img = image.convert("RGB")
        if max(img.size) > DESCRIBE_MAX_SIDE:
            k = DESCRIBE_MAX_SIDE / float(max(img.size))
            img = img.resize((max(28, round(img.width * k)), max(28, round(img.height * k))))
        messages = [
            {"role": "system", "content": [{"type": "text", "text": DESCRIBE_SYSTEM}]},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Describe this image as the JSON caption."}]}]
        dev = self._offload_device or device or self._pipe._execution_device
        offloaded = self._offload_device is not None
        self._encoder_in()
        head.to(dev)
        try:
            inputs = proc.apply_chat_template(messages, tokenize=True,
                                              add_generation_prompt=True, return_dict=True,
                                              return_tensors="pt").to(dev)
            with torch.no_grad(), _no_compile_without_triton():
                out = enhancer.generate(**inputs, max_new_tokens=int(max_new_tokens),
                                        do_sample=False)
            new = out[:, inputs["input_ids"].shape[1]:]
            return proc.batch_decode(new, skip_special_tokens=True)[0].strip()
        finally:
            if offloaded:
                head.to("cpu")
                self._encoder_out()

    # --- img2img / inpaint, emulated (lot 3) ----------------------------------------------
    # Ideogram4Pipeline has neither: it always starts from pure noise on its own sigma
    # schedule. SDEdit on a flow-matching model is x_t = (1 - sigma) * x0 + sigma * noise:
    # the image is encoded to Ideogram's tokens, noised to the sigma of the first kept
    # step, and only the tail of the schedule runs. The schedule is Ideogram's own
    # (logit-normal, resolution-aware mu, std 1.5; the scheduler applies no shift on top),
    # recomputed here with diffusers' functions to know that sigma before the call, then
    # cut at the same index inside it (scheduler.set_timesteps is wrapped for the call).
    # The strength is a FRACTION OF THE STEPS, diffusers' and crispz-flux3's rule (0.35 =
    # the last ~35 % of the steps); at least one step always runs. The per-step guidance is
    # rotated so the tail of the official schedule (the low-guidance last steps) still
    # lands on the last steps. Inpaint: after each step the tokens outside the mask are
    # put back to the init image noised to the sigma just reached (latent blending, the
    # crispz-flux3 technique); the mask is max-pooled to the 16-pixel token grid.
    def _encode_tokens(self, image, width, height, device):
        """Normalized Ideogram tokens (1, h/16 * w/16, 128) of `image` at width x height:
        the exact inverse of the pipeline's decode (bn, then (p, p, c) packing)."""
        import numpy as np
        from PIL import Image
        vae = self._pipe.vae
        img = image.convert("RGB")
        if img.size != (width, height):
            img = img.resize((width, height), Image.LANCZOS)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 127.5 - 1.0)
        x = x.permute(2, 0, 1)[None].to(device, vae.dtype)
        with torch.no_grad():
            r = vae.encode(x).latent_dist.mode().float()
        b, c, hh, ww = r.shape
        tok = r.view(b, c, hh // 2, 2, ww // 2, 2).permute(0, 2, 4, 3, 5, 1)
        tok = tok.reshape(b, (hh // 2) * (ww // 2), 4 * c)
        mean = vae.bn.running_mean.view(1, 1, -1).to(tok.device, torch.float32)
        std = torch.sqrt(vae.bn.running_var.float() + vae.config.batch_norm_eps)
        return (tok - mean) / std.view(1, 1, -1).to(tok.device)

    @staticmethod
    def _mask_tokens(mask_image, width, height, device):
        """(1, seq, 1): 1 = repaint, 0 = keep; a token is repainted as soon as ANY of its
        pixels is (max-pool), so a thin stroke never vanishes below the grid."""
        import numpy as np
        from PIL import Image
        m = mask_image.convert("L")
        if m.size != (width, height):
            m = m.resize((width, height), Image.NEAREST)
        t = torch.from_numpy(np.asarray(m, dtype=np.float32) / 255.0)[None, None]
        t = torch.nn.functional.max_pool2d(t, kernel_size=GRID, stride=GRID)
        return t.flatten(2).transpose(1, 2).to(device)

    @staticmethod
    def _is_full_mask(mask_image):
        try:
            return mask_image.convert("L").getextrema()[0] >= 250
        except Exception:
            return False

    def _prepare_emulated(self, kw, init, mask_image, strength, steps, w, h, sched,
                          generator, user_cb):
        """Fills kw (latents, rotated guidance) for an img2img / inpaint call and returns
        (step callback, restore) -- restore undoes the scheduler wrap after the call."""
        from diffusers.pipelines.ideogram4.pipeline_ideogram4 import (
            _logit_normal_sigmas, _resolution_aware_mu)
        p = self._pipe
        dev = self._offload_device or p._execution_device
        s = 1.0 if strength is None else float(min(max(float(strength), 0.0), 1.0))
        prof = self.profile
        full = _logit_normal_sigmas(steps, _resolution_aware_mu(h, w, prof.mu), std=prof.std)
        start = min(int(max(steps - steps * s, 0)), steps - 1)
        sigma0 = float(full[start])
        npp = int(kw.get("num_images_per_prompt") or 1)
        x0 = self._encode_tokens(init, w, h, dev).expand(npp, -1, -1).contiguous()
        g = generator[0] if isinstance(generator, (list, tuple)) else generator
        noise = torch.randn(x0.shape, generator=g, dtype=torch.float32,
                            device=g.device if g is not None else "cpu").to(x0.device)
        kw["latents"] = (1.0 - sigma0) * x0 + sigma0 * noise
        if sched is not None:
            kw["guidance_schedule"] = list(sched[start:]) + list(sched[:start])
        sch = p.scheduler
        real_set = sch.set_timesteps

        def set_timesteps(*a, **k):
            if k.get("sigmas") is not None:
                k["sigmas"] = list(k["sigmas"])[start:]
            return real_set(*a, **k)
        sch.set_timesteps = set_timesteps
        # The progress bar and the live preview count the steps that RUN, not the full
        # schedule: diffusers sizes its bar on num_inference_steps (it showed 14/48 and
        # stopped there), and the preview callback reads _cz_run_steps (lot 7).
        run = steps - start
        real_bar = p.progress_bar
        p.progress_bar = lambda iterable=None, total=None: real_bar(
            iterable=iterable, total=run if total is not None else None)
        p._cz_run_steps = run

        def restore():
            for obj, name in ((sch, "set_timesteps"), (p, "progress_bar")):
                try:
                    delattr(obj, name)
                except AttributeError:
                    pass
            p._cz_run_steps = None
        mask = None
        if mask_image is not None and not self._is_full_mask(mask_image):
            mask = self._mask_tokens(mask_image, w, h, x0.device)
        kind = "inpaint" if mask is not None else "img2img"
        _log(f"{kind} (emulated): strength {s:.2f}, {steps - start} of {steps} steps from "
             f"sigma {sigma0:.3f}")
        if mask is None:
            return user_cb, restore

        def blend(pipe, i, t, cb_kwargs):
            lat = cb_kwargs["latents"]
            sig = pipe.scheduler.sigmas
            nxt = float(sig[i + 1]) if i + 1 < len(sig) else 0.0
            known = (1.0 - nxt) * x0 + nxt * noise
            cb_kwargs["latents"] = mask * lat + (1.0 - mask) * known.to(lat.dtype)
            if user_cb is not None:
                out = user_cb(pipe, i, t, cb_kwargs)
                if isinstance(out, dict):
                    cb_kwargs = out
            return cb_kwargs
        return blend, restore

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
        emulated = strength is not None or mask_image is not None
        if emulated:
            init = image[0] if isinstance(image, (list, tuple)) and image else image
            if init is None:
                raise ValueError("img2img / inpaint need an image to start from")
        elif image is not None:
            raise NotImplementedError(
                "Ideogram 4.0 does not edit images (text to image only). Local editing: "
                "send the image to crispz-flux3; cloud editing (Ideogram 4.5) comes in lot 4.")
        if negative_prompt or negative_prompt_embeds is not None:
            self._say_once("neg", "negative prompt ignored: Ideogram 4's unconditional "
                                  "branch takes no text at all (asymmetric CFG).")
        if prompt_embeds is not None and prompt is None:
            raise ValueError("Ideogram 4: pass the prompt as text (embeddings depend on the "
                             "image grid and are rebuilt by the pipeline).")
        prof = self.profile
        steps = int(num_inference_steps or prof.steps)
        if prof.distilled and steps != prof.steps:
            self._say_once(("steps", steps),
                           f"{prof.label}: {steps} steps asked, the published recipe is "
                           f"{prof.steps}; run as asked, outside the recipe")
        if emulated:
            # Output size = the asked one, else the image's own (on Ideogram's grid).
            width = width or init.size[0]
            height = height or init.size[1]
        w, h = fit_size(width, height)
        if (width and height) and (w, h) != (int(width), int(height)):
            self._say_once(("size", int(width), int(height)),
                           f"size {width}x{height} -> {w}x{h} (Ideogram 4: multiples of 16, "
                           f"{MIN_SIDE} to {MAX_SIDE} a side, ratio up to 6:1)")
        kw = {"prompt": prompt if prompt is not None else "", "width": w, "height": h,
              "num_inference_steps": steps, "mu": prof.mu, "std": prof.std,
              "num_images_per_prompt": int(num_images_per_prompt or 1),
              "generator": generator, "output_type": output_type,
              "prompt_upsampling": bool(prompt_upsampling)}
        if prof.guidance is None:
            sched = guidance_schedule(steps, guidance_scale)
            kw["guidance_schedule"] = sched
        else:
            # A distilled checkpoint has no CFG: its scale is part of the recipe. Another
            # value is refused here, never applied (with the zero unconditional branch,
            # g * cond would just scale the velocity: a burnt, noisy image).
            if guidance_scale is not None and abs(float(guidance_scale) - prof.guidance) > 1e-9:
                raise ValueError(f"{prof.label} runs at guidance {prof.guidance:g} (no CFG, "
                                 f"distilled); {float(guidance_scale):g} was asked")
            sched = None
            kw["guidance_scale"] = prof.guidance
            kw["guidance_schedule"] = None
        if FAST_ATTENTION:
            if not prompt_upsampling:
                # (prompt_upsampling reuses max_sequence_length as its max_new_tokens.)
                n = prompt_token_count(self._pipe, kw["prompt"])
                if n and n <= 2048:
                    kw["max_sequence_length"] = n
            if install_mask_skip(self._pipe) | install_sparse_text_norm(self._pipe):
                self._say_once("fast", "attention: text region sized to the prompt, "
                                       "all-True masks dropped (flash / efficient SDPA), "
                                       "text-conditioning norm on the text rows only")
        restore = None
        if emulated:
            callback_on_step_end, restore = self._prepare_emulated(
                kw, init, mask_image, strength, steps, w, h, sched, generator,
                callback_on_step_end)
        if callback_on_step_end is not None:
            kw["callback_on_step_end"] = callback_on_step_end
            kw["callback_on_step_end_tensor_inputs"] = ["latents"]
        self._keep_encoder_offload()
        self._encoder_in()
        try:
            out = self._pipe(**kw)
        finally:
            self._encoder_out()        # stopped before the first transformer, or an error
            if restore is not None:
                restore()
        images = out.images if hasattr(out, "images") else out[0]
        return types.SimpleNamespace(images=images) if return_dict else (images,)


# --- Loading -------------------------------------------------------------------------------
def load(repo=DEFAULT_REPO, dtype=torch.bfloat16, precision="nf4", transformer=None,
         text_encoder=None, token=None, prompt_enhancer=False):
    """A ready IdeogramPipe on the CPU (the caller places it). The nf4 repo carries its
    own quantization config; `precision` is accepted for symmetry with cz_flux3.
    `prompt_enhancer=True` also loads the head that turns the text encoder into the
    local magic prompt (lot 2).

    `transformer` a LoadedSource (lot 8, a distilled single-branch checkpoint): the
    shared components come from `repo` (pinned to COMPONENTS_REVISION when it is the
    official repo), neither nf4 transformer is downloaded or loaded, and the
    unconditional branch is ZeroUnconditionalTransformer."""
    from diffusers import Ideogram4Pipeline
    kw = {"torch_dtype": dtype}
    if token:
        kw["token"] = token
    prof, source = None, None
    if isinstance(transformer, LoadedSource):
        prof, source = transformer.profile, transformer.source
        kw["transformer"] = transformer.transformer
        kw["unconditional_transformer"] = None
        if repo == COMPONENTS_REPO:
            kw["revision"] = COMPONENTS_REVISION
    elif transformer is not None:
        kw["transformer"] = transformer
    if prompt_enhancer:
        from diffusers import Ideogram4PromptEnhancerHead
        kw["prompt_enhancer_head"] = Ideogram4PromptEnhancerHead.from_pretrained(
            "diffusers/qwen3-vl-8b-instruct-lm-head", torch_dtype=dtype)
    pipe = Ideogram4Pipeline.from_pretrained(repo, **kw)
    if prof is not None:
        pipe.register_modules(unconditional_transformer=ZeroUnconditionalTransformer(dtype))
    return IdeogramPipe(pipe, repo=repo, profile=prof, source=source)


def wrap_distilled(pipe, loaded):
    """An Ideogram4Pipeline whose transformer is `loaded` (a LoadedSource) and whose
    unconditional branch is the zero module; for tests and tools building from parts."""
    pipe.register_modules(transformer=loaded.transformer,
                          unconditional_transformer=ZeroUnconditionalTransformer(
                              loaded.transformer.dtype))
    return IdeogramPipe(pipe, repo=getattr(pipe, "name_or_path", "parts"),
                        profile=loaded.profile, source=loaded.source)


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
