"""crispz-ideogram - magic prompt: a plain idea -> an Ideogram 4 JSON caption (lot 2).

Three engines, chosen in the settings (`magic_prompt_engine`):

  local         (default) Qwen3-VL-8B, the text encoder already loaded, plus its small LM
                head (diffusers' upsample_prompt). Schema-constrained by `outlines`.
                Nothing leaves the machine.
  ollama        a local Ollama model with a system prompt: diffusers' CAPTION_SYSTEM_MESSAGE
                (default) or Ideogram's open v1 prompt (`magic_prompt_ollama_system` =
                "ideogram_v1"). The caption schema is passed as Ollama's `format`, so the
                answer is JSON of the right shape (Ollama >= 0.5). Local too.
  ideogram_api  Ideogram's hosted magic prompt (api.ideogram.ai, United States). OFF unless
                `ideogram_cloud_enabled` is true AND an API key is set; every call is logged.
                Free at the time of writing, but the prompt leaves the machine.

Every answer goes through the same cleaning: thinking blocks and code fences dropped, the
outer {...} kept, a top-level "aspect_ratio" removed (Ideogram's v1 prompt emits it), keys
put in the canonical order, then cz_caption.validate. Returns (caption_text, warnings).
"""

import json
import os
import re
import urllib.error
import urllib.request

import cz_caption
from cz_core import CONFIG, _log, _prefs

ENGINES = ("local", "ollama", "ideogram_api")
ENGINE_LABELS = {"local": "Local (Qwen3-VL-8B, on this machine)",
                 "ollama": "Ollama (local LLM)",
                 "ideogram_api": "Ideogram API (cloud, United States)"}
IDEOGRAM_MAGIC_PROMPT_URL = "https://api.ideogram.ai/v1/ideogram-v4/magic-prompt"
OLLAMA_SYSTEMS = ("diffusers", "ideogram_v1")
_HERE = os.path.dirname(os.path.abspath(__file__))
_V1_PATH = os.path.join(_HERE, "third_party", "ideogram4", "magic_prompt_v1.txt")


class MagicPromptError(RuntimeError):
    """A magic prompt that could not produce a caption, with a message to show as is."""


def _setting(key, default=None):
    """The UI preference when set, else `default`, which every caller passes as
    CONFIG.get("<key>", ...) (literal, so tools/check_config_keys.py sees the read)."""
    v = _prefs.get(key)
    return default if v is None else v


def engine():
    e = _setting("magic_prompt_engine", CONFIG.get("magic_prompt_engine", "local"))
    e = str(e or "local").strip().lower()
    return e if e in ENGINES else "local"


def cloud_enabled():
    return bool(_setting("ideogram_cloud_enabled",
                         CONFIG.get("ideogram_cloud_enabled", False)))


def api_key():
    key = os.environ.get("IDEOGRAM_API_KEY") or _setting(
        "ideogram_api_key", CONFIG.get("ideogram_api_key", ""))
    return (key or "").strip()


def cloud_status():
    """(usable, reason): the Ideogram API is usable only when both switches are on."""
    if not cloud_enabled():
        return False, ("Ideogram cloud functions are off (`ideogram_cloud_enabled`: false). "
                       "They send your prompts and images to api.ideogram.ai, in the "
                       "United States.")
    if not api_key():
        return False, "No Ideogram API key (`ideogram_api_key` or env IDEOGRAM_API_KEY)."
    return True, ("Ideogram cloud on: what a Cloud button sends goes to api.ideogram.ai "
                  "(United States).")


# --- Cleaning ------------------------------------------------------------------------
_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*|\s*```$")


def clean(raw, strip_bboxes=False):
    """Raw model text -> (caption_text, warnings). Raises MagicPromptError when no JSON
    object can be read from it (the excerpt is in the message)."""
    import prompt_improve
    text = prompt_improve.strip_thinking(raw or "").strip()
    text = _FENCE_RE.sub("", text).strip()
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        raise MagicPromptError(f"the magic prompt returned no JSON object: {text[:160]!r}")
    try:
        cap = cz_caption.parse(text[a:b + 1])
    except ValueError as e:
        raise MagicPromptError(f"the magic prompt returned invalid JSON ({e}): "
                               f"{text[a:a + 160]!r}")
    cap.pop("aspect_ratio", None)
    for el in (cap.get("compositional_deconstruction") or {}).get("elements") or []:
        if isinstance(el, dict) and "bbox_2d" in el:
            # Qwen3-VL's grounding order [x1, y1, x2, y2] -> Ideogram's [y0, x0, y1, x1].
            b = cz_caption.parse_bbox(el.pop("bbox_2d"))
            if b:
                x1, y1, x2, y2 = b
                el["bbox"] = [min(y1, y2), min(x1, x2), max(y1, y2), max(x1, x2)]
    if strip_bboxes:
        for el in (cap.get("compositional_deconstruction") or {}).get("elements") or []:
            if isinstance(el, dict):
                el.pop("bbox", None)
    cz_caption.reorder_keys(cap)
    return cz_caption.to_text(cap), cz_caption.validate(cap)


# --- Engines -------------------------------------------------------------------------
def _local(prompt, width, height, seed):
    import cz_pipeline
    return cz_pipeline.magic_prompt_local(prompt, width, height, seed=seed)


def _diffusers_messages(prompt, aspect):
    from diffusers.pipelines.ideogram4.prompt_enhancer import (CAPTION_SYSTEM_MESSAGE,
                                                                CAPTION_USER_TEMPLATE)
    return CAPTION_SYSTEM_MESSAGE, CAPTION_USER_TEMPLATE.format(aspect_ratio=aspect,
                                                                original_prompt=prompt)


def _v1_messages(prompt, aspect):
    """Ideogram's open v1 system prompt ([SYSTEM] / [USER] sections, Apache-2.0)."""
    with open(_V1_PATH, encoding="utf-8") as f:
        raw = f.read()
    sections, cur, buf = {}, None, []
    for line in raw.splitlines():
        s = line.strip()
        if s.startswith("[") and s.endswith("]") and " " not in s:
            if cur:
                sections[cur] = "\n".join(buf).strip()
            cur, buf = s[1:-1].lower(), []
        else:
            buf.append(line)
    if cur:
        sections[cur] = "\n".join(buf).strip()
    user = sections.get("user", "TARGET IMAGE ASPECT RATIO: {{aspect_ratio}} (width:height).\n"
                                "User idea: {{original_prompt}}")
    user = user.replace("{{aspect_ratio}}", aspect).replace("{{original_prompt}}", prompt)
    return sections["system"], user


# The shape Ollama is asked to produce (its structured-output `format`): the schema the
# diffusers enhancer constrains to, plus the optional style block.
OLLAMA_FORMAT = {
    "type": "object",
    "properties": {
        "high_level_description": {"type": "string"},
        "compositional_deconstruction": {
            "type": "object",
            "properties": {
                "background": {"type": "string"},
                "elements": {"type": "array", "minItems": 1, "items": {
                    "type": "object",
                    "properties": {"type": {"type": "string", "enum": ["obj", "text"]},
                                   "text": {"type": "string"},
                                   "desc": {"type": "string"}},
                    "required": ["type", "desc"]}}},
            "required": ["background", "elements"]}},
    "required": ["high_level_description", "compositional_deconstruction"]}


def _ollama(prompt, width, height, seed, model=None):
    import cz_ollama
    import prompt_improve
    aspect = cz_caption.aspect_ratio(width, height)
    which = _setting("magic_prompt_ollama_system",
                     CONFIG.get("magic_prompt_ollama_system", "diffusers"))
    which = str(which or "diffusers")
    system, user = (_v1_messages if which == "ideogram_v1" else _diffusers_messages)(prompt, aspect)
    model = model or _setting("magic_prompt_ollama_model",
                              CONFIG.get("magic_prompt_ollama_model", "")) or None
    base = cz_ollama._improve_base()
    if not model:
        models = prompt_improve.list_models(base=base)
        if not models:
            raise MagicPromptError("Ollama: no model installed (ollama pull <model>)")
        model = models[0]
    payload = {"model": model, "system": system, "prompt": user,
               **cz_ollama._ollama_gen_opts(temperature=_setting(
                   "magic_prompt_temperature", CONFIG.get("magic_prompt_temperature", 0.7)))}
    opts = payload.setdefault("options", {})
    opts["num_predict"] = max(int(opts.get("num_predict") or 0), 2048)
    if seed is not None and int(seed) >= 0:
        opts["seed"] = int(seed)
    if which != "ideogram_v1":
        payload["format"] = OLLAMA_FORMAT     # v1 adds keys of its own: left free
    try:
        out = prompt_improve.http("/api/generate", payload, base=base, timeout=300)
    except Exception as e:
        raise MagicPromptError(f"Ollama ({model}): {e}")
    _log(f"magic prompt (ollama, {model}, system {which}) done")
    return out.get("response") or ""


def _ideogram_api(prompt, width, height, seed):
    ok, why = cloud_status()
    if not ok:
        raise MagicPromptError(why)
    aspect = cz_caption.aspect_ratio(width, height).replace(":", "x")
    _log(f"CLOUD: magic prompt sent to api.ideogram.ai (United States), "
         f"{len(prompt)} characters, aspect {aspect}")
    body = json.dumps({"text_prompt": prompt, "aspect_ratio": aspect}).encode("utf-8")
    req = urllib.request.Request(IDEOGRAM_MAGIC_PROMPT_URL, data=body, method="POST",
                                 headers={"Api-Key": api_key(),
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:200]
        raise MagicPromptError(f"Ideogram API: HTTP {e.code} {detail}")
    except Exception as e:
        raise MagicPromptError(f"Ideogram API unreachable: {e}")
    jp = data.get("json_prompt")
    if not jp:
        raise MagicPromptError(f"Ideogram API returned no json_prompt: {str(data)[:200]}")
    return json.dumps(jp, ensure_ascii=False)


def expand(prompt, width, height, seed=None, which=None, model=None):
    """A plain idea -> (caption_text, warnings, engine_used). Raises MagicPromptError."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise MagicPromptError("nothing to expand: the prompt is empty")
    if cz_caption.is_caption(prompt):
        raise MagicPromptError("the prompt is already a JSON caption")
    which = which or engine()
    if which == "ollama":
        raw = _ollama(prompt, width, height, seed, model=model)
    elif which == "ideogram_api":
        raw = _ideogram_api(prompt, width, height, seed)
    else:
        which = "local"
        raw = _local(prompt, width, height, seed)
    # Ideogram's own magic prompts drop the LLM's boxes by default (strip_bboxes=True):
    # we keep the API's (it is their model) and drop none of the local ones (the local
    # schemas carry no box anyway).
    text, warnings = clean(raw)
    if warnings:
        _log(f"magic prompt ({which}): {len(warnings)} schema warning(s): "
             + "; ".join(warnings[:3]))
    return text, warnings, which


def describe(image, which="local"):
    """An image -> (caption_text, warnings). `which` = "local" (the loaded Qwen3-VL-8B,
    nothing leaves the machine) or "ideogram_api" (Ideogram's Describe 4.0, CLOUD: the
    image goes to the United States; same switches as the other cloud functions)."""
    if image is None:
        raise MagicPromptError("no image to describe")
    if which == "ideogram_api":
        import cz_ideogram_api as api
        try:
            raw = api.run(api.build_describe(image)).caption
        except api.CloudError as e:
            raise MagicPromptError(str(e))
    else:
        import cz_pipeline
        raw = cz_pipeline.describe_local(image)
    text, warnings = clean(raw)
    if warnings:
        _log(f"describe: {len(warnings)} schema warning(s): " + "; ".join(warnings[:3]))
    return text, warnings

