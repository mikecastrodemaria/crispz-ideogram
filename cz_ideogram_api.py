"""crispz-ideogram - the Ideogram cloud functions (lot 4), OFF by default.

Ideogram 4.5 is closed: its precise edit, and the other tools below, exist only on
api.ideogram.ai (United States). Everything here SENDS YOUR IMAGES AND PROMPTS OFF THE
MACHINE, so:

  - nothing runs unless `ideogram_cloud_enabled` is true AND an API key is set
    (`ideogram_api_key` or env IDEOGRAM_API_KEY); the reason is given otherwise;
  - every call writes a `CLOUD:` line in the log saying what leaves, and where;
  - every call (quote or run) is appended to the cost journal
    (<output dir>/ideogram_cloud_journal.jsonl); the API key is never written anywhere;
  - outputs that the API can keep out of the public gallery are always sent with
    `private=true` (remove-background, replace-background, SVG); the operations that have
    no such field say so in the UI (remix keeps its source in your Ideogram account).

Endpoints (API v2, read from https://developer.ideogram.ai/openapi/ideogram-api-v2.json
on 01/11/2026):

  edit                POST /v2/image/precise-edit/ideogram-4-5     multipart, sync
  replace_background  POST /v2/image/replace-background/ideogram-4-5  multipart, async
  remix               POST /v2/image/remix/ideogram-4               multipart, sync
  remove_background   POST /v2/image/remove-background/ideogram-1   multipart, sync
  vectorize           POST /v2/svg/vectorize/ideogram-vector-1      multipart, async
  svg                 POST /v2/svg/generate/ideogram-vector-1       multipart, async
  upscale             POST /v2/image/upscale/<model>                multipart, sync
  transparent         POST /v2/image/generate/ideogram-4-transparent  JSON, sync
  describe            POST /v2/image/describe/ideogram-4            multipart, sync, no quote
  (async results)     GET  /v2/generations/{generation_id}

`dry_run=true` (every operation but describe) validates and prices a request without
running it. The PriceQuote has no published schema: it is shown and journaled as
returned, with the amount picked out when a field looks like one.
"""

import datetime
import io
import json
import os
import time
from dataclasses import dataclass, field

from PIL import Image

from cz_core import CONFIG, _log, _dbg

API_BASE = "https://api.ideogram.ai"
WHERE = "api.ideogram.ai (United States)"

# --- What each operation is ------------------------------------------------------------


@dataclass(frozen=True)
class Op:
    key: str
    label: str
    path: str
    body: str = "multipart"          # "multipart" | "json"
    quote: bool = True               # dry_run supported
    output: str = "image"            # "image" | "svg" | "caption"
    model: str = ""


OPS = {
    "edit": Op("edit", "Precise edit (Ideogram 4.5)",
               "/v2/image/precise-edit/ideogram-4-5", model="ideogram-4-5"),
    "replace_background": Op("replace_background", "Replace background (Ideogram 4.5)",
                             "/v2/image/replace-background/ideogram-4-5",
                             model="ideogram-4-5"),
    "remix": Op("remix", "Remix (Ideogram 4.0)", "/v2/image/remix/ideogram-4",
                model="ideogram-4"),
    "remove_background": Op("remove_background", "Remove background",
                            "/v2/image/remove-background/ideogram-1", model="ideogram-1"),
    "vectorize": Op("vectorize", "Vectorize (image to SVG)",
                    "/v2/svg/vectorize/ideogram-vector-1", output="svg",
                    model="ideogram-vector-1"),
    "svg": Op("svg", "Generate SVG", "/v2/svg/generate/ideogram-vector-1", output="svg",
              model="ideogram-vector-1"),
    "upscale": Op("upscale", "Upscale", "/v2/image/upscale/{model}"),
    "transparent": Op("transparent", "Transparent (Ideogram 4.0)",
                      "/v2/image/generate/ideogram-4-transparent", body="json",
                      model="ideogram-4-transparent"),
    "describe": Op("describe", "Describe (Ideogram 4.0)", "/v2/image/describe/ideogram-4",
                   quote=False, output="caption", model="ideogram-4"),
}

UPSCALE_MODELS = ("auto", "topaz-text-refine", "topaz-standard-2", "topaz-bloom-2",
                  "topaz-redefine", "topaz-wonder-3-5")
UPSCALE_FACTORS = ("x2", "x4", "x8")
EDIT_QUALITIES = ("very_low", "low", "medium", "high")
SVG_QUALITIES = ("low", "medium", "high")
SVG_MODES = ("auto", "simple", "detailed")
RENDERING_SPEEDS = ("turbo", "default", "quality")
MAGIC_PROMPT = ("auto", "on", "off")
OUTPUT_RESOLUTIONS = ("1k", "2k", "4k", "8k")
ASPECT_RATIOS = ("1x4", "1x3", "1x2", "9x16", "10x16", "2x3", "3x4", "4x5", "1x1", "5x4",
                 "4x3", "3x2", "16x10", "16x9", "2x1", "3x1", "4x1")
REMIX_RESOLUTIONS = (
    "2048x2048", "1440x2880", "2880x1440", "1664x2496", "2496x1664", "1792x2240",
    "2240x1792", "1440x2560", "2560x1440", "1600x2560", "2560x1600", "1728x2304",
    "2304x1728", "1296x3168", "3168x1296", "1152x2944", "2944x1152", "1248x3328",
    "3328x1248", "1280x3072", "3072x1280", "1024x3072", "3072x1024", "1024x1024",
    "896x1120", "1120x896", "864x1152", "1152x864", "832x1248", "1248x832", "800x1280",
    "1280x800", "720x1280", "1280x720", "720x1440", "1440x720", "512x1536", "1536x512")
EDIT_MAX_PIXELS = 4194304            # 2048 x 2048: larger images come back scaled down
EDIT_MAX_REFS = 4                    # 3 with a mask (the mask takes a reference slot)
UPSCALE_MAX_SIDE = 8192


class CloudError(RuntimeError):
    """A cloud call that cannot run or failed, with a message to show as is."""


# --- The switches (shared with the magic prompt of lot 2) -------------------------------

def status():
    """(usable, reason). Both switches are needed: `ideogram_cloud_enabled` and a key."""
    import cz_magic
    return cz_magic.cloud_status()


def _require():
    ok, why = status()
    if not ok:
        raise CloudError(why)


def _api_key():
    import cz_magic
    return cz_magic.api_key()


def timeout_s():
    """How long a call (and its polling) may take, in seconds."""
    try:
        return max(30, int(CONFIG.get("ideogram_cloud_timeout", 300) or 300))
    except (TypeError, ValueError):
        return 300


# --- A request, built before anything is sent -------------------------------------------


@dataclass
class Request:
    op: str
    path: str
    fields: dict = field(default_factory=dict)       # multipart text fields, or the JSON body
    files: list = field(default_factory=list)        # [(field, filename, bytes, mime)]
    sends: str = ""                                  # what leaves the machine, for the log
    prompt: str = ""
    source: object = None                            # the source PIL image (metadata only)

    @property
    def url(self):
        return API_BASE + self.path


def _png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _as_image(img, what="image"):
    if img is None:
        raise CloudError(f"no {what}")
    if not isinstance(img, Image.Image):
        try:
            img = Image.open(img)
        except Exception as e:
            raise CloudError(f"{what} unreadable: {e}")
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
    return img


def _file(name, img, what="image"):
    img = _as_image(img, what)
    return (name, f"{name}.png", _png_bytes(img), "image/png"), img


def _bool(v):
    return "true" if v else "false"


def _seed_field(fields, seed):
    if seed is not None and int(seed) >= 0:
        fields["seed"] = str(int(seed) % 2147483648)


def api_mask(mask, size):
    """The app's mask (white = repaint, as in Inpaint) -> the API's (BLACK = edit, white =
    keep), binarized at 50 %, at the image's size. Raises when the mask is empty or full:
    the API wants both areas."""
    m = _as_image(mask, "mask").convert("L")
    if m.size != tuple(size):
        _log(f"cloud edit: mask resized {m.size[0]}x{m.size[1]} -> {size[0]}x{size[1]} "
             f"(the API wants the image's size)")
        m = m.resize(tuple(size), Image.NEAREST)
    m = m.point(lambda v: 0 if v >= 128 else 255)      # painted (white) -> black = edit
    lo, hi = m.getextrema()
    if lo == hi:
        raise CloudError("the mask must cover part of the image, not none or all of it "
                         "(paint the area to edit; no mask = edit the whole image)")
    return m


def build_edit(image, prompt, mask=None, refs=(), quality="medium", seed=None,
               num_images=1):
    """Precise edit, Ideogram 4.5. `prompt` = an instruction in plain words or a JSON
    caption (used as is). `mask` uses the app's convention (white = edit)."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise CloudError("the edit needs an instruction")
    f, img = _file("image", image)
    req = Request("edit", OPS["edit"].path, prompt=prompt, source=img)
    req.files.append(f)
    refs = [r for r in (refs or []) if r is not None]
    cap = EDIT_MAX_REFS - (1 if mask is not None else 0)
    if len(refs) > cap:
        _log(f"cloud edit: {len(refs)} reference images, the API takes {cap}"
             f"{' with a mask' if mask is not None else ''}: the last "
             f"{len(refs) - cap} dropped")
        refs = refs[:cap]
    for r in refs:
        req.files.append(_file("reference_images", r, "reference image")[0])
    fields = {"prompt": prompt,
              "quality": quality if quality in EDIT_QUALITIES else "medium",
              "num_images": str(max(1, min(8, int(num_images or 1))))}
    _seed_field(fields, seed)
    big = img.width * img.height > EDIT_MAX_PIXELS
    if mask is not None:
        m = api_mask(mask, img.size)
        req.files.append(("mask", "mask.png", _png_bytes(m), "image/png"))
        if big:
            fields["context_window"] = "auto"      # keeps the full resolution
    elif big:
        _log(f"cloud edit: {img.width}x{img.height} is over 2048x2048 in area and there "
             f"is no mask: Ideogram edits the whole image and returns it SCALED DOWN. "
             f"Paint a mask to keep the full resolution.")
    req.fields = fields
    req.sends = (f"image {img.width}x{img.height}, {len(refs)} reference(s), "
                 f"{'a mask, ' if mask is not None else ''}the instruction "
                 f"({len(prompt)} characters)")
    return req


def build_replace_background(image, background_reference, num_images=1):
    f, img = _file("image", image)
    req = Request("replace_background", OPS["replace_background"].path, source=img)
    req.files.append(f)
    req.files.append(_file("background_reference", background_reference,
                           "background reference")[0])
    req.fields = {"num_images": str(max(1, min(4, int(num_images or 1)))),
                  "private": _bool(True)}
    req.sends = f"image {img.width}x{img.height} and the background reference"
    return req


def build_remix(image, prompt, image_weight=None, resolution="", rendering_speed="default",
                num_images=1):
    prompt = (prompt or "").strip()
    if not prompt:
        raise CloudError("the remix needs a prompt")
    f, img = _file("image", image)
    req = Request("remix", OPS["remix"].path, prompt=prompt, source=img)
    req.files.append(f)
    fields = {"prompt": prompt,
              "rendering_speed": (rendering_speed if rendering_speed in RENDERING_SPEEDS
                                  else "default"),
              "num_images": str(max(1, min(8, int(num_images or 1))))}
    if resolution:
        if resolution not in REMIX_RESOLUTIONS:
            raise CloudError(f"remix resolution {resolution!r} is not one the API offers")
        fields["resolution"] = resolution
    if image_weight:
        w = max(1, min(100, int(image_weight)))
        if resolution:
            rw, rh = (int(v) for v in resolution.split("x"))
            if abs(rw / rh - img.width / img.height) > 0.02:
                raise CloudError("image weight needs the source's shape: leave the "
                                 "resolution empty, or the weight on Auto")
        fields["image_weight"] = str(w)
    req.fields = fields
    req.sends = (f"image {img.width}x{img.height} and the prompt ({len(prompt)} characters); "
                 f"Ideogram keeps the source in your account")
    return req


def build_remove_background(image):
    f, img = _file("image", image)
    req = Request("remove_background", OPS["remove_background"].path, source=img)
    req.files.append(f)
    req.fields = {"private": _bool(True)}
    req.sends = f"image {img.width}x{img.height}"
    return req


def build_vectorize(image, mode="auto"):
    f, img = _file("image", image)
    req = Request("vectorize", OPS["vectorize"].path, source=img)
    req.files.append(f)
    req.fields = {"mode": mode if mode in SVG_MODES else "auto", "private": _bool(True)}
    req.sends = f"image {img.width}x{img.height}"
    return req


def build_svg(prompt, images=(), aspect_ratio="auto", quality="medium", seed=None,
              num_images=1):
    prompt = (prompt or "").strip()
    if not prompt:
        raise CloudError("SVG generation needs a prompt")
    req = Request("svg", OPS["svg"].path, prompt=prompt)
    for im in [i for i in (images or []) if i is not None]:
        req.files.append(_file("images", im, "source image")[0])
    fields = {"prompt": prompt,
              "aspect_ratio": aspect_ratio if aspect_ratio in ASPECT_RATIOS else "auto",
              "quality": quality if quality in SVG_QUALITIES else "medium",
              "num_images": str(max(1, min(8, int(num_images or 1)))),
              "private": _bool(True)}
    _seed_field(fields, seed)
    req.fields = fields
    req.sends = f"the prompt ({len(prompt)} characters), {len(req.files)} image(s)"
    return req


def build_upscale(image, factor="x2", model="auto", seed=None):
    model = model if model in UPSCALE_MODELS else "auto"
    factor = factor if factor in UPSCALE_FACTORS else "x2"
    f, img = _file("image", image)
    k = int(factor[1:])
    if max(img.size) * k > UPSCALE_MAX_SIDE:
        raise CloudError(f"{img.width}x{img.height} {factor} would exceed "
                         f"{UPSCALE_MAX_SIDE} px on a side: the API refuses it")
    req = Request("upscale", OPS["upscale"].path.format(model=model), source=img)
    req.files.append(f)
    req.fields = {"upscale_factor": factor}
    _seed_field(req.fields, seed)
    req.sends = f"image {img.width}x{img.height} ({model}, {factor})"
    return req


def nearest_aspect(width, height):
    """The API's aspect ratio closest to width x height."""
    r = float(width) / float(height)
    best = min(ASPECT_RATIOS, key=lambda a: abs(
        (int(a.split("x")[0]) / int(a.split("x")[1])) - r))
    return best


def build_transparent(prompt, width=1024, height=1024, magic_prompt="auto",
                      output_resolution="1k", rendering_speed="default", seed=None,
                      num_images=1):
    """Ideogram 4.0 Transparent: a native alpha background. A JSON caption is used as is
    (Ideogram replaces its background description with a transparent one)."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise CloudError("the transparent generation needs a prompt")
    body = {"prompt": prompt,
            "magic_prompt": magic_prompt if magic_prompt in MAGIC_PROMPT else "auto",
            "aspect_ratio": nearest_aspect(width, height),
            "output_resolution": (output_resolution if output_resolution in OUTPUT_RESOLUTIONS
                                  else "1k"),
            "rendering_speed": (rendering_speed if rendering_speed in RENDERING_SPEEDS
                                else "default"),
            "num_images": max(1, min(8, int(num_images or 1)))}
    if seed is not None and int(seed) >= 0:
        body["seed"] = int(seed) % 2147483648
    req = Request("transparent", OPS["transparent"].path, fields=body, prompt=prompt)
    req.sends = f"the prompt ({len(prompt)} characters)"
    return req


def build_describe(image):
    f, img = _file("image", image)
    req = Request("describe", OPS["describe"].path, source=img)
    req.files.append(f)
    req.fields = {"include_bbox": _bool(True), "include_style_descriptions": _bool(True)}
    req.sends = f"image {img.width}x{img.height}"
    return req


BUILDERS = {"edit": build_edit, "replace_background": build_replace_background,
            "remix": build_remix, "remove_background": build_remove_background,
            "vectorize": build_vectorize, "svg": build_svg, "upscale": build_upscale,
            "transparent": build_transparent, "describe": build_describe}


# --- Transport (one function, replaced in the tests) ------------------------------------

def _http(method, url, headers=None, data=None, files=None, json_body=None, timeout=120):
    """-> (status_code, response_headers, body_bytes). The only network call of the module."""
    import requests
    r = requests.request(method, url, headers=headers or {}, data=data, files=files,
                         json=json_body, timeout=timeout)
    return r.status_code, dict(r.headers), r.content


def _download(url, timeout=120):
    """The result links are signed: fetched WITHOUT the API key."""
    code, hdrs, body = _http("GET", url, timeout=timeout)
    if code != 200:
        raise CloudError(f"result download failed: HTTP {code}")
    return body, hdrs


_REASONS = {
    "insufficient_funds": "no credit left on the Ideogram account",
    "subscription_required": "this needs an Ideogram API subscription",
    "daily_limit": "the account's daily limit is reached",
    "priority_credit_required": "this queue needs priority credits",
    "inflight_limit": "too many generations in progress on the account",
    "feature_limit": "the account's plan does not include this feature",
}


def _error(code, body):
    try:
        j = json.loads(body.decode("utf-8"))
    except Exception:
        j = {}
    msg = (j.get("error") or j.get("message") or j.get("detail")
           or body.decode("utf-8", "replace")[:200])
    if isinstance(msg, (dict, list)):
        msg = json.dumps(msg)[:200]
    reason = _REASONS.get(j.get("reject_reason"))
    head = {401: "the API key was refused", 402: "payment required",
            422: "refused (safety or unprocessable request)",
            429: "rate limited, try again in a moment"}.get(code, f"HTTP {code}")
    return CloudError(f"Ideogram API: {head}" + (f": {reason}" if reason else "")
                      + (f" ({msg})" if msg else ""))


def _call(req, dry_run=False):
    _require()
    op = OPS[req.op]
    if dry_run and not op.quote:
        raise CloudError(f"{op.label} has no price estimate")
    url = req.url + ("?dry_run=true" if dry_run else "")
    if dry_run:
        _log(f"CLOUD: {op.label} PRICE ESTIMATE -> {WHERE} (dry run, nothing generated "
             f"or billed); sends {req.sends}")
    else:
        _log(f"CLOUD: {op.label} -> {WHERE}; sends {req.sends}")
    headers = {"Api-Key": _api_key()}
    if op.body == "json":
        code, hdrs, body = _http("POST", url, headers=headers, json_body=req.fields,
                                 timeout=timeout_s())
    else:
        files = [(f, (fn, b, mime)) for f, fn, b, mime in req.files]
        code, hdrs, body = _http("POST", url, headers=headers, data=req.fields, files=files,
                                 timeout=timeout_s())
    if code != 200:
        raise _error(code, body)
    try:
        return json.loads(body.decode("utf-8"))
    except Exception:
        raise CloudError(f"Ideogram API: unreadable answer {body[:160]!r}")


# --- Price estimates -------------------------------------------------------------------

_AMOUNT_HINTS = ("price", "cost", "amount", "total", "usd", "credit")


def _amounts(obj, prefix=""):
    """Every numeric field whose name looks like an amount: [(dotted_key, value)]."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and any(
                    h in str(k).lower() for h in _AMOUNT_HINTS):
                out.append((key, v))
            elif isinstance(v, str) and any(h in str(k).lower() for h in _AMOUNT_HINTS):
                try:
                    out.append((key, float(v)))
                except ValueError:
                    pass
            out.extend(_amounts(v, key) if isinstance(v, (dict, list)) else [])
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(_amounts(v, f"{prefix}[{i}]"))
    return out


def _find_key(obj, names):
    """The first string value under one of `names`, at any depth."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in names and isinstance(v, str):
                return v
        for v in obj.values():
            r = _find_key(v, names)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_key(v, names)
            if r:
                return r
    return None


def format_quote(q):
    """A PriceQuote -> one line for the UI. The schema is not published: the amount-like
    fields are named, the rest is shown raw (trimmed)."""
    if not isinstance(q, dict):
        return f"quote: {q!r}"[:300]
    cur = _find_key(q, ("currency", "currency_code")) or ""
    am = _amounts(q)
    if am:
        parts = []
        for k, v in am[:4]:
            if k.endswith("micros"):
                parts.append(f"{k} = {v / 1e6:.4f} USD")
            else:
                parts.append(f"{k} = {v}{(' ' + cur) if cur else ''}")
        return "Estimate: " + "; ".join(parts)
    return "Estimate (raw): " + json.dumps(q, ensure_ascii=False)[:300]


def quote(req, output_dir=None):
    """dry_run: validated and priced, nothing generated, stored or billed. -> (quote, line)"""
    q = _call(req, dry_run=True)
    line = format_quote(q)
    _log(f"CLOUD: {OPS[req.op].label}: {line}")
    journal({"op": req.op, "endpoint": req.path, "dry_run": True, "quote": q}, output_dir)
    return q, line


# --- Running, polling, results -------------------------------------------------------------


@dataclass
class Result:
    op: str
    images: list = field(default_factory=list)       # PIL images
    svgs: list = field(default_factory=list)         # SVG texts
    caption: str = ""                                # describe
    items: list = field(default_factory=list)        # the API's data items (urls dropped)
    generation_id: str = ""
    seed: object = None
    cost_usd: object = None                          # usage_cost_usd_micros / 1e6, if given
    blocked: int = 0                                 # items refused by Ideogram's safety


def _poll(generation_id, sleep=time.sleep, now=time.monotonic):
    """GET /v2/generations/{id} until completed or failed (or the timeout)."""
    url = f"{API_BASE}/v2/generations/{generation_id}"
    deadline = now() + timeout_s()
    wait = 2.0
    while True:
        code, hdrs, body = _http("GET", url, headers={"Api-Key": _api_key()}, timeout=60)
        if code == 429:
            try:
                wait = float(hdrs.get("Retry-After") or hdrs.get("retry-after") or 5)
            except ValueError:
                wait = 5.0
        elif code != 200:
            raise _error(code, body)
        else:
            j = json.loads(body.decode("utf-8"))
            st = j.get("status")
            if st == "completed":
                return j
            if st == "failed":
                raise CloudError(f"Ideogram API: the generation failed "
                                 f"({j.get('failure_reason') or 'no reason given'})")
            wait = min(wait * 1.5, 10.0)
        if now() + wait > deadline:
            raise CloudError(f"Ideogram API: no result after {timeout_s()} s "
                             f"(generation {generation_id}; the result may still arrive in "
                             f"your Ideogram account)")
        sleep(wait)


def run(req, sleep=time.sleep, output_dir=None):
    """Sends the request, waits for the result (polling when the API answers with a
    generation id only), downloads it. -> Result."""
    op = OPS[req.op]
    j = _call(req)
    res = Result(req.op, generation_id=j.get("generation_id") or j.get("description_id")
                 or "", seed=j.get("seed"))
    if op.output == "caption":
        jp = j.get("json_prompt")
        if not jp:
            raise CloudError(f"Ideogram API returned no json_prompt: {str(j)[:200]}")
        res.caption = json.dumps(jp, ensure_ascii=False)
        journal({"op": req.op, "endpoint": req.path, "generation_id": res.generation_id},
                output_dir)
        return res
    data = j.get("data")
    if data is None:
        if not res.generation_id:
            raise CloudError(f"Ideogram API: neither results nor a generation id: "
                             f"{str(j)[:200]}")
        _log(f"CLOUD: {op.label}: accepted, waiting for generation {res.generation_id}")
        g = _poll(res.generation_id, sleep=sleep)
        data = g.get("data") or []
        micros = g.get("usage_cost_usd_micros")
        if micros is not None:
            res.cost_usd = micros / 1e6
    for it in data:
        url = it.get("url")
        meta = {k: v for k, v in it.items() if k != "url"}
        if not url:
            res.blocked += 1
            meta["blocked"] = True
            res.items.append(meta)
            continue
        body, hdrs = _download(url)
        kind = (it.get("mime_type") or hdrs.get("Content-Type")
                or hdrs.get("content-type") or "")
        if it.get("object_type") == "svg.generation" or "svg" in str(kind) or op.output == "svg":
            res.svgs.append(body.decode("utf-8", "replace"))
        else:
            res.images.append(Image.open(io.BytesIO(body)).copy())
        res.items.append(meta)
    if res.blocked:
        _log(f"CLOUD: {op.label}: {res.blocked} result(s) refused by Ideogram's safety "
             f"checks (no image returned for them)")
    _log(f"CLOUD: {op.label} done: {len(res.images)} image(s), {len(res.svgs)} SVG(s)"
         + (f", billed {res.cost_usd:.4f} USD" if res.cost_usd is not None else ""))
    journal({"op": req.op, "endpoint": req.path, "generation_id": res.generation_id,
             "images": len(res.images), "svgs": len(res.svgs), "blocked": res.blocked,
             "cost_usd": res.cost_usd}, output_dir)
    return res


# --- Cost journal and saving ----------------------------------------------------------------

def _output_root(output_dir=None):
    from cz_core import DEFAULT_OUTPUT_DIR, HERE
    d = output_dir or DEFAULT_OUTPUT_DIR
    return d if os.path.isabs(d) else os.path.join(HERE, d)


JOURNAL_NAME = "ideogram_cloud_journal.jsonl"


def journal_path(output_dir=None):
    return os.path.join(_output_root(output_dir), JOURNAL_NAME)


def journal(entry, output_dir=None):
    """One JSON line per call. Never the key, never the images."""
    entry = {"time": datetime.datetime.now().isoformat(timespec="seconds"), **entry}
    try:
        p = journal_path(output_dir)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        _log(f"CLOUD: cost journal not written ({e})")


def metadata(req, res, index=0):
    op = OPS[req.op]
    it = res.items[index] if index < len(res.items) else {}
    m = {"cloud": "ideogram", "where": WHERE, "op": req.op, "operation": op.label,
         "endpoint": req.path, "model": op.model or req.path.rsplit("/", 1)[-1],
         "generation_id": res.generation_id, "seed": it.get("seed", res.seed),
         "resolution": it.get("resolution"),
         "prompt": it.get("prompt") or req.prompt or None,
         "settings": {k: v for k, v in (req.fields or {}).items() if k != "prompt"}}
    if res.cost_usd is not None:
        m["cost_usd"] = res.cost_usd
    if req.source is not None:
        m["source_size"] = list(req.source.size)
    return {k: v for k, v in m.items() if v not in (None, "", {})}


def save(req, res, save_mode="local", output_dir=None, output_format="png"):
    """Saves every result with its metadata (PNG chunk + .json sidecar, like the local
    images). Transparent outputs and cut-outs stay PNG whatever the format setting (alpha).
    SVGs are written as .svg with a .json sidecar. -> list of paths."""
    from cz_imageio import build_output_path, save_image
    if save_mode == "display":
        return []
    paths = []
    keep_alpha = req.op in ("transparent", "remove_background")
    for i, img in enumerate(res.images):
        fmt = "png" if (keep_alpha or img.mode == "RGBA") else output_format
        meta = metadata(req, res, i)
        dst = build_output_path(None, save_mode, output_dir, fmt, tag=f"cloud_{req.op}",
                                seed=meta.get("seed"), size=img.size, index=i)
        if dst:
            save_image(img, dst, fmt, meta=meta)
            paths.append(dst)
    for i, svg in enumerate(res.svgs):
        meta = metadata(req, res, i)
        dst = build_output_path(None, save_mode, output_dir, "png", tag=f"cloud_{req.op}",
                                seed=meta.get("seed"), index=i)
        if dst:
            dst = os.path.splitext(dst)[0] + ".svg"
            with open(dst, "w", encoding="utf-8") as f:
                f.write(svg)
            with open(dst + ".json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            paths.append(dst)
    if paths:
        _dbg(f"cloud {req.op}: saved {paths}")
    return paths


def svg_preview(svg_text, size=768):
    """A PNG preview of an SVG for the gallery, when cairosvg is installed; else None
    (the SVG file is saved either way, and the UI says why there is no preview)."""
    try:
        import cairosvg
    except Exception:
        return None
    try:
        png = cairosvg.svg2png(bytestring=svg_text.encode("utf-8"), output_width=size)
        return Image.open(io.BytesIO(png)).copy()
    except Exception as e:
        _dbg(f"svg preview failed: {e}")
        return None
