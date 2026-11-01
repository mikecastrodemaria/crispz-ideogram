"""crispz-ideogram - structured JSON captions for Ideogram 4 (lot 2).

Ideogram 4 was trained on JSON captions only. The schema, from the reference repo
(ideogram-oss/ideogram4, docs/prompting.md and src/ideogram4/caption_verifier.py):

    {"high_level_description": "...",                         optional, recommended
     "style_description": {                                    optional
         "aesthetics": "...", "lighting": "...",
         "photo": "..." | (medium first, then) "art_style": "...",
         "medium": "...", "color_palette": ["#RRGGBB", ...]},  palette optional, <= 16
     "compositional_deconstruction": {                         required
         "background": "...",
         "elements": [
             {"type": "obj",  "bbox": [y0, x0, y1, x1], "desc": "...", "color_palette": [...]},
             {"type": "text", "bbox": [...], "text": "VERBATIM", "desc": "...", "color_palette": [...]}]}}

Key ORDER matters (the model saw one order in training):
  style, photo:     aesthetics, lighting, photo, medium, color_palette
  style, non-photo: aesthetics, lighting, medium, art_style, color_palette
  obj element:      type, bbox, desc, color_palette
  text element:     type, bbox, text, desc, color_palette
bbox = [y_min, x_min, y_max, x_max], integers 0..1000, origin top-left. Element palettes
hold up to 5 colours. Hex is uppercase #RRGGBB only. Serialized minified, UTF-8 literal
(separators=(",", ":"), ensure_ascii=False).

The diffusers prompt enhancer (diffusers.pipelines.ideogram4.prompt_enhancer) uses a
SUBSET of it (no style_description, no bbox): its output is still valid here.

This module: build a caption from editor fields, read one back into fields, validate
(the reference CaptionVerifier, vendored in third_party/ideogram4, plus a few checks of
our own), reorder keys, normalize colours, convert boxes. No torch, no gradio.
"""

import json
import math
import re

from third_party.ideogram4.caption_verifier import CaptionVerifier

BBOX_SCALE = 1000
STYLE_PALETTE_MAX = CaptionVerifier.style_description_palette_max      # 16
ELEMENT_PALETTE_MAX = CaptionVerifier.element_palette_max              # 5

# The `medium` values the docs name; free text is accepted too.
MEDIUMS = ("graphic_design", "illustration", "photograph", "3d_render", "painting")

_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")
_VERIFIER = CaptionVerifier()


# --- Serialization -------------------------------------------------------------------
def to_text(caption):
    """The caption as the model wants it: minified, UTF-8 literal."""
    return json.dumps(caption, separators=(",", ":"), ensure_ascii=False)


def to_pretty(caption):
    """Indented, for the raw-JSON editor (to_text is what goes to the model)."""
    return json.dumps(caption, indent=2, ensure_ascii=False)


def parse(text):
    """A caption dict from JSON text. Raises ValueError with a readable message."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON at line {e.lineno} column {e.colno}: {e.msg}")
    if not isinstance(data, dict):
        raise ValueError("a caption is a JSON object ({...})")
    return data


def is_caption(text):
    """True when `text` is a JSON caption (and must therefore bypass the styles, the
    {a|b} variants and anything else that rewrites a plain prompt)."""
    if not isinstance(text, str):
        return False
    t = text.strip()
    if not (t.startswith("{") and t.endswith("}")):
        return False
    try:
        data = json.loads(t)
    except ValueError:
        return False
    return isinstance(data, dict) and (
        "compositional_deconstruction" in data or "high_level_description" in data)


# --- Colours and boxes ---------------------------------------------------------------
def normalize_hex(value):
    """'#abc' / 'aabbcc' / '#AaBbCc' -> '#AABBCC'; None when it is not a colour."""
    if not isinstance(value, str):
        return None
    m = _HEX_RE.match(value.strip())
    if not m:
        return None
    h = m.group(1)
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return "#" + h.upper()


def parse_palette(value, limit):
    """A palette from a list or from free text ('#fff, 1b1b2f #E43F5A'): normalized,
    duplicates dropped, cut at `limit`. Returns (palette, rejected_tokens)."""
    if isinstance(value, str):
        tokens = [t for t in re.split(r"[\s,;]+", value) if t]
    else:
        tokens = list(value or [])
    out, bad = [], []
    for t in tokens:
        h = normalize_hex(t)
        if h is None:
            bad.append(str(t))
        elif h not in out:
            out.append(h)
    return out[:limit], bad + [f"(over {limit}) {h}" for h in out[limit:]]


def bbox_from_pixels(x0, y0, x1, y1, width, height):
    """A box drawn in pixels on a width x height canvas -> [y_min, x_min, y_max, x_max]
    in 0..1000, ordered and clamped."""
    def n(v, size):
        return max(0, min(BBOX_SCALE, int(round(float(v) * BBOX_SCALE / float(size)))))
    xa, xb = sorted((n(x0, width), n(x1, width)))
    ya, yb = sorted((n(y0, height), n(y1, height)))
    return [ya, xa, yb, xb]


def bbox_to_pixels(bbox, width, height):
    """[y_min, x_min, y_max, x_max] in 0..1000 -> (x0, y0, x1, y1) in pixels."""
    y0, x0, y1, x1 = bbox
    return (round(x0 * width / BBOX_SCALE), round(y0 * height / BBOX_SCALE),
            round(x1 * width / BBOX_SCALE), round(y1 * height / BBOX_SCALE))


def parse_bbox(value):
    """A bbox from a list or from text ('120, 80, 300, 920'); None when empty/invalid."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        parts = [p for p in re.split(r"[\s,;\[\]]+", value) if p]
    else:
        parts = list(value)
    try:
        vals = [int(round(float(p))) for p in parts]
    except (TypeError, ValueError):
        return None
    if len(vals) != 4:
        return None
    return [max(0, min(BBOX_SCALE, v)) for v in vals]


def aspect_ratio(width, height):
    """'W:H' reduced, as the magic prompts expect it."""
    g = math.gcd(int(width), int(height)) or 1
    return f"{int(width) // g}:{int(height) // g}"


# --- Key order -----------------------------------------------------------------------
def reorder_keys(caption):
    """Puts every object of a caption in the canonical key order, in place, and returns
    it. Unknown keys are kept, after the known ones (validate() still flags them).
    Same rules as reorder_caption_keys in ideogram-oss/ideogram4 (Apache-2.0)."""
    def ordered(d, order):
        known = [k for k in order if k in d]
        extra = [k for k in d if k not in order]
        return {k: d[k] for k in (*known, *extra)}

    if not isinstance(caption, dict):
        return caption
    top = ("high_level_description", "style_description", "compositional_deconstruction")
    caption_sorted = ordered(caption, top)
    caption.clear()
    caption.update(caption_sorted)
    sd = caption.get("style_description")
    if isinstance(sd, dict):
        try:
            caption["style_description"] = ordered(sd, _VERIFIER._style_description_key_order(sd))
        except ValueError:
            pass        # both or neither of photo / art_style: validate() says so
    cd = caption.get("compositional_deconstruction")
    if isinstance(cd, dict):
        cd = ordered(cd, _VERIFIER.compositional_deconstruction_key_order)
        if isinstance(cd.get("elements"), list):
            out = []
            for el in cd["elements"]:
                if isinstance(el, dict):
                    try:
                        el = ordered(el, _VERIFIER._element_key_order(el))
                    except ValueError:
                        pass
                out.append(el)
            cd["elements"] = out
        caption["compositional_deconstruction"] = cd
    return caption


# --- Build from editor fields / read back --------------------------------------------
def build(high_level_description="", style=None, background="", elements=()):
    """A caption dict from the editor's fields, keys in the canonical order, empty
    optional fields left out.

    style: {"kind": "photo"|"art", "aesthetics", "lighting", "photo" or "art_style",
            "medium", "color_palette"} -- included when any of its text fields is set.
    elements: dicts {"type": "obj"|"text", "text", "desc", "bbox", "color_palette"}."""
    cap = {}
    hld = (high_level_description or "").strip()
    if hld:
        cap["high_level_description"] = hld
    st = style or {}
    detail_key = "photo" if st.get("kind") == "photo" else "art_style"
    detail = (st.get(detail_key) or st.get("detail") or "").strip()
    fields = {k: (st.get(k) or "").strip() for k in ("aesthetics", "lighting", "medium")}
    palette, _ = parse_palette(st.get("color_palette") or [], STYLE_PALETTE_MAX)
    if any(fields.values()) or detail or palette:
        sd = {"aesthetics": fields["aesthetics"], "lighting": fields["lighting"]}
        if detail_key == "photo":
            sd["photo"] = detail
            sd["medium"] = fields["medium"] or "photograph"
        else:
            sd["medium"] = fields["medium"] or "graphic_design"
            sd["art_style"] = detail
        if palette:
            sd["color_palette"] = palette
        cap["style_description"] = sd
    els = []
    for e in elements or ():
        kind = "text" if (e.get("type") == "text") else "obj"
        text = e.get("text") or ""
        desc = (e.get("desc") or "").strip()
        bbox = parse_bbox(e.get("bbox"))
        pal, _ = parse_palette(e.get("color_palette") or [], ELEMENT_PALETTE_MAX)
        if not desc and not (kind == "text" and text):
            continue                                  # an empty editor row
        el = {"type": kind}
        if bbox:
            el["bbox"] = bbox
        if kind == "text":
            el["text"] = text
        el["desc"] = desc
        if pal:
            el["color_palette"] = pal
        els.append(el)
    cap["compositional_deconstruction"] = {"background": (background or "").strip(),
                                           "elements": els}
    return cap


def fields_of(caption):
    """The editor fields of a caption (the inverse of build, for the raw-JSON -> form
    round trip). Unknown keys are not carried over: validate() reports them first."""
    sd = caption.get("style_description") or {}
    cd = caption.get("compositional_deconstruction") or {}
    kind = "photo" if "photo" in sd else "art"
    style = {"kind": kind,
             "aesthetics": sd.get("aesthetics", ""), "lighting": sd.get("lighting", ""),
             "detail": sd.get("photo" if kind == "photo" else "art_style", ""),
             "medium": sd.get("medium", ""),
             "color_palette": list(sd.get("color_palette") or [])}
    elements = []
    for e in cd.get("elements") or []:
        if isinstance(e, dict):
            elements.append({"type": e.get("type", "obj"), "text": e.get("text", ""),
                             "desc": e.get("desc", ""), "bbox": e.get("bbox"),
                             "color_palette": list(e.get("color_palette") or [])})
    return {"high_level_description": caption.get("high_level_description", ""),
            "style": style, "background": cd.get("background", ""),
            "elements": elements}


# --- Validation ----------------------------------------------------------------------
def validate(caption_or_text):
    """Warnings for a caption (dict or JSON text); [] means it matches the schema.

    The reference CaptionVerifier (key presence, order, types, bbox range, hex format,
    palette sizes, \\uXXXX escapes) plus our own: empty descriptions, a text element
    without text, an empty element list, an inverted or zero-area box."""
    if isinstance(caption_or_text, str):
        warnings = _VERIFIER.check_ensure_ascii_false(caption_or_text)
        try:
            caption = parse(caption_or_text)
        except ValueError as e:
            return warnings + [str(e)]
    else:
        warnings, caption = [], caption_or_text
    warnings += _VERIFIER.verify(caption)
    if not isinstance(caption, dict):
        return warnings
    sd = caption.get("style_description")
    if isinstance(sd, dict):
        for k in ("aesthetics", "lighting", "medium", "photo", "art_style"):
            if k in sd and isinstance(sd[k], str) and not sd[k].strip():
                warnings.append(f"style_description.{k}: empty")
    cd = caption.get("compositional_deconstruction")
    if isinstance(cd, dict):
        if isinstance(cd.get("background"), str) and not cd["background"].strip():
            warnings.append("compositional_deconstruction.background: empty")
        els = cd.get("elements")
        if isinstance(els, list):
            if not els:
                warnings.append("compositional_deconstruction.elements: empty (the "
                                "model draws what the elements name)")
            for i, e in enumerate(els):
                if not isinstance(e, dict):
                    continue
                if not str(e.get("desc", "")).strip():
                    warnings.append(f"elements[{i}].desc: empty")
                if e.get("type") == "text" and not str(e.get("text", "")):
                    warnings.append(f"elements[{i}].text: empty (a text element renders "
                                    "these exact characters)")
                b = e.get("bbox")
                if (isinstance(b, list) and len(b) == 4
                        and all(isinstance(v, int) for v in b)
                        and (b[0] == b[2] or b[1] == b[3])):
                    warnings.append(f"elements[{i}].bbox: zero area {b}")
    return warnings
