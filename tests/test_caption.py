"""cz_caption: the Ideogram 4 JSON caption (schema of ideogram-oss/ideogram4, docs/prompting.md).

No model, no GPU. The fixtures are the examples of the reference docs and of diffusers'
prompt enhancer: they must pass the reference CaptionVerifier with no warning, and the
editor round trip (fields -> caption -> fields) must not lose anything.

Run:  .venv/Scripts/python -m pytest tests/test_caption.py -q
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_caption as C  # noqa: E402

# docs/prompting.md, the first example (photo style, palette, bboxes).
DOC_PHOTO = {
    "high_level_description": "A golden retriever riding a skateboard down a sunny sidewalk.",
    "style_description": {
        "aesthetics": "warm, playful, vibrant",
        "lighting": "bright afternoon sunlight, long soft shadows",
        "photo": "shallow depth of field, eye-level, 85mm lens",
        "medium": "photograph",
        "color_palette": ["#F5C542", "#87CEEB", "#4A4A4A", "#FFFFFF", "#2E8B57"]},
    "compositional_deconstruction": {
        "background": "A sun-drenched suburban sidewalk lined with green hedges.",
        "elements": [
            {"type": "obj", "bbox": [200, 300, 800, 900], "desc": "A golden retriever."},
            {"type": "obj", "bbox": [250, 750, 750, 950], "desc": "A worn red skateboard."}]}}

# docs/prompting.md, the corporate card (art_style, text elements, no bbox).
DOC_DESIGN = {
    "high_level_description": "A clean, modern business card layout for a tech company.",
    "style_description": {
        "aesthetics": "minimal, professional, geometric",
        "lighting": "even, diffuse studio lighting",
        "medium": "graphic_design",
        "art_style": "flat vector design, generous whitespace, sans-serif typography",
        "color_palette": ["#FFFFFF", "#F0F0F0", "#333333", "#0066FF", "#00CC88"]},
    "compositional_deconstruction": {
        "background": "A solid off-white card surface with subtle paper texture.",
        "elements": [
            {"type": "text", "text": "ACME TECH", "desc": "Bold dark grey company name."},
            {"type": "text", "text": "hello@acme.tech", "desc": "Small blue contact email."}]}}

# diffusers' prompt enhancer example (no style_description, no bbox, \n in a text).
DIFFUSERS_ENHANCER = ('{"high_level_description":"A minimalist jazz festival poster.",'
                      '"compositional_deconstruction":{"background":"Solid deep teal.",'
                      '"elements":[{"type":"obj","desc":"A flat geometric saxophone."},'
                      '{"type":"text","text":"JAZZ\\nFESTIVAL","desc":"Large cream headline."}]}}')


def test_reference_examples_are_valid():
    assert C.validate(DOC_PHOTO) == []
    assert C.validate(DOC_DESIGN) == []
    assert C.validate(DIFFUSERS_ENHANCER) == []
    assert C.is_caption(DIFFUSERS_ENHANCER)


def test_editor_round_trip_rebuilds_the_same_caption():
    for cap in (DOC_PHOTO, DOC_DESIGN, json.loads(DIFFUSERS_ENHANCER)):
        f = C.fields_of(cap)
        again = C.build(f["high_level_description"], f["style"], f["background"],
                        f["elements"])
        assert again == cap, (again, cap)
        assert list(again) == list(cap)          # top-level key order too


def test_build_orders_keys_and_drops_empty_optionals():
    cap = C.build(
        "A poster.",
        {"kind": "art", "aesthetics": "bold", "lighting": "flat", "detail": "flat vector",
         "color_palette": "#fff, 1b1b2f"},
        "Solid red.",
        [{"type": "text", "text": "GRAND OPENING", "desc": "Huge headline.",
          "bbox": "100, 80, 300, 920", "color_palette": ["#ffffff"]},
         {"type": "obj", "desc": "", "text": ""},                     # an empty row
         {"type": "obj", "desc": "A croissant basket."}])
    sd = cap["style_description"]
    assert list(sd) == ["aesthetics", "lighting", "medium", "art_style", "color_palette"]
    assert sd["medium"] == "graphic_design" and sd["color_palette"] == ["#FFFFFF", "#1B1B2F"]
    els = cap["compositional_deconstruction"]["elements"]
    assert len(els) == 2
    assert list(els[0]) == ["type", "bbox", "text", "desc", "color_palette"]
    assert els[0]["bbox"] == [100, 80, 300, 920]
    assert list(els[1]) == ["type", "desc"]
    assert C.validate(cap) == []
    assert C.build("x")["compositional_deconstruction"] == {"background": "", "elements": []}
    assert "style_description" not in C.build("x", {"kind": "photo"})


def test_photo_style_order():
    cap = C.build("A cat.", {"kind": "photo", "aesthetics": "soft", "lighting": "window",
                             "detail": "85mm"}, "A room.", [{"desc": "A cat."}])
    assert list(cap["style_description"]) == ["aesthetics", "lighting", "photo", "medium"]
    assert cap["style_description"]["medium"] == "photograph"
    assert C.validate(cap) == []


def test_serialization_is_minified_utf8():
    cap = C.build("Café « crème »", None, "Fond", [{"type": "text", "text": "日本", "desc": "x"}])
    t = C.to_text(cap)
    assert ", " not in t and ": " not in t
    assert "Café" in t and "日本" in t and "\\u" not in t
    escaped = json.dumps(cap)                                  # ensure_ascii=True
    assert any("ensure_ascii" in w for w in C.validate(escaped))


def test_colours():
    assert C.normalize_hex("#abc") == "#AABBCC"
    assert C.normalize_hex("aabbcc") == "#AABBCC"
    assert C.normalize_hex("#1b1B2f") == "#1B1B2F"
    assert C.normalize_hex("red") is None and C.normalize_hex("#12345") is None
    pal, bad = C.parse_palette("#fff red #FFF 000000", 16)
    assert pal == ["#FFFFFF", "#000000"] and bad == ["red"]
    pal, bad = C.parse_palette(["#000"] + [f"#{i:06X}" for i in range(1, 7)], 5)
    assert len(pal) == 5 and len(bad) == 2


def test_boxes():
    # A box drawn on a 2048 x 1024 canvas, corners given in any order.
    b = C.bbox_from_pixels(1536, 768, 512, 256, 2048, 1024)
    assert b == [250, 250, 750, 750]
    assert C.bbox_to_pixels(b, 2048, 1024) == (512, 256, 1536, 768)
    assert C.bbox_from_pixels(-10, -10, 5000, 5000, 100, 100) == [0, 0, 1000, 1000]
    assert C.parse_bbox("[10, 20, 30, 40]") == [10, 20, 30, 40]
    assert C.parse_bbox("10 20 30") is None and C.parse_bbox("") is None
    assert C.aspect_ratio(1920, 320) == "6:1"


def test_validate_catches_what_matters():
    bad = {"high_level_description": "x",
           "style_description": {"aesthetics": "a", "lighting": "b", "photo": "c",
                                 "art_style": "d", "medium": "e"},
           "compositional_deconstruction": {
               "elements": [], "background": ""}}
    w = C.validate(bad)
    assert any("both 'photo' and 'art_style'" in x for x in w)
    assert any("key order" in x for x in w)
    w = C.validate({"compositional_deconstruction": {"background": "b", "elements": [
        {"desc": "no type"},
        {"type": "text", "desc": "d", "text": ""},
        {"type": "obj", "bbox": [500, 0, 400, 10], "desc": "inverted"},
        {"type": "obj", "bbox": [10, 10, 10, 50], "desc": "flat"},
        {"type": "obj", "desc": "x", "color_palette": ["#fff"]},
        {"type": "obj", "desc": "y", "extra": 1}]}})
    joined = "\n".join(w)
    for needle in ("'type' must exist", "elements[1].text: empty", "ymin (500) > ymax (400)",
                   "zero area", "not a valid #RRGGBB", "unknown keys ['extra']"):
        assert needle in joined, (needle, w)
    assert C.validate("{not json") and "invalid JSON" in C.validate("{not json")[0]
    assert any("'compositional_deconstruction' must exist" in x
               for x in C.validate({"high_level_description": "only"}))


def test_reorder_keys_fixes_an_api_answer():
    scrambled = {"compositional_deconstruction": {"elements": [
                    {"desc": "d", "text": "T", "type": "text", "bbox": [0, 0, 10, 10]}],
                 "background": "b"},
                 "style_description": {"medium": "photograph", "photo": "p",
                                       "lighting": "l", "aesthetics": "a"},
                 "high_level_description": "h"}
    assert C.validate(scrambled)
    C.reorder_keys(scrambled)
    assert C.validate(scrambled) == []
    assert list(scrambled) == ["high_level_description", "style_description",
                               "compositional_deconstruction"]


def test_is_caption():
    assert not C.is_caption("a poster that says GRAND OPENING")
    assert not C.is_caption("{red|blue} car")
    assert not C.is_caption('{"a": 1}')
    assert C.is_caption('  {"high_level_description": "x"}  ')


def test_prompt_pipeline_leaves_a_caption_alone():
    """Styles and {a|b} variants rewrite plain prompts; on a JSON caption they would
    produce invalid JSON, so the caption goes through untouched (and the log says so)."""
    import cz_prompt
    cap = C.to_text(C.build("A sign that says {OPEN|CLOSED}.", None, "Wall",
                            [{"type": "text", "text": "OPEN|CLOSED", "desc": "sign"}]))
    p, _ = cz_prompt.expand_prompt_pair(cap, "", 123)
    assert p == cap
    style = next(iter(cz_prompt.STYLES))
    p2, _ = cz_prompt._apply_styles(cap, "", [style])
    assert p2 == cap and C.validate(p2) == []
    # a plain prompt still gets both
    p3, _ = cz_prompt.expand_prompt_pair("a {red|red} car", "", 1)
    assert p3 == "a red car"


def test_design_presets_are_valid_and_native_sizes():
    """Every preset: a schema-valid caption, and a size Ideogram 4 takes as is (the
    adapter would otherwise resize it and say so)."""
    import cz_design
    import cz_ideogram
    assert len(cz_design.PRESETS) == 9
    for name in cz_design.PRESETS:
        cap = cz_design.caption_of(name)
        assert C.validate(cap) == [], (name, C.validate(cap))
        w, h = cz_design.size_of(name)
        assert cz_ideogram.fit_size(w, h) == (w, h), (name, w, h)
        assert any(e["type"] == "text" for e in cap["compositional_deconstruction"]["elements"])


def test_editor_form_round_trip_and_boxes():
    """cz_caption_ui's pure half: form <-> caption, rows <-> elements, the box bridge."""
    import cz_caption_ui as U
    import cz_design
    cap = cz_design.caption_of("Banner (6:1)")
    form = U.caption_to_form(cap)
    assert U.form_to_caption(*form) == cap
    rows = form[-1]
    labels = U.element_choices(rows)
    assert labels[0].startswith("1 · text · HEADLINE") and labels[0].endswith("[box]")
    rows2 = U.apply_bbox(rows, labels[1], [10, 20, 30, 40])
    assert rows2[1][3] == "10, 20, 30, 40" and rows2[0] == rows[0]
    assert U.apply_bbox(rows, labels[1], None)[1][3] == ""
    assert U.apply_bbox(rows, "nonsense", [1, 2, 3, 4]) == U._rows(rows)
    assert U.parse_bridge("[100, 50, 300, 950]|1730000000") == [100, 50, 300, 950]
    assert U.parse_bridge("") is None and U.parse_bridge("junk|1") is None
    # a Dataframe hands rows back with None cells and short rows
    assert U.rows_to_elements([["text", "HI", None, None]])[0]["text"] == "HI"
    html = U.render_board(cap, 1920, 320, labels[0])
    assert 'id="cz_board"' in html and "<rect" in html and "6:1" in html
    assert 'stroke-width="3"' in html                     # the selected element
    assert "#cz_bbox_bridge" in U.HEAD_JS
    w = U.warnings_md(["a", "b"])
    assert "2 warning(s)" in w and U.warnings_md([]).startswith("Schema: valid")
