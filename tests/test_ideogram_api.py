"""Lot 4: the Ideogram cloud functions (cz_ideogram_api, cz_cloud_ui).

Nothing here touches the network: the module's one transport function (_http) is
replaced by a fake server that records every request. What is guarded:
  - OFF by default: with the switches off (or no key) nothing is sent, the reason is said,
    the UI buttons are greyed out;
  - what is sent: the right endpoint, the Api-Key header, the fields of the published
    API v2 (01/11/2026), the mask in the API's convention (black = edit), private=true
    where the API allows it;
  - every call logs a CLOUD: line naming the United States, and is journaled without the
    key; price estimates use dry_run=true and bill nothing;
  - results: sync and async (polling, 429 Retry-After, failure, timeout), safety-blocked
    items counted, images and SVGs saved with their metadata.
"""
import io
import json
import os
import sys

import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_ideogram_api as api  # noqa: E402
import cz_magic  # noqa: E402

KEY = "sk-test-0123456789"


def _png(img):
    b = io.BytesIO()
    img.save(b, "PNG")
    return b.getvalue()


class Server:
    """A fake api.ideogram.ai: answers by (method, path) and records the calls."""

    def __init__(self):
        self.calls = []
        self.routes = {}

    def on(self, method, path, *answers):
        self.routes.setdefault((method, path), []).extend(answers)

    def __call__(self, method, url, headers=None, data=None, files=None, json_body=None,
                 timeout=120):
        self.calls.append({"method": method, "url": url, "headers": dict(headers or {}),
                           "data": dict(data or {}), "files": list(files or []),
                           "json": json_body})
        path = url.split("api.ideogram.ai", 1)[-1].split("?")[0] if "api.ideogram.ai" in url \
            else url
        q = self.routes.get((method, path))
        if not q:
            raise AssertionError(f"unexpected call {method} {url}")
        ans = q.pop(0) if len(q) > 1 else q[0]
        code, body = ans[0], ans[1]
        hdrs = ans[2] if len(ans) > 2 else {}
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        return code, hdrs, body


@pytest.fixture
def server(monkeypatch, tmp_path):
    s = Server()
    monkeypatch.setattr(api, "_http", s)
    monkeypatch.setattr(api, "journal_path", lambda output_dir=None: str(tmp_path / "j.jsonl"))
    return s


@pytest.fixture
def logs(monkeypatch):
    lines = []
    monkeypatch.setattr(api, "_log", lambda m: lines.append(str(m)))
    import cz_cloud_ui
    monkeypatch.setattr(cz_cloud_ui, "_log", lambda m: lines.append(str(m)))
    return lines


@pytest.fixture
def on(monkeypatch):
    monkeypatch.delenv("IDEOGRAM_API_KEY", raising=False)
    monkeypatch.setitem(cz_magic._prefs, "ideogram_cloud_enabled", True)
    monkeypatch.setitem(cz_magic._prefs, "ideogram_api_key", KEY)


@pytest.fixture
def off(monkeypatch):
    monkeypatch.delenv("IDEOGRAM_API_KEY", raising=False)
    monkeypatch.setitem(cz_magic._prefs, "ideogram_cloud_enabled", False)
    monkeypatch.setitem(cz_magic._prefs, "ideogram_api_key", "")


def _img(w=64, h=48, color=(200, 30, 30)):
    return Image.new("RGB", (w, h), color)


def _journal(tmp_path):
    p = tmp_path / "j.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


# --- Off by default ----------------------------------------------------------------------

def test_off_by_default_nothing_is_sent(server, off, logs):
    ok, why = api.status()
    assert not ok and "ideogram_cloud_enabled" in why and "United States" in why
    for build in (lambda: api.build_remove_background(_img()),
                  lambda: api.build_edit(_img(), "make it blue")):
        req = build()
        with pytest.raises(api.CloudError, match="off"):
            api.run(req)
        with pytest.raises(api.CloudError, match="off"):
            api.quote(req)
    assert server.calls == []


def test_enabled_without_a_key_is_still_off(server, monkeypatch):
    monkeypatch.delenv("IDEOGRAM_API_KEY", raising=False)
    monkeypatch.setitem(cz_magic._prefs, "ideogram_cloud_enabled", True)
    monkeypatch.setitem(cz_magic._prefs, "ideogram_api_key", "")
    ok, why = api.status()
    assert not ok and "API key" in why
    with pytest.raises(api.CloudError, match="API key"):
        api.run(api.build_remove_background(_img()))
    assert server.calls == []


def test_the_ui_greys_the_buttons_out_and_says_why(off, server):
    import gradio as gr
    import cz_cloud_ui
    with gr.Blocks():
        p = cz_cloud_ui.build_tab()
    assert p["run_btn"].interactive is False and p["quote_btn"].interactive is False
    assert "ideogram_cloud_enabled" in p["status"].value
    assert "United States" in p["banner"].value and "Off" in p["banner"].value
    imgs, msg, _ = cz_cloud_ui.handle("remove_background", False, _img(), [], "", "", {},
                                      1024, 1024, "display", None, "png", [])
    assert imgs == [] and "off" in msg
    assert server.calls == []


# --- What is sent -------------------------------------------------------------------------

def test_edit_sends_the_v2_fields_and_the_mask_in_the_api_convention(server, on, logs):
    src = _img(64, 48)
    mask = Image.new("L", (64, 48), 0)
    mask.paste(255, (16, 12, 48, 36))                  # the app's convention: white = edit
    refs = [_img(32, 32, (0, 0, 255)) for _ in range(5)]
    req = api.build_edit(src, "add sunglasses", mask=mask, refs=refs, quality="high",
                         seed=7, num_images=2)
    names = [f[0] for f in req.files]
    assert names.count("reference_images") == 3, "a mask takes a reference slot"
    assert any("dropped" in m for m in logs)
    m = Image.open(io.BytesIO([f for f in req.files if f[0] == "mask"][0][2])).convert("L")
    assert m.getpixel((30, 20)) == 0 and m.getpixel((2, 2)) == 255, \
        "the API wants BLACK = edit, white = keep"
    assert req.fields == {"prompt": "add sunglasses", "quality": "high", "num_images": "2",
                          "seed": "7"}
    server.on("POST", "/v2/image/precise-edit/ideogram-4-5",
              (200, {"generation_kind": "sampling", "generation_id": "g1", "seed": 7,
                     "data": [{"url": "https://cdn.example/r.png", "prompt": "add sunglasses",
                               "resolution": "64x48", "is_image_safe": True, "seed": 7}]}))
    server.on("GET", "https://cdn.example/r.png", (200, _png(_img(64, 48, (1, 2, 3))),
                                                    {"Content-Type": "image/png"}))
    res = api.run(req)
    post, get = server.calls
    assert post["url"] == "https://api.ideogram.ai/v2/image/precise-edit/ideogram-4-5"
    assert post["headers"] == {"Api-Key": KEY}
    assert post["data"]["quality"] == "high"
    assert "Api-Key" not in get["headers"], "the signed result link gets no key"
    assert len(res.images) == 1 and res.images[0].size == (64, 48)
    cloud = [m for m in logs if m.startswith("CLOUD:")]
    assert cloud and "United States" in cloud[0] and "image 64x48" in cloud[0]
    assert all(KEY not in m for m in logs)


def test_a_big_image_keeps_its_resolution_only_with_a_mask(on, logs):
    big = Image.new("RGB", (2100, 2100))
    mask = Image.new("L", big.size, 0)
    mask.paste(255, (0, 0, 300, 300))
    assert api.build_edit(big, "x", mask=mask).fields["context_window"] == "auto"
    req = api.build_edit(big, "x")
    assert "context_window" not in req.fields
    assert any("SCALED DOWN" in m for m in logs), "the degradation is announced"


def test_a_mask_with_nothing_or_everything_painted_is_refused(on):
    for fill in (0, 255):
        with pytest.raises(api.CloudError, match="mask"):
            api.build_edit(_img(), "x", mask=Image.new("L", (64, 48), fill))


def test_private_is_forced_where_the_api_has_it(on):
    for req in (api.build_remove_background(_img()),
                api.build_replace_background(_img(), _img()),
                api.build_vectorize(_img()), api.build_svg("a fox logo")):
        assert req.fields["private"] == "true", req.op


def test_upscale_beyond_8192_is_refused_before_sending(server, on):
    with pytest.raises(api.CloudError, match="8192"):
        api.build_upscale(Image.new("RGB", (2100, 1000)), factor="x4")
    req = api.build_upscale(_img(), factor="x8", model="topaz-text-refine")
    assert req.path == "/v2/image/upscale/topaz-text-refine"
    assert req.fields["upscale_factor"] == "x8"
    assert server.calls == []


def test_transparent_is_a_json_body_with_the_nearest_aspect(server, on):
    cap = '{"high_level_description":"a red fox logo"}'
    req = api.build_transparent(cap, 1376, 768, output_resolution="2k", seed=3)
    assert req.fields["prompt"] == cap, "a JSON caption is passed as is"
    assert req.fields["aspect_ratio"] == "16x9" and req.fields["output_resolution"] == "2k"
    server.on("POST", "/v2/image/generate/ideogram-4-transparent",
              (200, {"generation_id": "t", "seed": 3,
                     "data": [{"url": "https://cdn.example/t.png", "prompt": cap,
                               "resolution": "1376x768", "is_image_safe": True, "seed": 3}]}))
    server.on("GET", "https://cdn.example/t.png",
              (200, _png(Image.new("RGBA", (8, 8), (0, 0, 0, 0)))))
    res = api.run(req)
    assert server.calls[0]["json"]["aspect_ratio"] == "16x9" and server.calls[0]["files"] == []
    assert res.images[0].mode == "RGBA"


def test_remix_weight_and_another_shape_are_refused_together(on):
    with pytest.raises(api.CloudError, match="shape"):
        api.build_remix(_img(64, 48), "x", image_weight=50, resolution="1024x1024")
    req = api.build_remix(_img(64, 64), "x", image_weight=50, resolution="1024x1024")
    assert req.fields["image_weight"] == "50"


# --- Price estimates --------------------------------------------------------------------------

def test_a_price_estimate_is_a_dry_run_and_is_journaled_without_the_key(server, on, logs,
                                                                          tmp_path):
    server.on("POST", "/v2/image/remove-background/ideogram-1",
              (200, {"price": {"amount": 0.04, "currency": "USD"}}))
    q, line = api.quote(api.build_remove_background(_img()))
    assert server.calls[0]["url"].endswith("?dry_run=true")
    assert "0.04" in line
    assert any("PRICE ESTIMATE" in m and "nothing generated" in m for m in logs)
    j = _journal(tmp_path)
    assert j[-1]["dry_run"] is True and j[-1]["op"] == "remove_background"
    assert KEY not in (tmp_path / "j.jsonl").read_text()


def test_format_quote_reads_any_amount_and_falls_back_on_the_raw_answer():
    assert "0.0800 USD" in api.format_quote({"total_usd_micros": 80000})
    assert "credits = 2" in api.format_quote({"credits": 2})
    assert api.format_quote({"weird": "shape"}).startswith("Estimate (raw)")


def test_describe_has_no_price_estimate(server, on):
    with pytest.raises(api.CloudError, match="no price"):
        api.quote(api.build_describe(_img()))
    assert server.calls == []


# --- Async, errors, safety ---------------------------------------------------------------------

def test_async_operations_are_polled_with_retry_after(server, on, logs, tmp_path):
    server.on("POST", "/v2/image/replace-background/ideogram-4-5",
              (200, {"generation_id": "abc"}))
    server.on("GET", "/v2/generations/abc",
              (429, b"", {"Retry-After": "3"}),
              (200, {"generation_id": "abc", "status": "pending", "created": "x"}),
              (200, {"generation_id": "abc", "status": "completed", "created": "x",
                     "usage_cost_usd_micros": 120000,
                     "data": [{"object_type": "image.generation", "url": "https://cdn.example/b.png",
                               "prompt": "", "resolution": "64x48", "is_image_safe": True,
                               "seed": 1}]}))
    server.on("GET", "https://cdn.example/b.png", (200, _png(_img())))
    waits = []
    res = api.run(api.build_replace_background(_img(), _img()), sleep=waits.append)
    assert waits[0] == 3.0, "Retry-After is honoured"
    assert res.cost_usd == pytest.approx(0.12) and len(res.images) == 1
    assert _journal(tmp_path)[-1]["cost_usd"] == pytest.approx(0.12)


def test_a_failed_generation_says_why(server, on):
    server.on("POST", "/v2/svg/vectorize/ideogram-vector-1", (200, {"generation_id": "f"}))
    server.on("GET", "/v2/generations/f",
              (200, {"generation_id": "f", "status": "failed", "created": "x",
                     "failure_reason": "content_policy_violation"}))
    with pytest.raises(api.CloudError, match="content_policy_violation"):
        api.run(api.build_vectorize(_img()), sleep=lambda s: None)


def test_polling_gives_up_at_the_timeout(server, on, monkeypatch):
    monkeypatch.setitem(api.CONFIG, "ideogram_cloud_timeout", 30)
    server.on("POST", "/v2/svg/vectorize/ideogram-vector-1", (200, {"generation_id": "p"}))
    server.on("GET", "/v2/generations/p",
              (200, {"generation_id": "p", "status": "pending", "created": "x"}))
    clock = [0.0]
    with pytest.raises(api.CloudError, match="no result after 30 s"):
        api._poll("p", sleep=lambda s: clock.__setitem__(0, clock[0] + s),
                  now=lambda: clock[0])


@pytest.mark.parametrize("code,body,expect", [
    (401, {"error": "bad key"}, "API key was refused"),
    (402, {"error": "x", "reject_reason": "insufficient_funds"}, "no credit left"),
    (429, {"error": "slow down", "reject_reason": "inflight_limit"}, "too many generations"),
    (422, {"error": "unsafe"}, "safety"),
])
def test_http_errors_are_named(server, on, code, body, expect):
    server.on("POST", "/v2/image/remove-background/ideogram-1", (code, body))
    with pytest.raises(api.CloudError, match=expect):
        api.run(api.build_remove_background(_img()))


def test_a_safety_blocked_result_is_counted_not_downloaded(server, on, logs):
    server.on("POST", "/v2/image/remix/ideogram-4",
              (200, {"generation_id": "r", "seed": 1,
                     "data": [{"url": None, "prompt": "p", "resolution": "64x48",
                               "is_image_safe": False, "seed": 1}]}))
    res = api.run(api.build_remix(_img(), "p"))
    assert res.blocked == 1 and res.images == [] and len(server.calls) == 1
    assert any("refused by Ideogram's safety" in m for m in logs)


# --- Results saved with their metadata ------------------------------------------------------------

def test_results_and_svgs_are_saved_with_metadata(server, on, tmp_path, monkeypatch):
    import cz_imageio
    monkeypatch.setitem(cz_imageio.CONFIG, "date_subfolders", False)
    server.on("POST", "/v2/svg/generate/ideogram-vector-1", (200, {"generation_id": "s"}))
    server.on("GET", "/v2/generations/s",
              (200, {"generation_id": "s", "status": "completed", "created": "x",
                     "data": [{"object_type": "svg.generation", "asset_id": "a",
                               "url": "https://cdn.example/s.svg",
                               "mime_type": "image/svg+xml", "width": 10, "height": 10}]}))
    server.on("GET", "https://cdn.example/s.svg", (200, b"<svg xmlns='x'/>"))
    req = api.build_svg("a fox logo", seed=5)
    res = api.run(req, sleep=lambda s: None)
    assert res.svgs == ["<svg xmlns='x'/>"]
    paths = api.save(req, res, "custom", str(tmp_path / "out"), "jpg")
    assert len(paths) == 1 and paths[0].endswith(".svg")
    meta = json.loads(open(paths[0] + ".json").read())
    assert meta["cloud"] == "ideogram" and meta["endpoint"] == "/v2/svg/generate/ideogram-vector-1"
    assert meta["prompt"] == "a fox logo" and "United States" in meta["where"]
    assert KEY not in json.dumps(meta)


def test_a_cut_out_stays_png_whatever_the_format(server, on, tmp_path, monkeypatch):
    import cz_imageio
    monkeypatch.setitem(cz_imageio.CONFIG, "date_subfolders", False)
    monkeypatch.setattr(cz_imageio, "save_image",
                        lambda img, dst, fmt, meta=None: open(dst, "wb").write(b"x"))
    server.on("POST", "/v2/image/remove-background/ideogram-1",
              (200, {"generation_id": "rb", "data": [{"url": "https://cdn.example/c.png",
                                                      "is_image_safe": True}]}))
    server.on("GET", "https://cdn.example/c.png",
              (200, _png(Image.new("RGBA", (8, 8), (0, 0, 0, 0)))))
    req = api.build_remove_background(_img())
    res = api.run(req)
    paths = api.save(req, res, "custom", str(tmp_path / "o"), "jpg")
    assert paths[0].endswith(".png")


# --- The other entry points ---------------------------------------------------------------------

def test_cloud_describe_goes_through_the_caption_cleaning(server, on):
    server.on("POST", "/v2/image/describe/ideogram-4",
              (200, {"description_id": "d", "created": "x",
                     "json_prompt": {"high_level_description": "A poster.",
                                     "compositional_deconstruction": {
                                         "background": "white", "elements": []}}}))
    text, warnings = cz_magic.describe(_img(), which="ideogram_api")
    assert json.loads(text)["high_level_description"] == "A poster."
    assert server.calls[0]["data"]["include_bbox"] == "true"


def test_cloud_describe_while_off_is_a_magic_prompt_error(server, off):
    with pytest.raises(cz_magic.MagicPromptError, match="off"):
        cz_magic.describe(_img(), which="ideogram_api")
    assert server.calls == []


def test_ui_handle_runs_a_dry_run_and_the_editor_mask_reaches_the_edit(server, on):
    import cz_cloud_ui
    server.on("POST", "/v2/image/precise-edit/ideogram-4-5", (200, {"price_usd": 0.06}))
    bg = _img(64, 48)
    comp = bg.copy()
    comp.paste((255, 255, 255), (10, 10, 30, 30))
    imgs, msg, _ = cz_cloud_ui.handle("edit", True, {"background": bg, "composite": comp},
                                      [None] * 4, "", "make it night", {"quality": "low"},
                                      1024, 1024, "display", None, "png", [])
    assert "0.06" in msg and "Nothing was generated" in msg and imgs == []
    call = server.calls[0]
    assert call["data"]["prompt"] == "make it night", "empty instruction = the main prompt"
    assert any(f[0] == "mask" for f in call["files"])


def test_settings_save_switches_on_and_never_echoes_the_key(off, monkeypatch, tmp_path):
    import cz_cloud_ui
    import cz_core
    saved = {}
    monkeypatch.setattr(cz_core, "_save_prefs_keys", lambda d: saved.update(d))
    cleared, msg, ok = cz_cloud_ui.settings_save(True, KEY)
    assert ok and cleared == "" and KEY not in msg
    assert saved == {"ideogram_cloud_enabled": True, "ideogram_api_key": KEY}
    assert api.status()[0]
    cleared, msg, ok = cz_cloud_ui.settings_save(False, "")
    assert not ok and saved["ideogram_api_key"] == KEY, "an empty key field keeps the saved key"


def test_protocol_caps_report_the_cloud_without_running_it(on):
    import cz_protocol
    c = cz_protocol.caps_dict()["supports"]["cloud"]
    assert c["enabled"] and "edit" in c["ops"] and c["via"] == "ui"


# --- Hidden while off (Mike, 02/11/2026: the studio is used locally) --------------------------

def _caption_panel():
    import gradio as gr
    import cz_caption_ui
    with gr.Blocks():
        return cz_caption_ui.build_panel()


def test_while_off_no_cloud_engine_and_no_cloud_describe(off):
    assert "ideogram_api" not in cz_magic.visible_engines()
    p = _caption_panel()
    assert p["describe_cloud_btn"].visible is False
    assert "ideogram_api" not in [v for _, v in p["engine"].choices]
    assert "United States" not in (p["engine"].info or "")


def test_a_cloud_engine_left_in_the_settings_falls_back_on_local_and_says_so(off, monkeypatch):
    lines = []
    monkeypatch.setattr(cz_magic, "_log", lambda m: lines.append(str(m)))
    monkeypatch.setitem(cz_magic._prefs, "magic_prompt_engine", "ideogram_api")
    cz_magic._ENGINE_FALLBACK_SAID.clear()
    assert cz_magic.engine() == "local" and cz_magic.engine() == "local"
    assert sum("local engine is used instead" in m for m in lines) == 1


def test_while_on_the_cloud_entries_are_shown(on):
    assert "ideogram_api" in cz_magic.visible_engines()
    p = _caption_panel()
    assert p["describe_cloud_btn"].visible is True
    assert "ideogram_api" in [v for _, v in p["engine"].choices]


def _cloud_tab_visible():
    import gradio as gr
    import cz_ui
    demo = cz_ui.build_ui()
    tabs = [b for b in demo.blocks.values()
            if isinstance(b, gr.Tab) and b.label == "Cloud (Ideogram)"]
    assert len(tabs) == 1
    return tabs[0].visible


def test_the_cloud_tab_is_hidden_while_off_and_shown_when_on(off, monkeypatch):
    assert _cloud_tab_visible() is False
    monkeypatch.setitem(cz_magic._prefs, "ideogram_cloud_enabled", True)
    assert _cloud_tab_visible() is True
