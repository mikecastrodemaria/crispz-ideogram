"""Lot 5: local vectorization (cz_vector, the Vectorize (SVG) tab). vtracer, nothing
leaves the machine."""
import json
import os
import sys

import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_vector as V  # noqa: E402

pytestmark = pytest.mark.skipif(not V.available()[0], reason="vtracer not installed")


def _logo(bg=(255, 255, 255, 255)):
    img = Image.new("RGBA", (256, 256), bg)
    d = ImageDraw.Draw(img)
    d.ellipse([30, 30, 226, 226], fill=(220, 40, 40, 255))
    d.rectangle([90, 100, 166, 160], fill=(20, 40, 160, 255))
    return img


def test_every_preset_gives_an_svg():
    for preset in V.PRESETS:
        svg, info = V.vectorize(_logo(), preset, colors=4)
        assert svg.lstrip().startswith("<?xml") or "<svg" in svg[:200]
        assert info["paths"] >= 1 and info["tool"] == "vtracer" and info["size"] == [256, 256]


def test_a_transparent_background_traces_no_background_shape():
    _, opaque = V.vectorize(_logo(), V.DEFAULT_PRESET)
    _, cut = V.vectorize(_logo(bg=(0, 0, 0, 0)), V.DEFAULT_PRESET)
    assert cut["paths"] < opaque["paths"]


def test_quantize_caps_the_colours_and_keeps_alpha():
    img = Image.new("RGBA", (64, 64))
    for x in range(64):
        for y in range(64):
            img.putpixel((x, y), (x * 4, y * 4, 128, 0 if x < 8 else 255))
    q = V.quantize(img, 6)
    assert len(q.convert("RGB").getcolors(4096)) <= 6
    assert q.getpixel((2, 2))[3] == 0 and q.getpixel((40, 40))[3] == 255


def test_fewer_colours_give_fewer_paths_on_a_noisy_image():
    img = Image.effect_noise((128, 128), 60).convert("RGB")
    _, many = V.vectorize(img, V.DEFAULT_PRESET, colors=0)
    _, few = V.vectorize(img, V.DEFAULT_PRESET, colors=3)
    assert few["paths"] < many["paths"]


def test_remove_bg_goes_through_the_local_rembg(monkeypatch):
    import cz_face
    seen = []
    monkeypatch.setattr(cz_face, "_remove_bg",
                        lambda im: seen.append(im.size) or _logo(bg=(0, 0, 0, 0)))
    _, info = V.vectorize(_logo(), V.DEFAULT_PRESET, remove_bg=True)
    assert seen == [(256, 256)] and info["remove_bg"] is True


def test_a_huge_image_is_scaled_down_and_said(monkeypatch):
    lines = []
    monkeypatch.setattr(V, "_log", lambda m: lines.append(str(m)))
    monkeypatch.setattr(V, "MAX_SIDE", 128)
    _, info = V.vectorize(_logo(), V.DEFAULT_PRESET)
    assert info["size"] == [128, 128] and any("scaled down" in m for m in lines)


def test_missing_vtracer_is_named(monkeypatch):
    monkeypatch.setattr(V, "available", lambda: (False, "vtracer is not installed: pip ..."))
    with pytest.raises(V.VectorError, match="vtracer is not installed"):
        V.vectorize(_logo())


def test_save_writes_the_svg_and_its_sidecar(tmp_path, monkeypatch):
    import cz_imageio
    monkeypatch.setitem(cz_imageio.CONFIG, "date_subfolders", False)
    svg, info = V.vectorize(_logo(), V.DEFAULT_PRESET)
    path = V.save(svg, info, "custom", str(tmp_path))
    assert path.endswith(".svg") and open(path).read() == svg
    meta = json.load(open(path + ".json"))
    assert meta["local"] is True and meta["preset"] == V.DEFAULT_PRESET
    assert V.save(svg, info, "display", str(tmp_path)) is None


def test_the_preview_is_an_img_never_inline_svg():
    html = V.preview_html('<svg xmlns="http://www.w3.org/2000/svg"><script>x</script></svg>')
    assert "<img" in html and "data:image/svg+xml;base64," in html and "<script" not in html


def test_the_tab_handler_returns_preview_file_and_status(tmp_path):
    import cz_ui
    html, path, msg = cz_ui._ui_vectorize(_logo(), V.DEFAULT_PRESET, 4, False, "display",
                                          str(tmp_path))
    assert "<img" in html and path.endswith(".svg") and os.path.isfile(path)
    assert "Vectorized locally" in msg and "Display only" in msg
    os.remove(path)
    assert cz_ui._ui_vectorize(None, V.DEFAULT_PRESET, 4, False, "display", "")[2] == \
        "Drop an image first."
