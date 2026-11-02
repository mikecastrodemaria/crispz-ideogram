"""Lot 5: the local alternatives to the cloud functions. Remove BG (rembg) saves a
transparent PNG with its metadata; nothing leaves the machine."""
import json
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_ui  # noqa: E402


def test_remove_bg_saves_a_transparent_png_with_metadata(tmp_path, monkeypatch):
    import cz_imageio
    monkeypatch.setitem(cz_imageio.CONFIG, "date_subfolders", False)
    monkeypatch.setattr(cz_ui, "_remove_bg",
                        lambda im: Image.new("RGBA", im.size, (0, 0, 0, 0)))
    imgs, msg, hist, _ = cz_ui._ui_remove_bg(Image.new("RGB", (40, 30)), [], "custom",
                                             str(tmp_path))
    assert imgs[0].mode == "RGBA" and "locally" in msg and hist[0] is imgs[0]
    pngs = [p for p in os.listdir(tmp_path) if p.endswith(".png")]
    assert len(pngs) == 1
    meta = json.load(open(tmp_path / (pngs[0] + ".json")))
    assert meta["local"] is True and meta["tool"].startswith("rembg")


def test_remove_bg_failure_is_said(monkeypatch):
    def boom(im):
        raise RuntimeError("rembg not installed. pip install rembg")
    monkeypatch.setattr(cz_ui, "_remove_bg", boom)
    imgs, msg, _, _ = cz_ui._ui_remove_bg(Image.new("RGB", (8, 8)), [], "display", "")
    assert imgs == [] and "rembg not installed" in msg


def test_the_flux3_edit_tab_and_input_mode_are_hidden():
    """Ideogram 4 does not edit: the inherited FLUX.3 'Edit (references)' tab and input
    mode must not be offered (editing goes to the Edit in crispz-flux3 tab)."""
    import gradio as gr
    demo = cz_ui.build_ui()
    tabs = {b.label: b.visible for b in demo.blocks.values() if isinstance(b, gr.Tab)}
    assert tabs.get(cz_ui.EDIT_MODE) is False
    assert tabs.get("Edit in crispz-flux3") is True and tabs.get("Vectorize (SVG)") is True
    radios = [b for b in demo.blocks.values()
              if isinstance(b, gr.Radio) and b.label == "Input mode"]
    assert radios and radios[0].visible is False
