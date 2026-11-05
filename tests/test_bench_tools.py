"""tools/gpu_lot2_check.py without a GPU: the 'blocked by safety filter' card detector and
the hand-made caption that passes the schema with real texts."""
# scope: ideogram
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from PIL import Image, ImageDraw  # noqa: E402

import gpu_lot2_check as G  # noqa: E402


def test_blocked_card_detector():
    card = Image.new("RGB", (768, 1344), (112, 112, 110))
    ImageDraw.Draw(card).text((60, 650), "Image blocked by safety filter", fill=(220, 220, 215))
    assert G.is_blocked_card(card)
    for f in ("sample-gen.jpg", "sample-edit.jpg"):
        assert not G.is_blocked_card(Image.open(os.path.join(ROOT, "assets", f)))
    assert not G.is_blocked_card(Image.new("RGB", (512, 512), (200, 20, 20)))
    poster = Image.new("RGB", (512, 512), (120, 120, 120))
    ImageDraw.Draw(poster).rectangle([50, 50, 460, 300], fill=(10, 10, 10))
    assert not G.is_blocked_card(poster)


def test_hand_caption_is_valid_with_real_texts():
    import cz_caption
    cap, size = G.hand_caption()
    assert cz_caption.validate(cap) == [] and size == (1440, 2032)
    texts = [e.get("text") for e in cap["compositional_deconstruction"]["elements"]]
    assert "GRAND OPENING" in texts and "HEADLINE" not in texts
