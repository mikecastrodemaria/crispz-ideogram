"""crispz-ideogram - local vectorization: an image -> an SVG (lot 5).

The local answer to Ideogram's cloud vectorizer: vtracer (visioncortex, MIT, a Rust
library with Python wheels for Windows, Linux and macOS). Nothing leaves the machine.

It traces colour regions into SVG paths, so it suits what Ideogram 4 is made for here:
logos, icons, flat illustrations, posters with solid colours. A photo gives a heavy SVG
of thousands of patches (said in the result line, with the path count and the size).

Two steps help a generated logo:
  - `colors` > 0 quantizes the image to that many colours first (PIL median cut, no
    dithering): the faint gradients and the noise of a diffusion render otherwise each
    become shapes of their own;
  - `remove_bg` cuts the subject out first (the local rembg of the Remove BG tab): the
    background then has no path at all instead of one big rectangle.
"""

import io
import os
import re
import time

from PIL import Image

from cz_core import _log

PRESETS = {
    "Logo (flat colours)": dict(colormode="color", hierarchical="stacked", mode="spline",
                                filter_speckle=8, color_precision=6, layer_difference=24,
                                corner_threshold=60, length_threshold=4.0,
                                splice_threshold=45, path_precision=3),
    "Illustration (more colours)": dict(colormode="color", hierarchical="stacked",
                                        mode="spline", filter_speckle=4, color_precision=7,
                                        layer_difference=12, corner_threshold=60,
                                        length_threshold=4.0, splice_threshold=45,
                                        path_precision=3),
    "Black and white": dict(colormode="binary", mode="spline", filter_speckle=6,
                            corner_threshold=60, length_threshold=4.0,
                            splice_threshold=45, path_precision=3),
    "Sharp corners (pixel art, geometric)": dict(colormode="color", hierarchical="stacked",
                                                 mode="polygon", filter_speckle=2,
                                                 color_precision=6, layer_difference=16,
                                                 path_precision=2),
}
DEFAULT_PRESET = "Logo (flat colours)"
MAX_SIDE = 4096           # larger inputs are scaled down first (said): tracing time explodes


class VectorError(RuntimeError):
    """A vectorization that cannot run, with a message to show as is."""


def available():
    """(ok, reason)."""
    try:
        import vtracer  # noqa: F401
        return True, "vtracer (local)"
    except Exception as e:
        return False, (f"vtracer is not installed ({type(e).__name__}): "
                       f".venv\\Scripts\\pip install vtracer==0.6.15")


def quantize(img, colors):
    """RGBA image -> the same with at most `colors` colours (alpha kept as is)."""
    alpha = img.getchannel("A") if img.mode == "RGBA" else None
    rgb = img.convert("RGB").quantize(colors=int(colors), method=Image.Quantize.MEDIANCUT,
                                      dither=Image.Dither.NONE).convert("RGB")
    if alpha is not None:
        rgb.putalpha(alpha)
    return rgb


def vectorize(image, preset=DEFAULT_PRESET, colors=0, remove_bg=False):
    """-> (svg_text, info dict). Raises VectorError."""
    ok, why = available()
    if not ok:
        raise VectorError(why)
    import vtracer
    if image is None:
        raise VectorError("no image")
    opts = PRESETS.get(preset)
    if opts is None:
        raise VectorError(f"unknown preset {preset!r}")
    t0 = time.time()
    img = image if isinstance(image, Image.Image) else Image.open(image)
    img = img.convert("RGBA")
    if max(img.size) > MAX_SIDE:
        k = MAX_SIDE / max(img.size)
        new = (max(1, round(img.width * k)), max(1, round(img.height * k)))
        _log(f"vectorize: {img.width}x{img.height} scaled down to {new[0]}x{new[1]} "
             f"before tracing (over {MAX_SIDE} px a side)")
        img = img.resize(new, Image.LANCZOS)
    if remove_bg:
        import cz_face
        img = cz_face._remove_bg(img).convert("RGBA")
    if colors and int(colors) > 0 and opts.get("colormode") != "binary":
        img = quantize(img, int(colors))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    svg = vtracer.convert_raw_image_to_svg(buf.getvalue(), img_format="png", **opts)
    info = {"tool": "vtracer", "preset": preset, "options": opts,
            "colors": int(colors or 0), "remove_bg": bool(remove_bg),
            "size": [img.width, img.height], "paths": count_paths(svg),
            "bytes": len(svg.encode("utf-8")), "seconds": round(time.time() - t0, 2)}
    _log(f"vectorize (local, vtracer, {preset}): {info['paths']} paths, "
         f"{info['bytes'] / 1024:.0f} KB in {info['seconds']} s")
    return svg, info


def count_paths(svg):
    return len(re.findall(r"<path\b", svg or ""))


def summary(info):
    heavy = (" Heavy for a vector: a photo-like image traces into thousands of patches; "
             "try fewer colours or Black and white." if info["paths"] > 3000 else "")
    return (f"Vectorized locally ({info['preset']}): {info['paths']} paths, "
            f"{info['bytes'] / 1024:.0f} KB, {info['seconds']} s.{heavy}")


def preview_html(svg, max_height=420):
    """The SVG shown through an <img> data URI: the browser draws it, and an <img> never
    runs what an SVG may carry."""
    import base64
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return (f'<div style="background:repeating-conic-gradient(#ccc 0 25%,#fff 0 50%) 0 0/'
            f'16px 16px;padding:8px;text-align:center"><img alt="SVG preview" '
            f'style="max-width:100%;max-height:{int(max_height)}px" '
            f'src="data:image/svg+xml;base64,{b64}"></div>')


def save(svg, info, save_mode="local", output_dir=None, source_name=None):
    """Writes the .svg and its .json sidecar like the other outputs. -> path or None."""
    import json
    from cz_imageio import build_output_path
    if save_mode == "display":
        return None
    dst = build_output_path(None, save_mode, output_dir, "png", tag="vector",
                            size=tuple(info.get("size") or (0, 0)))
    if not dst:
        return None
    dst = os.path.splitext(dst)[0] + ".svg"
    with open(dst, "w", encoding="utf-8") as f:
        f.write(svg)
    meta = {"local": True, "op": "vectorize", **info}
    if source_name:
        meta["source"] = source_name
    with open(dst + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return dst
