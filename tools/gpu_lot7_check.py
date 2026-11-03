"""Lot 7 on the GPU: the refine steps rescaled for Ideogram, the tiled upscale, the
detailer (the open points of lot 3), through the app's own functions.

  0. INIT      bench/lot2/hand.png when it exists (the lot 2 poster), else a 1024 render
               of the poster preset; and a 1024 portrait with a face for the detailer.
  1. UPSCALE   ESRGAN x2 + refine, denoise 0.30, at 12 steps (FLUX.3's old preset) and
               at 36 (the new one): the same seed, look at the TEXT in both.
  2. TILED     the "4K (tiled)" preset: x4, 1024 tiles, denoise 0.20, 36 steps.
  3. DETAILER  detail_faces on the portrait (the emulated inpaint per face crop).

Each output goes to bench/lot7/ with its time and VRAM peak; report.md lists them. The
numbers say whether it runs; the images say whether 36 steps is worth the time.
THE GPU MUST BE FREE (same check as tools/gpu_lot1_check.py): close crispz-flux3 and the
app. The ESRGAN model of the config must be in upscale_models/.

Usage:  .venv\\Scripts\\python tools\\gpu_lot7_check.py [--skip-tiled] [--steps-old 12]
"""
import json
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
OUT = os.path.join(ROOT, "bench", "lot7")
RESULTS = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": {}}
PORTRAIT = ('{"high_level_description":"A friendly studio portrait of a woman in her '
            'thirties smiling at the camera, head and shoulders.","style_description":'
            '{"photo":"85mm portrait, soft window light"},"compositional_deconstruction":'
            '{"background":"a plain warm grey backdrop","elements":[{"type":"subject",'
            '"description":"a smiling woman with short dark hair","bbox":[120,250,950,750]}]}}')


def _say(msg):
    print(f"[lot7] {msg}", flush=True)


def _peak():
    try:
        import torch
        return round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2)
    except Exception:
        return None


def _save():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False)
    lines = ["# Lot 7 on the GPU: refine steps, tiled upscale, detailer", "",
             f"Started {RESULTS['started']}" + (f", finished {RESULTS['finished']}"
                                                if "finished" in RESULTS else ", NOT finished"),
             "", "| step | seconds | peak GB | size | image / error |", "|---|---|---|---|---|"]
    for k, v in RESULTS["steps"].items():
        lines.append(f"| {k} | {v.get('seconds', '')} | {v.get('peak_gb', '')} | "
                     f"{v.get('size', '')} | {v.get('image') or v.get('error', '')} |")
    with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _step(tag, fn):
    try:
        import torch
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass                                  # a CPU dry run
    t0 = time.time()
    try:
        img = fn()
        if isinstance(img, tuple):
            img = img[0]
        path = os.path.join(OUT, f"{tag}.png")
        img.save(path)
        RESULTS["steps"][tag] = {"seconds": round(time.time() - t0, 1), "peak_gb": _peak(),
                                 "size": f"{img.width}x{img.height}",
                                 "image": os.path.relpath(path, ROOT)}
        _say(f"{tag}: {img.width}x{img.height} in {time.time() - t0:.1f} s, "
             f"peak {_peak()} GB")
        return img
    except Exception as e:
        RESULTS["steps"][tag] = {"error": f"{type(e).__name__}: {e}",
                                 "trace": traceback.format_exc()}
        _say(f"{tag} FAILED: {type(e).__name__}: {e}")
        return None
    finally:
        _save()


def main():
    import gpu_lot1_check as G1
    from PIL import Image
    G1.OUT = OUT
    os.makedirs(OUT, exist_ok=True)
    old = int(sys.argv[sys.argv.index("--steps-old") + 1]) if "--steps-old" in sys.argv else 12
    P = G1.prechecks()
    import cz_caption
    import cz_design
    import cz_detailer
    from cz_core import DEFAULT_MODEL as model     # the app's default ESRGAN model
    _say(f"ESRGAN model: {model}")
    hand = os.path.join(ROOT, "bench", "lot2", "hand.png")
    if os.path.isfile(hand):
        poster = Image.open(hand).convert("RGB")
        prompt = (open(os.path.join(ROOT, "bench", "lot2", "hand.json"), encoding="utf-8").read()
                  if os.path.isfile(os.path.join(ROOT, "bench", "lot2", "hand.json"))
                  else "a poster")
        _say(f"poster: {hand}")
    else:
        prompt = cz_caption.to_text(cz_design.caption_of("Poster (A-series, portrait)"))
        poster = _step("0_poster", lambda: P.generate(prompt, 720, 1024, 48, 12345))
        if poster is None:
            sys.exit(1)
    face = _step("0_portrait", lambda: P.generate(PORTRAIT, 1024, 1024, 48, 777))

    for n in (old, 36):
        _step(f"1_upscale_x2_refine_{n}steps", lambda n=n: P.process_one(
            poster, model, 2.0, 0.30, n, prompt, 7, 1024, 32, refine_tile=0))
    if "--skip-tiled" not in sys.argv:
        _step("2_upscale_x4_tiled_36steps", lambda: P.process_one(
            poster, model, 4.0, 0.20, 36, prompt, 7, 1024, 32, refine_tile=1024,
            refine_overlap=64))
    if face is not None:
        def faces():
            img, n = cz_detailer.detail_faces(face, PORTRAIT, 7, steps=36)
            RESULTS["faces_detailed"] = n
            _say(f"faces detailed: {n}")
            return img
        _step("3_detail_faces", faces)
    RESULTS["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _save()
    _say(f"done: {os.path.relpath(os.path.join(OUT, 'report.md'), ROOT)}")


if __name__ == "__main__":
    main()
