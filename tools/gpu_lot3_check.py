"""Lot 3 on the GPU: emulated img2img, inpaint, outpaint, refine (Ideogram 4).

The acceptance of lot 3 is on the CPU (tests/test_ideogram_emulated.py); this script
shows what the emulation gives on the real weights, through the app's own functions:

  0. INIT       bench/lot2/hand.png when it exists, else a fresh 1024 render of a
                JSON poster caption.
  1. IMG2IMG    the init at denoise 0.3 and 0.6 (P._refine_whole), same prompt.
  2. INPAINT    a box in the middle repainted with another subject (P.inpaint_run).
  3. OUTPAINT   the init extended to the right (P.outpaint_directions).
  4. REFRAME    the init reframed to 16:9 (P.reframe, contain).

Each output is saved in bench/lot3/ with its time; report.md lists them. Look at them:
the numbers say nothing about whether the text survived a 0.6 img2img.
THE GPU MUST BE FREE (same check as tools/gpu_lot1_check.py).

Usage:  .venv\\Scripts\\python tools\\gpu_lot3_check.py [--steps 48]
"""
import json
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
OUT = os.path.join(ROOT, "bench", "lot3")
RESULTS = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": {}}


def _say(msg):
    print(f"[lot3] {msg}", flush=True)


def _save():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False)
    lines = ["# Lot 3 on the GPU: emulated img2img / inpaint", "",
             f"Started {RESULTS['started']}" + (f", finished {RESULTS['finished']}"
                                                if "finished" in RESULTS else ", NOT finished"),
             "", "| step | seconds | size | image / error |", "|---|---|---|---|"]
    for k, v in RESULTS["steps"].items():
        lines.append(f"| {k} | {v.get('seconds', '')} | {v.get('size', '')} | "
                     f"{v.get('image') or v.get('error', '')} |")
    with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _step(tag, fn):
    t0 = time.time()
    try:
        img = fn()
        path = os.path.join(OUT, f"{tag}.png")
        img.save(path)
        RESULTS["steps"][tag] = {"seconds": round(time.time() - t0, 1),
                                 "size": f"{img.width}x{img.height}",
                                 "image": os.path.relpath(path, ROOT)}
        _say(f"{tag}: {img.width}x{img.height} in {time.time() - t0:.1f} s")
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
    from PIL import Image, ImageDraw
    G1.OUT = OUT
    os.makedirs(OUT, exist_ok=True)
    steps = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 48
    P = G1.prechecks()
    import cz_caption
    import cz_design

    hand = os.path.join(ROOT, "bench", "lot2", "hand.png")
    if os.path.isfile(hand):
        init = Image.open(hand).convert("RGB")
        prompt = open(os.path.join(ROOT, "bench", "lot2", "hand.json"), encoding="utf-8").read() \
            if os.path.isfile(os.path.join(ROOT, "bench", "lot2", "hand.json")) else "a poster"
        _say(f"init: {hand}")
    else:
        prompt = cz_caption.to_text(cz_design.caption_of("Poster (A-series, portrait)"))
        init = _step("0_init", lambda: P.generate(prompt, 720, 1024, steps, 12345))
        if init is None:
            sys.exit(1)

    _step("1_img2img_0.3", lambda: P._refine_whole(P.get_pipe("img2img"), init, 0.3, steps,
                                                   prompt, 7))
    _step("1_img2img_0.6", lambda: P._refine_whole(P.get_pipe("img2img"), init, 0.6, steps,
                                                   prompt, 7))
    w, h = init.size
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rectangle([w // 4, h // 3, 3 * w // 4, 2 * h // 3], fill=255)
    mask.save(os.path.join(OUT, "2_inpaint_mask.png"))
    _step("2_inpaint", lambda: P.inpaint_run(
        init, mask, "a large red retro bicycle, flat vector illustration", steps, 1.0, 7))
    _step("3_outpaint_right", lambda: P.outpaint_directions(init, None, ["right"], prompt,
                                                            steps, 7))
    _step("4_reframe_16x9", lambda: P.reframe(init, 16, 9, "contain", prompt, steps, 7))
    RESULTS["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _save()
    _say(f"done: {os.path.relpath(os.path.join(OUT, 'report.md'), ROOT)}")


if __name__ == "__main__":
    main()
