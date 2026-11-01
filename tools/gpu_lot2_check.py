"""Lot 2 validation on the GPU: JSON captions, local magic prompt, local Describe.

The acceptance of lot 2 (CLAUDE.md): a caption built by hand and a caption from the
LOCAL magic prompt both validate against the schema and both give an image. On top:

  1. HAND        the "Poster (A-series)" design preset with real texts -> validate ->
                 render (1024 wide, 48 steps).
  2. MAGIC       two plain ideas -> local magic prompt (Qwen3-VL-8B + LM head, outlines)
                 -> time, validity, warnings -> render the first.
  3. DESCRIBE    the HAND render -> local Describe -> caption: validity, does it read
                 the texts back.
  4. BLOCKED     every render is checked for Ideogram's "blocked by safety filter" card
                 (a flat grey image): it should never appear on these JSON captions.
  5. --fp-test   optional: the same harmless ideas as PLAIN text and as JSON, same seed,
                 counts the blocked cards on each side (Ideogram documents a high false
                 positive rate on non-JSON prompts).

Everything lands in bench/lot2/ (images, captions .json, results.json, report.md).
THE GPU MUST BE FREE (same check as tools/gpu_lot1_check.py).

Usage:
    .venv\\Scripts\\python tools\\gpu_lot2_check.py
    .venv\\Scripts\\python tools\\gpu_lot2_check.py --fp-test --steps 24
"""
import json
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
OUT = os.path.join(ROOT, "bench", "lot2")

IDEAS = [
    "a minimalist Swiss-style poster: a huge black \"4.5\" on an off-white grid, a red "
    "circle overlapping the numerals, a small caption \"Ideogram, real typography\" in the "
    "lower left corner",
    "a vintage coffee shop logo with the words \"BEAN THERE\" around a steaming cup",
]
HAND_TEXTS = {"HEADLINE": "GRAND OPENING", "DATE · PLACE": "SATURDAY 9 AM · RUE DU PAIN",
              "THE SUBJECT": "a basket of golden croissants"}

RESULTS = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": {}}


def _say(msg):
    print(f"[lot2] {msg}", flush=True)


def _arg(name, default=None, cast=str):
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            return cast(sys.argv[i + 1])
    return default


def _save():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False)
    _report()


def is_blocked_card(img):
    """Ideogram's refusal: a flat mid-grey frame with one line of light text. A real
    render is never that flat: < 2 % of the pixels away from the median grey by more than
    25 levels, and that median a neutral grey."""
    import numpy as np
    a = np.asarray(img.convert("RGB").resize((256, 256)), dtype=np.int16)
    med = np.median(a.reshape(-1, 3), axis=0)
    neutral = abs(med[0] - med[1]) < 8 and abs(med[1] - med[2]) < 8 and 80 < med[0] < 170
    off = (np.abs(a - med).max(axis=2) > 25).mean()
    return bool(neutral and off < 0.02)


def hand_caption():
    import cz_caption
    import cz_design
    name = "Poster (A-series, portrait)"
    p = cz_design.PRESETS[name]
    els = []
    for e in p["elements"]:
        e = dict(e)
        if e.get("type") == "text" and e.get("text") in HAND_TEXTS:
            e["text"] = HAND_TEXTS[e["text"]]
        e["desc"] = e["desc"].replace("THE SUBJECT", HAND_TEXTS["THE SUBJECT"])
        els.append(e)
    style = dict(p["style"], color_palette="#F4E9D8 #C8102E #1D1D1B #E8A33D")
    cap = cz_caption.build("A bold bakery grand-opening poster, flat graphic design.",
                           style, p["background"], els)
    return cap, cz_design.size_of(name)


def render(P, tag, caption_text, w, h, steps, seed):
    import torch
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    img = P.generate(caption_text, w, h, steps, seed)
    dt = time.time() - t0
    path = os.path.join(OUT, f"{tag}.png")
    img.save(path)
    blocked = is_blocked_card(img)
    r = {"size": f"{img.width}x{img.height}", "seconds": round(dt, 1),
         "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2),
         "blocked_card": blocked, "image": os.path.relpath(path, ROOT)}
    _say(f"{tag}: {r['size']} in {r['seconds']} s, blocked card: {blocked}")
    return img, r


def _write_caption(tag, text):
    with open(os.path.join(OUT, f"{tag}.json"), "w", encoding="utf-8") as f:
        f.write(text)


def main():
    import gpu_lot1_check as G1
    G1.OUT = OUT
    os.makedirs(OUT, exist_ok=True)
    steps = _arg("--steps", 48, int)
    seed = _arg("--seed", 12345, int)
    P = G1.prechecks()
    import cz_caption
    import cz_magic
    from diffusers.utils import is_outlines_available
    RESULTS["steps"]["0_env"] = {"outlines": is_outlines_available()}
    if not is_outlines_available():
        _say("outlines is NOT installed: the local magic prompt is unconstrained. "
             "pip install -r requirements.txt (outlines>=1.3) and run again.")
    off = _arg("--offload")
    if off:
        P.set_offload_mode(off)

    # 1. Hand-built caption
    try:
        cap, (w, h) = hand_caption()
        text = cz_caption.to_text(cap)
        _write_caption("hand", text)
        warnings = cz_caption.validate(cap)
        k = 1024 / float(max(w, h))
        w1, h1 = int(w * k) // 16 * 16, int(h * k) // 16 * 16
        _, r = render(P, "hand", text, w1, h1, steps, seed)
        r["schema_warnings"] = warnings
        RESULTS["steps"]["1_hand"] = r
    except Exception as e:
        RESULTS["steps"]["1_hand"] = {"error": f"{type(e).__name__}: {e}",
                                      "trace": traceback.format_exc()}
        _say(f"hand FAILED: {e}")
    _save()

    # 2. Local magic prompt
    magic = []
    for i, idea in enumerate(IDEAS):
        try:
            t0 = time.time()
            text, warnings, used = cz_magic.expand(idea, 1024, 1024, seed=seed + i,
                                                   which="local")
            dt = time.time() - t0
            _write_caption(f"magic_{i + 1}", text)
            r = {"idea": idea, "seconds": round(dt, 1), "schema_warnings": warnings,
                 "valid": not warnings, "chars": len(text)}
            _say(f"magic {i + 1}: {r['seconds']} s, {len(warnings)} warning(s)")
            if i == 0:
                _, rr = render(P, "magic_1", text, 1024, 1024, steps, seed)
                r["render"] = rr
            magic.append(r)
        except Exception as e:
            magic.append({"idea": idea, "error": f"{type(e).__name__}: {e}",
                          "trace": traceback.format_exc()})
            _say(f"magic {i + 1} FAILED: {type(e).__name__}: {e}")
        RESULTS["steps"]["2_magic"] = magic
        _save()

    # 3. Describe the hand render
    try:
        from PIL import Image
        src = os.path.join(OUT, "hand.png")
        t0 = time.time()
        text, warnings = cz_magic.describe(Image.open(src))
        dt = time.time() - t0
        _write_caption("describe_hand", text)
        found = [t for t in ("GRAND OPENING", "SATURDAY") if t.lower() in text.lower()]
        RESULTS["steps"]["3_describe"] = {"seconds": round(dt, 1), "schema_warnings": warnings,
                                          "texts_read_back": found}
        _say(f"describe: {dt:.1f} s, {len(warnings)} warning(s), texts read back: {found}")
    except Exception as e:
        RESULTS["steps"]["3_describe"] = {"error": f"{type(e).__name__}: {e}",
                                          "trace": traceback.format_exc()}
        _say(f"describe FAILED: {type(e).__name__}: {e}")
    _save()

    # 5. False positives, plain vs JSON (optional)
    if "--fp-test" in sys.argv:
        fp = {"plain_blocked": 0, "json_blocked": 0, "n": 0, "n_json": 0, "cases": []}
        for i, idea in enumerate(IDEAS):
            case = {"idea": idea}
            try:
                _, r = render(P, f"fp_plain_{i + 1}", idea, 1024, 1024, steps, seed)
                case["plain_blocked"] = r["blocked_card"]
                jtext = None
                for m in magic:
                    if m.get("idea") == idea and not m.get("error"):
                        with open(os.path.join(OUT, f"magic_{i + 1}.json"), encoding="utf-8") as f:
                            jtext = f.read()
                if jtext:
                    _, r = render(P, f"fp_json_{i + 1}", jtext, 1024, 1024, steps, seed)
                    case["json_blocked"] = r["blocked_card"]
                fp["n"] += 1
                fp["plain_blocked"] += int(bool(case.get("plain_blocked")))
                fp["json_blocked"] += int(bool(case.get("json_blocked")))
                fp["n_json"] += int("json_blocked" in case)
            except Exception as e:
                case["error"] = f"{type(e).__name__}: {e}"
            fp["cases"].append(case)
            RESULTS["steps"]["5_false_positives"] = fp
            _save()
        _say(f"false positives: plain {fp['plain_blocked']}/{fp['n']}, "
             f"JSON {fp['json_blocked']}/{fp['n_json']}")

    RESULTS["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _save()
    _say(f"done: {os.path.relpath(os.path.join(OUT, 'report.md'), ROOT)}")


def _report():
    s = RESULTS["steps"]
    L = ["# Lot 2 on the GPU: JSON captions", "",
         f"Started {RESULTS['started']}" + (f", finished {RESULTS['finished']}"
                                            if "finished" in RESULTS else ", NOT finished"), ""]
    env = s.get("0_env")
    if env:
        L += [f"- outlines installed: {env['outlines']}", ""]
    h = s.get("1_hand")
    if h:
        L += ["## Hand-built caption (poster preset)", "", "```", json.dumps(h, indent=1,
              ensure_ascii=False), "```", ""]
    m = s.get("2_magic")
    if m:
        L += ["## Local magic prompt", "", "| idea | s | warnings | render s | blocked |",
              "|---|---|---|---|---|"]
        for r in m:
            if "error" in r:
                L.append(f"| {r['idea'][:40]} | ERROR {r['error']} | | | |")
            else:
                rr = r.get("render") or {}
                L.append(f"| {r['idea'][:40]} | {r['seconds']} | {len(r['schema_warnings'])} | "
                         f"{rr.get('seconds', '')} | {rr.get('blocked_card', '')} |")
        L.append("")
    d = s.get("3_describe")
    if d:
        L += ["## Describe (local)", "", "```", json.dumps(d, indent=1, ensure_ascii=False),
              "```", ""]
    fp = s.get("5_false_positives")
    if fp:
        L += ["## False positives (blocked cards)", "",
              f"Plain text: {fp['plain_blocked']}/{fp['n']}, JSON: "
              f"{fp['json_blocked']}/{fp.get('n_json', 0)}", ""]
    with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
