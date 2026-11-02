"""Lot 4 against the real Ideogram API (no GPU needed).

  1. STATUS     the two switches (ideogram_cloud_enabled, the API key). With them off,
                the script stops there: that IS the first acceptance check (nothing sent).
  2. QUOTES     a dry-run price estimate of every function (nothing generated or billed),
                through the app's own request builders. A dry run still SENDS the image:
                it goes to api.ideogram.ai (United States) to be validated.
  3. SPEND      only with --spend: a precise edit (quality 'low') and a remove background
                on bench/lot2/hand.png (or a drawn test card), saved in bench/lot4/ with
                their metadata. This is billed by Ideogram.

The key is read like the app reads it (preferences.json, config, or env IDEOGRAM_API_KEY)
and is never printed. Output: bench/lot4/report.md, results.json, images.

Usage:  .venv\\Scripts\\python tools\\api_lot4_check.py            (status + quotes)
        .venv\\Scripts\\python tools\\api_lot4_check.py --spend    (+ edit + remove bg)
"""
import json
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "bench", "lot4")
RESULTS = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "quotes": {}, "runs": {}}


def _say(msg):
    print(f"[lot4] {msg}", flush=True)


def _save():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False, default=str)
    lines = ["# Lot 4: Ideogram cloud functions against the real API", "",
             f"Started {RESULTS['started']}", "", f"Status: {RESULTS.get('status')}", "",
             "## Price estimates (dry run)", "", "| function | estimate / error |", "|---|---|"]
    for k, v in RESULTS["quotes"].items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "## Real calls (--spend)", "", "| function | seconds | result |",
              "|---|---|---|"]
    for k, v in RESULTS["runs"].items():
        lines.append(f"| {k} | {v.get('seconds', '')} | {v.get('result') or v.get('error')} |")
    with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _test_image():
    from PIL import Image, ImageDraw
    hand = os.path.join(ROOT, "bench", "lot2", "hand.png")
    if os.path.isfile(hand):
        return Image.open(hand).convert("RGB"), hand
    img = Image.new("RGB", (768, 1024), (240, 232, 214))
    d = ImageDraw.Draw(img)
    d.rectangle([84, 120, 684, 520], fill=(200, 40, 40))
    d.ellipse([234, 600, 534, 900], fill=(30, 60, 160))
    return img, "drawn test card"


def main():
    import cz_ideogram_api as api
    from PIL import Image
    spend = "--spend" in sys.argv
    ok, why = api.status()
    RESULTS["status"] = "on" if ok else f"off: {why}"
    _say(f"status: {RESULTS['status']}")
    if not ok:
        _say("nothing sent. The cloud is off and hidden in the app. To use it: "
             "\"ideogram_cloud_enabled\": true in config.txt, restart the app, then the key "
             "in the Cloud (Ideogram) tab (or env IDEOGRAM_API_KEY).")
        _save()
        return
    img, src = _test_image()
    _say(f"source: {src} {img.width}x{img.height}")
    mask = Image.new("L", img.size, 0)
    w, h = img.size
    mask.paste(255, (w // 4, h // 3, 3 * w // 4, 2 * h // 3))
    small = img.resize((img.width // 2, img.height // 2))
    builds = {
        "edit": lambda: api.build_edit(img, "Make the background deep night blue.",
                                       mask=mask, quality="low"),
        "replace_background": lambda: api.build_replace_background(img, small),
        "remix": lambda: api.build_remix(img, "the same poster as a linocut print"),
        "remove_background": lambda: api.build_remove_background(img),
        "upscale": lambda: api.build_upscale(small, "x2", "topaz-text-refine"),
        "transparent": lambda: api.build_transparent("a red fox mascot logo, flat vector",
                                                     1024, 1024),
        "vectorize": lambda: api.build_vectorize(small),
        "svg": lambda: api.build_svg("a minimal coffee cup icon, two colours"),
    }
    for k, b in builds.items():
        try:
            _q, line = api.quote(b(), output_dir=OUT)
            RESULTS["quotes"][k] = line
            _say(f"quote {k}: {line}")
        except Exception as e:
            RESULTS["quotes"][k] = f"{type(e).__name__}: {e}"
            _say(f"quote {k} FAILED: {e}")
        _save()
    if not spend:
        _say("quotes done; add --spend for the real edit + remove background (billed)")
        _save()
        return
    for k in ("edit", "remove_background"):
        t0 = time.time()
        try:
            req = builds[k]()
            res = api.run(req, output_dir=OUT)
            paths = api.save(req, res, "custom", OUT, "png")
            RESULTS["runs"][k] = {"seconds": round(time.time() - t0, 1),
                                  "result": (f"{len(res.images)} image(s) "
                                             f"{[i.size for i in res.images]}, blocked "
                                             f"{res.blocked}, files "
                                             f"{[os.path.relpath(p, ROOT) for p in paths]}"
                                             + (f", billed {res.cost_usd} USD"
                                                if res.cost_usd is not None else ""))}
            _say(f"{k}: {RESULTS['runs'][k]['result']}")
        except Exception as e:
            RESULTS["runs"][k] = {"error": f"{type(e).__name__}: {e}",
                                  "trace": traceback.format_exc()}
            _say(f"{k} FAILED: {e}")
        _save()
    _say(f"done: {os.path.relpath(os.path.join(OUT, 'report.md'), ROOT)}; cost journal: "
         f"{os.path.relpath(api.journal_path(OUT), ROOT)}")


if __name__ == "__main__":
    main()
