"""Lot 8b: is a ComfyUI single file the same Instant as fal's release? Checked, then rendered.

  1. WEIGHTS   every tensor of the single file, converted to diffusers' layout (fused qkv
               split, attention.o renamed, prefix removed), compared bit for bit with
               fal/ideogram-v4-instant's transformer/ (in the HF cache since lot 8a). One
               tensor in memory at a time: no GPU, ~20 GB read from disk.
  2. RENDER    (--render, GPU, app closed) the same caption and seed through the app with
               fal's repo, then with the single file under the 'instant' profile: time,
               VRAM, and the pixel difference (0 expected when step 1 says identical).

The single file defaults to Hippotes/Ideogram4-Fal-ComfyUI's Ideogram4-instant_bf16
(18.6 GB, downloaded once into the HF cache, no licence gate on that repo; the weights
stay under the Ideogram 4 NON-COMMERCIAL licence). --file PATH takes a local one.
Output: bench/lot8b/report.md, results.json, the two renders.

Usage:  .venv\\Scripts\\python tools\\check_comfy_vs_fal.py [--file PATH] [--render]
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
OUT = os.path.join(ROOT, "bench", "lot8b")
FAL = "fal/ideogram-v4-instant"
HIP_REPO = "Hippotes/Ideogram4-Fal-ComfyUI"
HIP_FILE = "Ideogram4-instant_bf16.safetensors"
RESULTS = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": {}}


def _say(msg):
    print(f"[lot8b] {msg}", flush=True)


def _arg(name, default=None):
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def _save():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False, default=str)
    lines = ["# Lot 8b: ComfyUI single file against fal's release", "",
             f"Started {RESULTS['started']}" + (f", finished {RESULTS['finished']}"
                                                if "finished" in RESULTS else ", NOT finished"),
             ""]
    w = RESULTS["steps"].get("weights")
    if w:
        lines += [f"- file: `{w.get('file')}`",
                  f"- weights: {w.get('equal')} tensors identical, "
                  f"{len(w.get('different') or [])} different, "
                  f"{len(w.get('missing') or [])} missing in fal's folder "
                  f"({w.get('seconds')} s)", ""]
        for name, diff in (w.get("different") or [])[:10]:
            lines.append(f"  - `{name}`: max abs diff {diff}")
    r = RESULTS["steps"].get("render")
    if r:
        lines += ["", "| render | load s | render s | peak GB | image |", "|---|---|---|---|---|"]
        for k in ("fal", "file"):
            x = r.get(k) or {}
            lines.append(f"| {k} | {x.get('load_s', '')} | {x.get('seconds', '')} | "
                         f"{x.get('peak_gb', '')} | {x.get('image') or x.get('error', '')} |")
        lines += ["", f"Pixel difference (mean / max on 0-255): {r.get('mean_diff')} / "
                      f"{r.get('max_diff')}"]
    with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _single_file():
    path = _arg("--file")
    if path:
        return path
    from huggingface_hub import hf_hub_download
    _say(f"downloading {HIP_REPO}/{HIP_FILE} (18.6 GB, once, into the HF cache)")
    return hf_hub_download(HIP_REPO, HIP_FILE)


def weights(path):
    import cz_ideogram as E
    t0 = time.time()
    folder = os.path.join(E.fetch_source(FAL), "transformer")
    _say(f"comparing {os.path.basename(path)} with {folder}")
    res = E.compare_single_file(path, folder)
    res.update(file=path, seconds=round(time.time() - t0, 1))
    RESULTS["steps"]["weights"] = res
    _save()
    _say(f"{res['equal']} identical, {len(res['different'])} different, "
         f"{len(res['missing'])} missing")
    return not res["different"] and not res["missing"]


def render(path):
    import numpy as np
    import torch
    import gpu_lot1_check as G1
    from gpu_lot8_check import caption_texts
    G1.OUT = OUT
    P = G1.prechecks()
    text = caption_texts()["poster_text"]
    out, imgs = {}, {}
    for key, src in (("fal", FAL), ("file", path)):
        P.IDEOGRAM_PROFILE = "instant" if key == "file" else None
        P.set_zimage_transformer(src)
        P.GUIDANCE = 1.0
        t0 = time.time()
        P.get_pipe("txt2img")
        load_s = round(time.time() - t0, 1)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        img = P.generate(text, 1024, 1024, 8, 11)
        p = os.path.join(OUT, f"{key}.png")
        img.save(p)
        imgs[key] = np.asarray(img.convert("RGB"), dtype=np.int16)
        out[key] = {"load_s": load_s, "seconds": round(time.time() - t0, 1),
                    "peak_gb": round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2),
                    "image": os.path.relpath(p, ROOT)}
        _say(f"{key}: loaded in {load_s} s, rendered in {out[key]['seconds']} s")
        RESULTS["steps"]["render"] = out
        _save()
    d = np.abs(imgs["fal"] - imgs["file"])
    out["mean_diff"], out["max_diff"] = round(float(d.mean()), 3), int(d.max())
    _say(f"pixel difference: mean {out['mean_diff']}, max {out['max_diff']}")
    RESULTS["steps"]["render"] = out
    _save()


def main():
    os.makedirs(OUT, exist_ok=True)
    path = _single_file()
    same = weights(path)
    _say("weights IDENTICAL to fal's" if same else "weights DIFFER from fal's: see report.md")
    if "--render" in sys.argv:
        render(path)
    RESULTS["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _save()
    _say(f"done: {os.path.relpath(os.path.join(OUT, 'report.md'), ROOT)}")


if __name__ == "__main__":
    main()
