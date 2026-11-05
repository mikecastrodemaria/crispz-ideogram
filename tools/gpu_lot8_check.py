"""Lot 8a on the GPU: Ideogram 4 Instant (fal, distilled, 8 steps, one transformer)
against the base model, through the app's own code path.

  0. PRECHECKS   the GPU is free (same check as gpu_lot1_check), CUDA, bitsandbytes,
                 access to the gated base repo AND to fal/ideogram-v4-instant, whether
                 the pinned components revision is already in the HF cache.
  1. INSTANT     load (time, VRAM at rest: torch and driver), then 3 fixed JSON captions
                 x 3 fixed seeds at 1024: the first render of the session is reported
                 apart (CUDA warm-up), the others as the steady state.
  2. BASE        the same captions and seeds at 48 steps, guidance 7 (--skip-base to
                 skip it: ~9 minutes on a 5090).
  3. SHEET       contact sheet: one row per caption x seed, Instant | Base side by side.

Every render: time, VRAM peak, Ideogram's grey "blocked" card detector. The captions are
checked against Ideogram's caption schema before anything loads. Output: bench/lot8/
(images, captions, results.json, report.md, sheet.jpg). The times are measures, not
targets: the report says what was observed on this machine.

THE GPU MUST BE FREE: close the app, crispz-flux3 and any other model runtime
(`nvidia-smi` at 0 to 5 % utilisation). Accept the licence of fal/ideogram-v4-instant on
its Hugging Face page first (automatic approval), with the account of your token.

Usage:  .venv\\Scripts\\python tools\\gpu_lot8_check.py [--skip-base] [--seeds 11,22,33]
                                                       [--size 1024] [--offload model]
"""
import json
import os
import subprocess
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
OUT = os.path.join(ROOT, "bench", "lot8")
FAL = "fal/ideogram-v4-instant"
RESULTS = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": {}}

# Three fixed captions: dense text, a logo, a multi-element composition. Minified, keys
# in the schema's order, checked by the vendored verifier before the run.
CAPTIONS = {
    "poster_text": {
        "high_level_description": "A bold typographic event poster for a jazz night, "
        "dense with exact text, flat graphic design on cream paper.",
        "style_description": {
            "aesthetics": "Swiss-style typographic poster, strict grid, high contrast",
            "lighting": "flat even lighting, no shadows",
            "medium": "graphic_design",
            "art_style": "minimal flat vector typography with one geometric accent",
            "color_palette": ["#F2EBDD", "#1A1A1A", "#D7262E"]},
        "compositional_deconstruction": {
            "background": "plain warm cream paper with a faint print texture",
            "elements": [
                {"type": "text", "bbox": [60, 70, 260, 930], "text": "BLUE NOTE NIGHTS",
                 "desc": "huge condensed black sans serif headline, uppercase, tight "
                 "tracking, spanning the full width", "color_palette": ["#1A1A1A"]},
                {"type": "obj", "bbox": [300, 250, 640, 750],
                 "desc": "a large flat red circle, perfectly round, centred",
                 "color_palette": ["#D7262E"]},
                {"type": "text", "bbox": [700, 70, 790, 930],
                 "text": "FRIDAY 14 NOVEMBER 21:00",
                 "desc": "medium black sans serif line, uppercase, left aligned",
                 "color_palette": ["#1A1A1A"]},
                {"type": "text", "bbox": [810, 70, 940, 930],
                 "text": "Live quartet, free entry before 22:00. Le Caveau, 12 rue des "
                 "Arts, Lyon",
                 "desc": "small black sans serif paragraph on two lines, left aligned",
                 "color_palette": ["#1A1A1A"]}]}},
    "logo": {
        "high_level_description": "A clean flat logo for a coffee roaster: a coffee bean "
        "emblem above the brand name, centred on a white background.",
        "style_description": {
            "aesthetics": "modern minimal brand identity, generous negative space",
            "lighting": "flat, no shadows",
            "medium": "graphic_design",
            "art_style": "flat vector logo, two colours",
            "color_palette": ["#FFFFFF", "#3B2314", "#C98B4A"]},
        "compositional_deconstruction": {
            "background": "plain pure white background",
            "elements": [
                {"type": "obj", "bbox": [200, 380, 520, 620],
                 "desc": "a stylised coffee bean emblem, simple geometric shape with a "
                 "curved centre line", "color_palette": ["#3B2314", "#C98B4A"]},
                {"type": "text", "bbox": [580, 200, 720, 800], "text": "BEAN THERE",
                 "desc": "bold rounded sans serif wordmark, uppercase, centred",
                 "color_palette": ["#3B2314"]},
                {"type": "text", "bbox": [740, 300, 800, 700], "text": "COFFEE ROASTERS",
                 "desc": "small spaced-out sans serif tagline, uppercase, centred",
                 "color_palette": ["#C98B4A"]}]}},
    "composition": {
        "high_level_description": "A cosy illustrated bookshop window display with a cat "
        "on a stack of books, a hanging sign and warm evening light.",
        "style_description": {
            "aesthetics": "warm storybook illustration, detailed but readable",
            "lighting": "warm interior light against a blue dusk outside",
            "medium": "illustration",
            "art_style": "gouache-like digital illustration with soft textures",
            "color_palette": ["#2B3A55", "#F2A65A", "#E8D8C3", "#7A4E2D", "#3F6E5A"]},
        "compositional_deconstruction": {
            "background": "a wooden shop front at dusk, shelves of books behind the glass",
            "elements": [
                {"type": "text", "bbox": [40, 280, 160, 720], "text": "LA PAGE QUI DORT",
                 "desc": "hand-painted serif lettering on a hanging wooden sign",
                 "color_palette": ["#E8D8C3", "#7A4E2D"]},
                {"type": "obj", "bbox": [430, 330, 900, 680],
                 "desc": "a tall uneven stack of old hardcover books in muted colours",
                 "color_palette": ["#3F6E5A", "#7A4E2D", "#E8D8C3"]},
                {"type": "obj", "bbox": [280, 380, 470, 620],
                 "desc": "a ginger cat curled up asleep on top of the book stack",
                 "color_palette": ["#F2A65A"]},
                {"type": "obj", "bbox": [200, 760, 520, 960],
                 "desc": "a small lit table lamp with a fabric shade",
                 "color_palette": ["#F2A65A", "#E8D8C3"]}]}},
}


def _say(msg):
    print(f"[lot8] {msg}", flush=True)


def _arg(name, default=None, cast=str):
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            return cast(sys.argv[i + 1])
    return default


def _gb(n):
    return round(n / 1024 ** 3, 2)


def _driver_gb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return round(float(out.strip().splitlines()[0]) / 1024, 2)
    except Exception:
        return None


def _save():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False, default=str)
    _report()


def caption_texts():
    """{name: minified JSON} after the schema check (raises on any warning)."""
    from third_party.ideogram4.caption_verifier import CaptionVerifier
    out = {}
    for k, c in CAPTIONS.items():
        raw = json.dumps(c, ensure_ascii=False, separators=(",", ":"))
        warn = CaptionVerifier().verify_raw(raw)
        if warn:
            raise SystemExit(f"caption {k} fails the schema: {warn}")
        out[k] = raw
    return out


def check_fal_access(token):
    """'OK' or the refusal, without downloading the weights (one small file)."""
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(FAL, "transformer/config.json", token=token)
        return "OK"
    except Exception as e:
        return f"REFUSED: {type(e).__name__}: {e}"


def components_in_cache():
    """Whether the pinned revision of the base repo is already in the HF cache."""
    import cz_ideogram as E
    try:
        from huggingface_hub import scan_cache_dir
        for repo in scan_cache_dir().repos:
            if repo.repo_id == E.COMPONENTS_REPO:
                revs = sorted(r.commit_hash for r in repo.revisions)
                return {"cached_revisions": revs,
                        "pinned_present": E.COMPONENTS_REVISION in revs}
        return {"cached_revisions": [], "pinned_present": False}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def render(P, tag, text, size, steps, seed):
    import torch
    from gpu_lot2_check import is_blocked_card
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    img = P.generate(text, size, size, steps, seed)
    torch.cuda.synchronize()
    dt = time.time() - t0
    path = os.path.join(OUT, f"{tag}.png")
    img.save(path)
    r = {"seconds": round(dt, 1), "s_per_step": round(dt / steps, 2),
         "peak_reserved_gb": _gb(torch.cuda.max_memory_reserved()),
         "blocked_card": is_blocked_card(img), "image": os.path.relpath(path, ROOT)}
    _say(f"{tag}: {dt:.1f} s ({r['s_per_step']} s/step), peak {r['peak_reserved_gb']} GB, "
         f"blocked card: {r['blocked_card']}")
    return img, r


def run_model(P, label, transformer, steps, guidance, texts, seeds, size):
    import torch
    P.set_zimage_transformer(transformer)
    P.GUIDANCE = guidance
    t0 = time.time()
    pipe = P.get_pipe("txt2img")
    load = {"seconds": round(time.time() - t0, 1),
            "profile": pipe.profile.as_meta(),
            "rest_allocated_gb": _gb(torch.cuda.memory_allocated()),
            "rest_reserved_gb": _gb(torch.cuda.memory_reserved()),
            "rest_driver_gb": _driver_gb()}
    _say(f"{label} loaded in {load['seconds']} s, profile {load['profile']['profile']}, "
         f"rest {load['rest_allocated_gb']} GB allocated / {load['rest_driver_gb']} GB driver")
    res = {"load": load, "renders": {}}
    RESULTS["steps"][label] = res
    _save()
    imgs = {}
    first = True
    for name, text in texts.items():
        for seed in seeds:
            tag = f"{label}_{name}_s{seed}"
            try:
                img, r = render(P, tag, text, size, steps, seed)
                r["first_of_session"] = first
                first = False
                imgs[(name, seed)] = img
            except Exception as e:
                r = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}
                _say(f"{tag} FAILED: {r['error']}")
            res["renders"][tag] = r
            _save()
    warm = [r["seconds"] for r in res["renders"].values()
            if "seconds" in r and not r.get("first_of_session")]
    firsts = [r["seconds"] for r in res["renders"].values() if r.get("first_of_session")]
    res["first_render_s"] = firsts[0] if firsts else None
    res["warm_mean_s"] = round(sum(warm) / len(warm), 1) if warm else None
    res["peak_max_gb"] = max((r.get("peak_reserved_gb") or 0)
                             for r in res["renders"].values()) or None
    _save()
    return imgs


def sheet(rows, cols, cell=384):
    from PIL import Image, ImageDraw
    keys = sorted({k for c in cols.values() for k in c})
    if not keys:
        return None
    pad = 28
    W = len(cols) * cell
    H = len(keys) * (cell + pad) + pad
    out = Image.new("RGB", (W, H), (245, 245, 245))
    d = ImageDraw.Draw(out)
    for j, label in enumerate(cols):
        d.text((j * cell + 8, 6), label, fill=(0, 0, 0))
    for i, k in enumerate(keys):
        y = pad + i * (cell + pad)
        d.text((8, y - pad + 14), f"{k[0]}  seed {k[1]}", fill=(80, 80, 80))
        for j, label in enumerate(cols):
            img = cols[label].get(k)
            if img is not None:
                out.paste(img.convert("RGB").resize((cell, cell)), (j * cell, y))
    return out


def _report():
    lines = ["# Lot 8a on the GPU: Ideogram 4 Instant (fal) against the base model", "",
             f"Started {RESULTS['started']}" + (f", finished {RESULTS['finished']}"
                                                if "finished" in RESULTS else ", NOT finished"),
             ""]
    pre = RESULTS["steps"].get("0_prechecks") or {}
    if pre:
        lines += [f"- GPU {pre.get('gpu')}, torch {pre.get('torch')}, diffusers "
                  f"{pre.get('diffusers')}, bitsandbytes {pre.get('bitsandbytes')}",
                  f"- access to {FAL}: {pre.get('fal_access')}",
                  f"- pinned components revision in cache: {pre.get('components')}", ""]
    lines += ["| model | load s | rest alloc GB | rest driver GB | first render s | "
              "warm mean s | max peak GB |", "|---|---|---|---|---|---|---|"]
    for label in ("instant", "base"):
        r = RESULTS["steps"].get(label)
        if not r:
            continue
        ld = r["load"]
        lines.append(f"| {label} | {ld['seconds']} | {ld['rest_allocated_gb']} | "
                     f"{ld['rest_driver_gb']} | {r.get('first_render_s')} | "
                     f"{r.get('warm_mean_s')} | {r.get('peak_max_gb')} |")
    lines += ["", "| render | s | s/step | peak GB | blocked | image / error |",
              "|---|---|---|---|---|---|"]
    for label in ("instant", "base"):
        r = RESULTS["steps"].get(label) or {}
        for tag, x in (r.get("renders") or {}).items():
            lines.append(f"| {tag} | {x.get('seconds', '')} | {x.get('s_per_step', '')} | "
                         f"{x.get('peak_reserved_gb', '')} | {x.get('blocked_card', '')} | "
                         f"{x.get('image') or x.get('error', '')} |")
    with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    import gpu_lot1_check as G1
    G1.OUT = OUT
    os.makedirs(OUT, exist_ok=True)
    texts = caption_texts()
    for k, t in texts.items():
        with open(os.path.join(OUT, f"caption_{k}.json"), "w", encoding="utf-8") as f:
            f.write(t)
    seeds = [int(s) for s in str(_arg("--seeds", "11,22,33")).split(",") if s.strip()]
    size = _arg("--size", 1024, int)
    P = G1.prechecks()
    pre = dict(G1.RESULTS["steps"].get("0_prechecks") or {})
    pre["fal_access"] = check_fal_access(P._hf_token() or None)
    pre["components"] = components_in_cache()
    RESULTS["steps"]["0_prechecks"] = pre
    _save()
    if not pre["fal_access"].startswith("OK"):
        _say(f"No access to {FAL}: {pre['fal_access']}")
        _say(f"  Accept its licence on https://huggingface.co/{FAL} (automatic approval) "
             "with the account of your token, then run again.")
        sys.exit(2)
    if not (pre["components"] or {}).get("pinned_present"):
        _say("the pinned components revision is not in the cache yet: the first load "
             "downloads what differs from your cached revision (tokenizer, encoder, VAE).")
    off = _arg("--offload")
    if off:
        P.set_offload_mode(off)
    import cz_ideogram as E
    cols = {}
    inst = E.PROFILES["instant"]
    cols["Instant (8 steps)"] = run_model(P, "instant", FAL, inst.steps, inst.guidance,
                                          texts, seeds, size)
    if "--skip-base" not in sys.argv:
        cols["Base (48 steps)"] = run_model(P, "base", None, 48, 7.0, texts, seeds, size)
    sh = sheet(None, cols)
    if sh is not None:
        sh.save(os.path.join(OUT, "sheet.jpg"), quality=90)
        RESULTS["sheet"] = os.path.relpath(os.path.join(OUT, "sheet.jpg"), ROOT)
    RESULTS["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _save()
    _say(f"done: {os.path.relpath(os.path.join(OUT, 'report.md'), ROOT)}")


if __name__ == "__main__":
    main()
