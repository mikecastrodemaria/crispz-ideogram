"""Lot 1 validation on the GPU: Ideogram 4.0 through the app's own code path.

What it measures and checks, in this order (each step is written to the report as soon
as it is done, so a crash still leaves the earlier figures):

  0. PRECHECKS   no other process on the GPU (crispz-flux3 open = abort), CUDA, the
                 bitsandbytes build, access to the GATED repo with your token.
  1. LOADING     cz_pipeline.get_pipe("txt2img") from an empty state: time, VRAM at rest
                 (torch allocated / reserved, and the driver's figure, which also counts
                 the CUDA context and bitsandbytes' own buffers).
  2. 1024x1024   48 steps, twice: the first image pays the CUDA warm-up, the second one
                 is the steady state. Time, s/step, VRAM peak. Then once more with the
                 fast attention OFF (raw pipeline call): speed-up and pixel difference.
  3. 2048x2048   48 steps, once (native 2K). Same figures.
  4. STOP        a 1024 render stopped from another thread through request_stop(), as
                 the UI's Stop button does: delay before the call returns, and an image
                 must still come back.
  5. PREVIEW     the live-preview frames of the 1024 render, with the corrected
                 Ideogram projection AND the FLUX.2 matrix as is (the old behaviour), each
                 compared to the final image: mean error on a 0-255 scale.

Prompt: a poster with text ("GRAND OPENING"), so the images also say whether the text
renders. Everything lands in bench/lot1/ (images, frames, results.json, report.md).

THE GPU MUST BE FREE: close crispz-flux3 (and any other crispz) first. An app that has
not rendered yet shows nothing in nvidia-smi and still loads its model at the first
click: closing it is the only safe way (incident of 2026-10-28, FORK-flux3-CLAUDE.md §3).

Usage (Windows, from the repo root):
    .venv\\Scripts\\python tools\\gpu_lot1_check.py
    .venv\\Scripts\\python tools\\gpu_lot1_check.py --offload model --skip-2048

  --offload MODE   none | model | sequential | auto (default: the app's setting)
  --steps N        48 by default (the official schedule)
  --seed N         12345 by default
  --skip-2048      skips the 2048x2048 render
  --skip-raw       skips the 1024 render without the fast attention (A/B check)
  --only-2048      loads, then the 2048 render alone (denoise vs VAE decode peak)
  --force          runs even if another process holds the GPU (you have been warned)
"""
import json
import os
import subprocess
import sys
import threading
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "bench", "lot1")

PROMPT = ('A bold retro poster for a bakery opening. Large headline text "GRAND OPENING" '
          'at the top in thick cream serif letters, below it smaller text "SATURDAY 9 AM" '
          'in a red ribbon banner, a basket of golden croissants in the center, warm '
          'orange and teal palette, clean flat vector illustration style.')

RESULTS = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "prompt": PROMPT, "steps": {}}


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
    _write_report()


def _say(msg):
    print(f"[lot1] {msg}", flush=True)


def _gb(n):
    return round(n / 1024 ** 3, 2)


def _smi_used_gb():
    """The driver's view of the VRAM in use (all processes), in GB, or None."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        used, total = (float(x) for x in out.strip().splitlines()[0].split(","))
        return round(used / 1024, 2), round(total / 1024, 2)
    except Exception:
        return None, None


def _gpu_processes():
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                              "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout
        return [ln.strip() for ln in out.splitlines() if ln.strip()]
    except Exception as e:
        return [f"(nvidia-smi unavailable: {e})"]


# On Windows (WDDM) nvidia-smi lists EVERY process with a GPU context (explorer, the
# browsers, the Start menu...) with "[N/A]" for the memory: listing them all as blockers
# made the check useless (36 processes on Mike's desktop, none a model). What matters:
# a process that runs models (python, ComfyUI, Ollama...), a process using more than
# 1 GB when the driver tells, or more than BUSY_GB in use on the card overall.
BUSY_GB = 4.0
_MODEL_RUNTIMES = ("python", "comfy", "ollama", "crispz", "pinokio", "llama", "koboldcpp",
                   "lmstudio", "lm studio", "invokeai", "forge", "fooocus")
# A process whose name is hidden ("[Insufficient Permissions]", typically dwm.exe) is not
# a suspect by itself: the overall BUSY_GB check covers a hidden one that holds a model.


def _suspects(procs):
    out = []
    for p in procs:
        low = p.lower()
        mem = p.rsplit(",", 1)[-1].strip().lower()
        heavy = False
        if mem.endswith("mib"):
            try:
                heavy = float(mem[:-3]) > 1024
            except ValueError:
                pass
        if heavy or any(k in low for k in _MODEL_RUNTIMES):
            out.append(p)
    return out


# --- 0. Prechecks -----------------------------------------------------------------------
def prechecks():
    import torch
    r = {}
    procs = _gpu_processes()
    r["gpu_processes"] = procs
    suspects = _suspects(procs)
    used, total = _smi_used_gb()
    r["driver_used_at_start_gb"] = used
    busy = used is not None and used > BUSY_GB
    if (suspects or busy) and "--force" not in sys.argv:
        if suspects:
            _say("A MODEL PROCESS MAY HOLD THE GPU:")
            for p in suspects:
                _say(f"   {p}")
        if busy:
            _say(f"{used} GB of VRAM already in use (more than {BUSY_GB} GB): something "
                 "has a model loaded.")
        _say("Close crispz-flux3 (and any other crispz / ComfyUI / Ollama), then run "
             "again. --force skips this check.")
        sys.exit(2)
    _say(f"GPU free enough: {used} GB in use at start, {len(procs)} process(es) listed, "
         "none of them a model runtime")
    if not torch.cuda.is_available():
        _say("CUDA is not available to this torch build: nothing to measure.")
        sys.exit(2)
    r["gpu"] = torch.cuda.get_device_name(0)
    r["torch"] = torch.__version__
    r["cuda"] = torch.version.cuda
    try:
        import bitsandbytes as bnb
        r["bitsandbytes"] = bnb.__version__
    except Exception as e:
        r["bitsandbytes"] = f"IMPORT FAILED: {e}"
        _say(f"bitsandbytes does not import ({e}): the nf4 weights cannot load. "
             "pip install \"bitsandbytes>=0.45\" in this .venv.")
        RESULTS["steps"]["0_prechecks"] = r
        _save()
        sys.exit(2)
    import diffusers
    import transformers
    r["diffusers"] = diffusers.__version__
    r["transformers"] = transformers.__version__
    r["sdpa_kernels"] = _sdpa_kernels()
    _say(f"SDPA kernels usable at head_dim 256 (Ideogram 4): {r['sdpa_kernels']}")
    import cz_pipeline as P
    try:
        with open(P._hw_profile_path(), encoding="utf-8") as f:
            prof = json.load(f)
        r["hw_profile"] = prof
        if "downgrade" in json.dumps(prof).lower():
            _say(f"NOTE: {P._hw_profile_path()} holds a cached offload downgrade: 'auto' "
                 "will start in 'model'. Delete the file (or pass --offload) to decide "
                 "afresh.")
    except FileNotFoundError:
        r["hw_profile"] = None
    except Exception as e:
        r["hw_profile"] = f"unreadable: {e}"
    repo = P.BASE_REPO
    r["repo"] = repo
    token = P._hf_token()
    src = "hf_token (config) or HF_TOKEN (env)" if token else None
    if not token:
        try:
            from huggingface_hub import get_token
            token = get_token()
            src = "huggingface_hub login (hf auth login)" if token else None
        except Exception:
            pass
    r["hf_token_set"] = bool(token)
    r["hf_token_source"] = src or "none"
    if not os.path.isdir(repo):
        try:
            from huggingface_hub import auth_check
            auth_check(repo, token=token)
            r["gated_access"] = "OK"
        except Exception as e:
            r["gated_access"] = f"REFUSED: {type(e).__name__}: {e}"
            _say(f"No access to {repo}: {type(e).__name__}.")
            if not token:
                _say("  NO TOKEN FOUND: neither hf_token in config.txt, nor HF_TOKEN in the "
                     "environment, nor an `hf auth login`.")
            else:
                _say(f"  A token was sent (from {src}) and refused: the licence is not "
                     f"accepted on THIS repo with that account, or the token is a "
                     "fine-grained one without 'read access to public gated repos'.")
            _say(f"  Licence: https://huggingface.co/{repo} (the -diffusers repo itself).")
            RESULTS["steps"]["0_prechecks"] = r
            _save()
            sys.exit(2)
    RESULTS["steps"]["0_prechecks"] = r
    _save()
    _say(f"prechecks OK: {r['gpu']}, torch {r['torch']} (CUDA {r['cuda']}), "
         f"bitsandbytes {r['bitsandbytes']}, repo {repo}")
    return P


def _sdpa_kernels():
    """Which SDPA kernels accept Ideogram 4's attention shape (bf16, head_dim 256), with
    and without a boolean mask. Without flash or efficient, attention falls back to the
    math kernel, whose memory grows with the square of the sequence."""
    import torch
    import torch.nn.functional as F
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except Exception as e:
        return f"unknown ({e})"
    try:
        q = torch.randn(1, 18, 512, 256, device="cuda", dtype=torch.bfloat16)
        mask = torch.ones(1, 1, 512, 512, device="cuda", dtype=torch.bool)
    except Exception as e:
        return f"unknown ({type(e).__name__})"
    out = {}
    for name, be in (("flash", SDPBackend.FLASH_ATTENTION),
                     ("efficient", SDPBackend.EFFICIENT_ATTENTION),
                     ("cudnn", getattr(SDPBackend, "CUDNN_ATTENTION", None))):
        if be is None:
            continue
        for tag, m in (("", None), ("+mask", mask)):
            try:
                with sdpa_kernel(be):
                    F.scaled_dot_product_attention(q, q, q, attn_mask=m)
                out[name + tag] = "yes"
            except Exception:
                out[name + tag] = "no"
    del q, mask
    torch.cuda.empty_cache()
    return out


# --- Frames of the live preview: the app's callback, with a recorder on top --------------
class FrameRecorder:
    """Wraps cz_pipeline._store_preview: the app's own preview still runs; on top, at a
    few steps, the frame is kept with BOTH projections (Ideogram, and FLUX.2 as is)."""

    def __init__(self, P, keep_steps):
        self.P = P
        self.keep = set(keep_steps)
        self.frames = {}
        self.orig = P._store_preview

    def __enter__(self):
        P = self.P
        orig = self.orig

        def rec(latents, step, total, width=0, height=0, pipe=None):
            orig(latents, step, total, width, height, pipe=pipe)
            if step in self.keep or step == total:
                try:
                    fixed = P.latent_preview_image(latents, width, height,
                                                   proj=P._preview_proj_for(pipe))
                    old = P.latent_preview_image(latents, width, height)
                    self.frames[step] = (fixed, old, P._preview_proj_for(pipe) is not None)
                except Exception as e:
                    self.frames[step] = (None, None, f"error: {e}")
        P._store_preview = rec
        P.preview_begin()
        return self

    def __exit__(self, *a):
        self.P._store_preview = self.orig
        self.P.preview_end()


def _mae(preview, final):
    import numpy as np
    from PIL import Image
    w, h = preview.size
    a = np.asarray(preview.convert("RGB"), dtype=np.float32)
    b = np.asarray(final.convert("RGB").resize((w, h), Image.BOX), dtype=np.float32)
    return round(float(np.abs(a - b).mean()), 1)


# --- Renders -----------------------------------------------------------------------------
def render(P, tag, size, steps, seed):
    import torch
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    # Splits the peak: denoise (everything before the VAE decode) vs the decode itself.
    split = {}
    vae = getattr(getattr(P._BASE_PIPE, "_pipe", P._BASE_PIPE), "vae", None)
    real_decode = getattr(vae, "decode", None)
    if real_decode is not None:
        def decode(*a, **k):
            torch.cuda.synchronize()
            split["denoise_peak_reserved_gb"] = _gb(torch.cuda.max_memory_reserved())
            split["before_decode_alloc_gb"] = _gb(torch.cuda.memory_allocated())
            torch.cuda.reset_peak_memory_stats()
            t = time.time()
            out = real_decode(*a, **k)
            torch.cuda.synchronize()
            split["decode_peak_reserved_gb"] = _gb(torch.cuda.max_memory_reserved())
            split["decode_peak_alloc_gb"] = _gb(torch.cuda.max_memory_allocated())
            split["decode_seconds"] = round(time.time() - t, 1)
            return out
        vae.decode = decode
    t0 = time.time()
    try:
        img = P.generate(PROMPT, size, size, steps, seed)
    finally:
        if real_decode is not None:
            del vae.decode
    torch.cuda.synchronize()
    import cz_ideogram
    dt = time.time() - t0
    path = os.path.join(OUT, f"{tag}.png")
    img.save(path)
    used, total = _smi_used_gb()
    r = {"size": f"{img.width}x{img.height}", "steps": steps, "seconds": round(dt, 1),
         "s_per_step": round(dt / steps, 3),
         "peak_alloc_gb": _gb(torch.cuda.max_memory_allocated()),
         "peak_reserved_gb": max(_gb(torch.cuda.max_memory_reserved()),
                                 split.get("denoise_peak_reserved_gb", 0)),
         "driver_used_after_gb": used, "driver_total_gb": total,
         "offload": P._effective_offload(), "fast_attention": cz_ideogram.FAST_ATTENTION,
         "image": os.path.relpath(path, ROOT), **split}
    _say(f"{tag}: {r['size']} in {r['seconds']} s ({r['s_per_step']} s/step), "
         f"peak {r['peak_reserved_gb']} GB reserved")
    if split:
        _say(f"   denoise peak {split.get('denoise_peak_reserved_gb')} GB, VAE decode peak "
             f"{split.get('decode_peak_reserved_gb')} GB in {split.get('decode_seconds')} s")
    return img, r


def stop_test(P, steps, seed, stop_after_s):
    """Stops a 1024 render from another thread, like the UI's Stop button."""
    box = {}

    def run():
        try:
            box["img"] = P.generate(PROMPT, 1024, 1024, steps, seed)
        except Exception as e:
            box["err"] = f"{type(e).__name__}: {e}"
    th = threading.Thread(target=run, daemon=True)
    P.preview_begin()        # the frames tell at which step the denoise really stopped
    t0 = time.time()
    th.start()
    time.sleep(stop_after_s)
    t_stop = time.time()
    P.request_stop()
    th.join(timeout=300)
    t_end = time.time()
    P._STOP = False
    snap = P.preview_snapshot()
    P.preview_end()
    r = {"stop_after_s": stop_after_s, "returned_after_stop_s": round(t_end - t_stop, 1),
         "total_s": round(t_end - t0, 1), "thread_finished": not th.is_alive(),
         "image_returned": "img" in box, "error": box.get("err"),
         "last_preview_step": snap.get("step")}
    if "img" in box:
        p = os.path.join(OUT, "stop_partial.png")
        box["img"].save(p)
        r["image"] = os.path.relpath(p, ROOT)
    ok = (r["image_returned"] and r["returned_after_stop_s"] < 15
          and (r["last_preview_step"] or 0) < steps)
    r["verdict"] = "OK" if ok else "CHECK"
    _say(f"stop: returned {r['returned_after_stop_s']} s after the request, image "
         f"{'yes' if r['image_returned'] else 'NO'} -> {r['verdict']}")
    return r


def main():
    steps = _arg("--steps", 48, int)
    seed = _arg("--seed", 12345, int)
    os.makedirs(OUT, exist_ok=True)
    P = prechecks()
    import torch

    off = _arg("--offload")
    if off:
        P.set_offload_mode(off)
    P.LIVE_PREVIEW_ENABLED = True

    # 1. Loading
    try:
        P.free_vram()
        torch.cuda.synchronize()
        base_used, _ = _smi_used_gb()
        t0 = time.time()
        P.get_pipe("txt2img")
        torch.cuda.synchronize()
        dt = time.time() - t0
        used, total = _smi_used_gb()
        r = {"seconds": round(dt, 1), "offload": P._effective_offload(),
             "offload_asked": P.OFFLOAD_MODE,
             "alloc_gb": _gb(torch.cuda.memory_allocated()),
             "reserved_gb": _gb(torch.cuda.memory_reserved()),
             "driver_used_before_gb": base_used, "driver_used_gb": used,
             "driver_total_gb": total,
             "note": "the first run includes the download of ~16 GB"}
        RESULTS["steps"]["1_load"] = r
        _save()
        _say(f"loaded in {r['seconds']} s, offload {r['offload']}, at rest "
             f"{r['alloc_gb']} GB allocated / driver {used} GB of {total}")
    except Exception as e:
        RESULTS["steps"]["1_load"] = {"error": f"{type(e).__name__}: {e}",
                                      "trace": traceback.format_exc()}
        _save()
        _say(f"LOADING FAILED: {type(e).__name__}: {e}")
        sys.exit(1)

    # 2. 1024, cold then warm, with the preview frames on the warm one
    only2048 = "--only-2048" in sys.argv
    keep = {1, steps // 4, steps // 2, (3 * steps) // 4, steps}
    for tag, rec in (() if only2048 else (("gen1024_cold", False), ("gen1024_warm", True))):
        try:
            if rec:
                with FrameRecorder(P, keep) as fr:
                    img, r = render(P, tag, 1024, steps, seed)
                pv = {}
                for st, (fixed, old, used_ide) in sorted(fr.frames.items()):
                    if fixed is None:
                        pv[str(st)] = {"error": used_ide}
                        continue
                    fixed.save(os.path.join(OUT, f"preview_step{st:02d}_ideogram.png"))
                    old.save(os.path.join(OUT, f"preview_step{st:02d}_flux_matrix.png"))
                    pv[str(st)] = {"ideogram_projection_used": used_ide,
                                   "mae_ideogram": _mae(fixed, img),
                                   "mae_flux_matrix": _mae(old, img)}
                last = pv.get(str(steps), {})
                pv["verdict"] = ("OK" if last.get("mae_ideogram") is not None
                                 and last["mae_ideogram"] < last["mae_flux_matrix"]
                                 and last["mae_ideogram"] < 40 else "CHECK")
                RESULTS["steps"]["5_preview"] = pv
                _say(f"preview, last step: error {last.get('mae_ideogram')} (Ideogram "
                     f"projection) vs {last.get('mae_flux_matrix')} (FLUX matrix as is), "
                     f"0-255 -> {pv['verdict']}")
            else:
                img, r = render(P, tag, 1024, steps, seed)
            RESULTS["steps"][f"2_{tag}"] = r
        except Exception as e:
            RESULTS["steps"][f"2_{tag}"] = {"error": f"{type(e).__name__}: {e}",
                                            "trace": traceback.format_exc()}
            _say(f"{tag} FAILED: {type(e).__name__}: {e}")
        _save()

    # 2b. The same warm 1024 WITHOUT the fast attention (the raw pipeline call): the
    # speed-up, and the proof that the image does not change.
    if "--skip-raw" not in sys.argv and not only2048:
        import cz_ideogram
        import numpy as np
        from PIL import Image
        try:
            cz_ideogram.set_fast_attention(False)
            raw, r = render(P, "gen1024_raw_attention", 1024, steps, seed)
            fast_path = os.path.join(OUT, "gen1024_warm.png")
            if os.path.isfile(fast_path):
                a = np.asarray(Image.open(fast_path).convert("RGB"), dtype=np.int16)
                b = np.asarray(raw.convert("RGB"), dtype=np.int16)
                d = np.abs(a - b)
                r["vs_fast_mean_abs_diff"] = round(float(d.mean()), 3)
                r["vs_fast_max_abs_diff"] = int(d.max())
                warm = RESULTS["steps"].get("2_gen1024_warm", {}).get("seconds")
                if warm:
                    r["speedup_fast_vs_raw"] = round(r["seconds"] / warm, 2)
                _say(f"raw vs fast attention: mean pixel diff {r['vs_fast_mean_abs_diff']}, "
                     f"speed-up x{r.get('speedup_fast_vs_raw')}")
            RESULTS["steps"]["2_gen1024_raw_attention"] = r
        except Exception as e:
            RESULTS["steps"]["2_gen1024_raw_attention"] = {"error": f"{type(e).__name__}: {e}"}
            _say(f"raw attention render FAILED: {type(e).__name__}: {e}")
        finally:
            cz_ideogram.set_fast_attention(True)
        _save()

    # 3. 2048
    if "--skip-2048" not in sys.argv:
        try:
            _, r = render(P, "gen2048", 2048, steps, seed)
            RESULTS["steps"]["3_gen2048"] = r
        except Exception as e:
            RESULTS["steps"]["3_gen2048"] = {"error": f"{type(e).__name__}: {e}",
                                             "trace": traceback.format_exc()}
            _say(f"gen2048 FAILED: {type(e).__name__}: {e}")
        _save()

    # 4. Stop, about a third of the way into a warm 1024 render
    warm = RESULTS["steps"].get("2_gen1024_warm", {}).get("seconds") or 30
    if not only2048:
        try:
            RESULTS["steps"]["4_stop"] = stop_test(P, steps, seed + 1, max(2.0, warm / 3))
        except Exception as e:
            RESULTS["steps"]["4_stop"] = {"error": f"{type(e).__name__}: {e}"}
    RESULTS["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _save()
    _say(f"done: {os.path.relpath(os.path.join(OUT, 'report.md'), ROOT)}")


def _write_report():
    s = RESULTS["steps"]
    L = ["# Lot 1 on the GPU: Ideogram 4.0 (nf4)", "",
         f"Started {RESULTS['started']}" + (f", finished {RESULTS['finished']}"
                                            if "finished" in RESULTS else ", NOT finished"),
         ""]
    pre = s.get("0_prechecks", {})
    if pre:
        L += [f"- GPU: {pre.get('gpu')}, torch {pre.get('torch')} (CUDA {pre.get('cuda')})",
              f"- diffusers {pre.get('diffusers')}, transformers {pre.get('transformers')}, "
              f"bitsandbytes {pre.get('bitsandbytes')}",
              f"- repo {pre.get('repo')}: access {pre.get('gated_access', 'local folder')}", ""]
    ld = s.get("1_load", {})
    if ld:
        L += ["## Loading", "", "| | value |", "|---|---|"]
        for k in ("seconds", "offload_asked", "offload", "alloc_gb", "reserved_gb", "driver_used_before_gb",
                  "driver_used_gb", "error"):
            if k in ld:
                L.append(f"| {k} | {ld[k]} |")
        L.append("")
    rows = [(k, v) for k, v in s.items() if k.startswith(("2_", "3_"))]
    if rows:
        L += ["## Renders", "", "| render | size | s | s/step | peak alloc GB | "
              "peak reserved GB | driver after GB | offload |", "|---|---|---|---|---|---|---|---|"]
        for k, v in rows:
            if "error" in v:
                L.append(f"| {k} | ERROR: {v['error']} | | | | | | |")
            else:
                L.append(f"| {k} | {v['size']} | {v['seconds']} | {v['s_per_step']} | "
                         f"{v['peak_alloc_gb']} | {v['peak_reserved_gb']} | "
                         f"{v['driver_used_after_gb']} | {v['offload']} |")
        for k, v in rows:
            if "decode_peak_reserved_gb" in v:
                L.append(f"- {k}: denoise peak {v.get('denoise_peak_reserved_gb')} GB, VAE "
                         f"decode peak {v['decode_peak_reserved_gb']} GB (alloc "
                         f"{v.get('decode_peak_alloc_gb')}) in {v.get('decode_seconds')} s")
        raw = s.get("2_gen1024_raw_attention", {})
        if "vs_fast_mean_abs_diff" in raw:
            L.append(f"Fast attention vs raw call (1024, same seed): speed-up "
                     f"x{raw.get('speedup_fast_vs_raw')}, pixel difference mean "
                     f"{raw['vs_fast_mean_abs_diff']} / max {raw['vs_fast_max_abs_diff']} (0-255).")
        if pre.get("sdpa_kernels"):
            L.append(f"SDPA kernels at head_dim 256: {pre['sdpa_kernels']}")
        L.append("")
    st = s.get("4_stop")
    if st:
        L += ["## Stop", "", "```", json.dumps(st, indent=1), "```", ""]
    pv = s.get("5_preview")
    if pv:
        L += ["## Live preview (mean error against the final image, 0-255)", "",
              "| step | Ideogram projection | FLUX matrix as is |", "|---|---|---|"]
        for k, v in pv.items():
            if k != "verdict" and isinstance(v, dict) and "mae_ideogram" in v:
                L.append(f"| {k} | {v['mae_ideogram']} | {v['mae_flux_matrix']} |")
        L += ["", f"Verdict: {pv.get('verdict')}", ""]
    with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
