"""The FLUX.3 port on the real GPU: the protocol v1 paths through the ONE Flux3Pipe.

    gen      -> generate()        txt2img, 16 steps (distilled)
    edit x1  -> generate_omni()   one reference, cited as <ref_image_0>
    edit x2  -> generate_omni()   two references, <ref_image_0> and <ref_image_1>
    inpaint  -> inpaint_run()     per-step latent blending inside the adapter
    upscale  -> _refine_whole()   img2img, SDEdit on the flow path (strength 0.35)

A GPU end-to-end run, not a unit test: it loads the real checkpoint (gated: accept the
licence on https://huggingface.co/black-forest-labs/flux-3-image-klein and log in) and
renders five 1024x1024 images. It SKIPS cleanly without CUDA, without access to the repo,
or when CZ_E2E=0. The images land in tests/e2e_*.png (local only) and the timings and
VRAM peaks are printed for FORK.md section 7.

    FLUX3_E2E_MODEL       another base repo (default: the distilled one)
    FLUX3_E2E_PRECISION   bf16 (default) / fp8r / nvfp4
    CZ_E2E=0              skip even with a GPU (a CI runner with a card)

Run:  .venv/Scripts/python -m pytest tests/test_flux3_e2e.py -s
  or  .venv/Scripts/python tests/test_flux3_e2e.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

REPO = os.environ.get("FLUX3_E2E_MODEL") or "black-forest-labs/flux-3-image-klein"
# The base repo and the precision are PINNED before cz_pipeline is imported: without that
# the run follows the last model chosen in the UI (preferences.json) and could pull the
# -base repo or an nvfp4 build the card cannot run, for a reason foreign to the code.
os.environ["FLUX3_MODEL"] = REPO
os.environ["FLUX3_PRECISION"] = os.environ.get("FLUX3_E2E_PRECISION") or "bf16"
os.environ.pop("FLUX3_TRANSFORMER", None)

OUT = os.path.dirname(os.path.abspath(__file__))
P1 = ("a comic book panel, a woman with red hair running in the rain at night, "
      "ink lines, flat colors, dramatic angle")


def _why_skip():
    if os.environ.get("CZ_E2E", "1") == "0":
        return "CZ_E2E=0"
    import torch
    if not torch.cuda.is_available():
        return "needs CUDA (the real checkpoint is 5.6B parameters)"
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(REPO, "modular_model_index.json")
    except Exception as e:
        return f"no access to {REPO}: {type(e).__name__} (accept the licence + token)"
    return None


def _vram():
    import torch
    return torch.cuda.memory_allocated() / 1024 ** 3


def _peak():
    import torch
    return torch.cuda.max_memory_reserved() / 1024 ** 3


def _step(name, fn, log):
    import torch
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    out = fn()
    line = f"  {name:<24} {time.time() - t:6.2f}s   vram={_vram():5.2f} GB  peak={_peak():5.2f} GB"
    print(line, flush=True)
    log.append(line)
    return out


@pytest.mark.skipif(_why_skip() is not None, reason=_why_skip() or "")
def test_flux3_end_to_end():
    from PIL import Image, ImageDraw
    import cz_flux3
    import cz_pipeline as p
    log = []
    print(f"\nbase repo: {p.BASE_REPO}  precision: {p.PRECISION}  offload: {p.OFFLOAD_MODE}")

    print("\n[0] load")
    pipe = _step("load (_ensure_base)", p._ensure_base, log)
    assert isinstance(pipe, cz_flux3.Flux3Pipe), type(pipe).__name__
    assert p.get_pipe("img2img") is pipe and p.get_pipe("inpaint") is pipe

    steps = cz_flux3.DEFAULTS[cz_flux3.variant_of(p.BASE_REPO)]["steps"]
    print(f"\n[1] gen (txt2img, {steps} steps)")
    img = _step("generate", lambda: p.generate(P1, 1024, 1024, steps, 7), log)
    img.save(os.path.join(OUT, "e2e_1_gen.png"))
    assert img.size == (1024, 1024), img.size
    assert img.getextrema() != ((0, 0), (0, 0), (0, 0)), "a black image"

    print("\n[2] edit, 1 reference")
    e1 = _step("generate_omni x1", lambda: p.generate_omni(
        [img], "make <ref_image_0> a daytime scene, sunny, blue sky", "", 1024, 1024,
        steps, 7), log)
    e1.save(os.path.join(OUT, "e2e_2_edit1.png"))
    assert e1.size == (1024, 1024), e1.size

    print("\n[3] edit, 2 references (<ref_image_0>, <ref_image_1>)")
    ref2 = Image.new("RGB", (1024, 1024), (250, 240, 200))
    ImageDraw.Draw(ref2).ellipse((300, 300, 700, 700), fill=(220, 40, 40))
    e2 = _step("generate_omni x2", lambda: p.generate_omni(
        [img, ref2], "put the character from <ref_image_0> in front of the red circle "
        "of <ref_image_1>", "", 1024, 1024, steps, 7), log)
    e2.save(os.path.join(OUT, "e2e_3_edit2.png"))

    print("\n[4] inpaint (the white area of the mask)")
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rectangle((620, 60, 980, 380), fill=255)
    ip = _step("inpaint_run", lambda: p.inpaint_run(
        img, mask, "a large full moon behind heavy clouds", steps, 0.9, 7), log)
    ip.save(os.path.join(OUT, "e2e_4_inpaint.png"))
    assert ip.size == img.size
    # outside the mask the original is back, pixel for pixel (composite back)
    assert ip.getpixel((50, 900)) == img.getpixel((50, 900))

    print("\n[5] img2img / refine (SDEdit, strength 0.35)")
    r = _step("_refine_whole d=0.35", lambda: p._refine_whole(pipe, img, 0.35, steps, P1, 7), log)
    r.save(os.path.join(OUT, "e2e_5_refine.png"))
    assert r.size == img.size

    print("\n[6] stop: an interrupt returns the image as it stood")
    calls = []
    real_cb = p._step_end_kwargs

    def with_stop(*a, **k):
        def cb(pipe_, i, t, kw):
            calls.append(i)
            if i == 2:
                pipe_._interrupt = True
            return kw
        return {"callback_on_step_end": cb}
    p._step_end_kwargs = with_stop
    try:
        s = _step("generate (stop at 3)", lambda: p.generate(P1, 1024, 1024, steps, 7), log)
    finally:
        p._step_end_kwargs = real_cb
    assert calls == [0, 1, 2], calls
    assert s.size == (1024, 1024)

    report = os.path.join(OUT, "e2e_timings.txt")
    with open(report, "w", encoding="utf-8") as f:
        f.write(f"{p.BASE_REPO} {p.PRECISION} offload={p._effective_offload()}\n" + "\n".join(log) + "\n")
    print(f"\ntimings written to {report}")


if __name__ == "__main__":
    why = _why_skip()
    if why:
        print(f"SKIP test_flux3_e2e: {why}")
        sys.exit(0)
    test_flux3_end_to_end()
    print("ALL OK")
