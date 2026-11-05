"""Ideogram 4 on the real GPU through the app's own functions: gen 1024, gen 2048 (CZ_E2E_2048=0
skips it), img2img 0.35, inpaint, Stop at step 3. Skips without CUDA, without access to the
gated repo or with CZ_E2E=0. Images in tests/e2e_*.png, timings in tests/e2e_timings.txt."""
# scope: ideogram
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

REPO = os.environ.get("IDEOGRAM_E2E_MODEL") or "ideogram-ai/ideogram-4-nf4-diffusers"
# Pinned before cz_pipeline is imported: the run must not follow the last model or
# single file chosen in the UI (preferences.json).
os.environ["IDEOGRAM_MODEL"] = REPO
os.environ.pop("IDEOGRAM_TRANSFORMER", None)

OUT = os.path.dirname(os.path.abspath(__file__))
STEPS = 48
POSTER = ('{"high_level_description":"A bold minimalist event poster with the headline '
          '\\"GRAND OPENING\\" and the line \\"SATURDAY 9 AM\\".","style_description":'
          '{"art_style":"flat graphic design, Swiss style","color_palette":["#F2E8D5",'
          '"#D7263D","#1B1B1E"]},"compositional_deconstruction":{"background":"a warm '
          'cream paper background","elements":[{"type":"text","content":"GRAND OPENING",'
          '"bbox":[120,80,330,920]},{"type":"text","content":"SATURDAY 9 AM",'
          '"bbox":[780,200,880,800]}]}}')


def _why_skip():
    if os.environ.get("CZ_E2E", "1") == "0":
        return "CZ_E2E=0"
    try:
        import torch
    except Exception as e:
        return f"torch unavailable: {e}"
    if not torch.cuda.is_available():
        return "needs CUDA (nf4 weights through bitsandbytes)"
    if os.path.isdir(REPO):
        return None
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(REPO, "model_index.json")
    except Exception as e:
        return f"no access to {REPO}: {type(e).__name__} (accept the licence + token)"
    return None


def _peak():
    import torch
    return torch.cuda.max_memory_reserved() / 1024 ** 3


def _step(name, fn, log):
    import torch
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    out = fn()
    line = f"  {name:<26} {time.time() - t:7.1f} s   peak={_peak():5.2f} GB"
    print(line, flush=True)
    log.append(line)
    return out


@pytest.mark.skipif(_why_skip() is not None, reason=_why_skip() or "")
def test_ideogram_end_to_end():
    from PIL import ImageDraw, Image
    import cz_ideogram
    import cz_pipeline as p
    log = []
    print(f"\nbase repo: {p.BASE_REPO}  offload: {p.OFFLOAD_MODE}")

    pipe = _step("load", p._ensure_base, log)
    assert isinstance(pipe, cz_ideogram.IdeogramPipe), type(pipe).__name__

    img = _step("gen 1024", lambda: p.generate(POSTER, 1024, 1024, STEPS, 7), log)
    img.save(os.path.join(OUT, "e2e_1_gen1024.png"))
    assert img.size == (1024, 1024)
    assert img.getextrema() != ((0, 0), (0, 0), (0, 0)), "a black image"

    if os.environ.get("CZ_E2E_2048", "1") != "0":
        big = _step("gen 2048", lambda: p.generate(POSTER, 2048, 2048, STEPS, 7), log)
        big.save(os.path.join(OUT, "e2e_2_gen2048.png"))
        assert big.size == (2048, 2048)

    r = _step("img2img 0.35", lambda: p._refine_whole(p.get_pipe("img2img"), img, 0.35,
                                                     STEPS, POSTER, 7), log)
    r.save(os.path.join(OUT, "e2e_3_img2img.png"))
    assert r.size == img.size

    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rectangle((320, 420, 704, 700), fill=255)
    ip = _step("inpaint", lambda: p.inpaint_run(
        img, mask, "a large red retro bicycle, flat vector illustration", STEPS, 1.0, 7),
        log)
    ip.save(os.path.join(OUT, "e2e_4_inpaint.png"))
    assert ip.size == img.size
    assert ip.getpixel((40, 980)) == img.getpixel((40, 980)), "outside the mask is kept"

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
        s = _step("stop at step 3", lambda: p.generate(POSTER, 1024, 1024, STEPS, 7), log)
    finally:
        p._step_end_kwargs = real_cb
    assert calls[:3] == [0, 1, 2] and len(calls) == 3, calls
    assert s.size == (1024, 1024)

    report = os.path.join(OUT, "e2e_timings.txt")
    with open(report, "w", encoding="utf-8") as f:
        f.write(f"{p.BASE_REPO} offload={p._effective_offload()}\n" + "\n".join(log) + "\n")
    print(f"\ntimings written to {report}")
