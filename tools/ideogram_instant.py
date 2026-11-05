r"""Download, inspect and render an Ideogram Instant BF16 smoke test.

Run from the repo: .venv\Scripts\python tools/ideogram_instant.py --download --render
The download is resumable. No preferences or configuration files are changed.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


def check_test_poster(image):
    """A narrow smoke check for THIS red-on-white prompt, not a general quality score."""
    import numpy as np
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    white_fraction = float((pixels.min(axis=-1) > 220).mean())
    red_fraction = float(((red > 130) & (red > 1.5 * green) & (red > 1.5 * blue)).mean())
    return dict(passed=white_fraction > .15 and red_fraction > .005,
                white_fraction=round(white_fraction, 4), red_fraction=round(red_fraction, 4),
                note="Checks requested red lettering/white background only; review text and composition visually")


def main():
    from cz_ideogram_instant import FILENAME, REPO, STEPS, checkpoint_plan
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download BF16 if needed")
    parser.add_argument("--render", action="store_true", help="Render a CUDA test, close other GPU apps first")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "ideogram-instant-nsfw" / FILENAME)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "out" / "ideogram-instant-test.png")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    path = args.checkpoint.resolve()
    if args.download and not path.exists():
        from huggingface_hub import hf_hub_download
        downloaded = Path(hf_hub_download(REPO, FILENAME, local_dir=str(path.parent)))
        if downloaded.resolve() != path:
            parser.error(f"For --download, --checkpoint filename must be {FILENAME}")
    print(f"Inspecting {path}", flush=True)
    plan = checkpoint_plan(str(path))
    print(f"Validated {len(plan)} source tensors, {sum(map(len, plan.values()))} destination tensors", flush=True)
    if not args.render:
        return 0
    import torch
    import cz_ideogram
    if not torch.cuda.is_available():
        parser.error("CUDA GPU required for the full-size smoke test")
    if not path.name.lower().startswith("fal_ideo_4_nsfw"):
        parser.error("Keep the published checkpoint filename so Instant is selected explicitly")
    torch.cuda.reset_peak_memory_stats()
    free_gib = torch.cuda.mem_get_info()[0] / 1024**3
    offload = "model (staged encoder/transformer)"
    if free_gib < 22:
        parser.error(f"Only {free_gib:.1f} GiB of VRAM is free. Free at least 22 GiB before testing BF16.")
    print(f"Free VRAM: {free_gib:.1f} GiB; offload: {offload}", flush=True)
    started = time.perf_counter()
    transformer = cz_ideogram.load_transformer(cz_ideogram.DEFAULT_REPO, torch.bfloat16,
                                              single_file=str(path))
    print("Transformer loaded; loading shared components", flush=True)
    pipe = cz_ideogram.load(transformer=transformer, token=os.environ.get("HF_TOKEN"))
    print("Components loaded; placing model", flush=True)
    pipe.enable_model_cpu_offload()
    print("Model placed; generating", flush=True)
    prompt = json.dumps({
        "high_level_description": "A clean typographic poster with bold red lettering on a white background.",
        "compositional_deconstruction": {
            "background": "Plain white paper.",
            "elements": [{"type": "text", "text": "CRISPZ IDEOGRAM",
                          "desc": "Large bold red sans serif lettering, centered on the poster."}]}},
        ensure_ascii=False)
    image = pipe(prompt=prompt, width=1024, height=1024, num_inference_steps=args.steps,
                 guidance_scale=1.0, generator=torch.Generator("cpu").manual_seed(args.seed)).images[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    metadata = dict(checkpoint=str(path), steps=args.steps, guidance=1.0, shift=5.25,
                    sampler="euler", schedule="simple", seed=args.seed, prompt=prompt,
                    offload=offload,
                    elapsed_seconds=round(time.perf_counter() - started, 2),
                    peak_vram_gib=round(torch.cuda.max_memory_allocated() / 1024**3, 2),
                    poster_smoke_check=check_test_poster(image))
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {args.output}\n{json.dumps(metadata, indent=2)}")
    if not metadata["poster_smoke_check"]["passed"]:
        print("Rendering completed but the poster smoke check FAILED. "
              "This integration is experimental; a saved PNG alone does not validate the model.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Instant test failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
