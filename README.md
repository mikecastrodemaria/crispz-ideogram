# crispz-ideogram

The **Ideogram 4.0** studio of the crispz family: text in images and design (posters,
logos, packaging, slides, covers, dense typography), Fooocus-style, **100% local**.
Ideogram 4 is the open-weights model that renders text best; the other crispz studios
stay better at photos and editing.

## Licence: read this first

- **The model.** Ideogram 4.0's weights are under the **Ideogram 4 Non-Commercial
  licence**: personal use and research only. **Any commercial use of the self-hosted
  weights needs a licence negotiated with Ideogram.** No client deliverable made locally
  with this model without that licence. The licence also forbids circumventing or
  disabling the model's safety filter (built into the weights: a blocked image comes out
  as a grey card).
- **This app's code** is under CC BY-NC 4.0 (see [LICENSE.txt](LICENSE.txt)).
- Images made through the optional Ideogram cloud API follow the API's own terms, not
  the licence above (see *Local and cloud*).

## What it does

| | |
|---|---|
| **Generate** | Ideogram 4.0 text to image, 48 steps, the official per-step guidance schedule (the slider sets its peak, 7 by default), up to 2048x2048, ratios up to 6:1 |
| **JSON captions** | Ideogram 4 was trained on structured captions: an editor builds them (summary, style, palette, every text with its exact content and its box), boxes drawn with the mouse on a canvas at the chosen ratio, raw JSON mode |
| **Design presets** | poster (A series), logo, business card, book cover, slide 16:9, packaging, YouTube thumbnail, story 9:16, banner 6:1, each at its native size |
| **Magic prompt** | a plain idea into a JSON caption, locally (Qwen3-VL-8B, the model's own text encoder, constrained by outlines) or through Ollama |
| **Describe** | an image into a JSON caption, locally (Qwen3-VL-8B reads the texts verbatim and places the boxes) |
| **img2img, inpaint, outpaint, reframe** | **emulated**: Ideogram 4 has no native pipeline for them. SDEdit on the tail of its own schedule, latent blending under the mask |
| **Upscale** | ESRGAN + diffusion refine, tiled |
| **Vectorize (SVG)** | locally with vtracer: made for logos, icons and flat colours |
| **Remove BG** | locally with rembg |
| **Edit with references** | handed to **crispz-flux3** on the same machine (see below) |
| **The rest of the family** | styles, wildcards, LoRAs (Ideogram 4 ones only), X/Y/Z grids, job queue, Asset Browser, CivitAI search, face swap, detailer, CLI and the family protocol v1 (`czp`) |

There is **no negative prompt**: Ideogram 4's unconditional branch takes no text, so the
field is greyed out and says why.

## Requirements

- An **NVIDIA GPU** with CUDA: the nf4 weights load through bitsandbytes, CUDA only.
- **24 GB of VRAM** is comfortable up to 2048x2048 (32 GB measured below). The `auto`
  offload keeps the two transformers and the VAE on the card and moves only the text
  encoder when the card has under 40 GB.
- Python 3.11 or later (3.13 advised), Windows or Linux.

## Install

Windows: `install.bat`, then `run.bat`. Linux: `install.sh`, then `run.sh`. Or the
Pinokio launcher: [crispz-ideogram.pinokio](https://github.com/mikecastrodemaria/crispz-ideogram.pinokio).

Do **not** run `pip install -r requirements.txt` in the installed `.venv`: gradio 5.50
declares `pillow<12` and pip would downgrade Pillow 13 (the install scripts handle it).
To add one dependency, install it alone at its locked version.

### Before the first render: the licence and the token

1. Open https://huggingface.co/ideogram-ai/ideogram-4-nf4-diffusers and accept the
   licence **on that exact repo** (accepting it on `ideogram-4-nf4` is not enough).
2. Put a READ token in `config.txt` (`"hf_token": "hf_..."`) or run `hf auth login`.

The first render downloads about 15 GB (two nf4 transformers of 5.2 GB, the Qwen3-VL-8B
encoder 5.5 GB, the VAE), 20 s to load from the cache afterwards. The first magic prompt
adds a 1.2 GB head; the first Remove BG the 176 MB u2net model.

**An old `config.txt`** (written by crispz-flux3 or before 1.0.0) keeps its values:
check `default_model_profile` (48 steps, guidance 7) and the performance preset.

## Measured on an RTX 5090

Windows, torch 2.11 cu128, bitsandbytes 0.50.2, nf4, offload `auto` (encoder only),
48 steps, guidance 7.

| | |
|---|---|
| Load from the cache | 20 to 23 s |
| VRAM at rest | 9.9 GB allocated |
| 1024x1024 | 61 s (1.27 s/step), 19.2 GB peak |
| 2048x2048 | 394 s (8.2 s/step), 23.4 GB peak, VAE decode 1.1 s |
| Stop | the partial image comes back 1.4 s after the request |
| Magic prompt (local) | 10.7 to 18.5 s (44 s the first time: the head and the schema) |
| Describe (local) | 21.7 s, the poster's texts read back |
| img2img 720x1024, denoise 0.3 / 0.6 | 29 s (load included) / 26 s |
| Inpaint, outpaint, reframe (0.7 to 1 Mpx) | 40 s, 57 s, 63 s |
| Vectorize a logo / a 768 px photo (CPU) | 0.05 s, 3 paths / 3 s, about 2,000 paths |
| Remove BG (CPU, after the first call) | 1.1 s |

## Getting good text

- **Use JSON captions.** Plain text works, but the model was trained on JSON: put every
  text in its own element with its exact content and a box. Start from a design preset
  or the magic prompt, then fix the texts in the editor.
- **The safety filter has false positives** on plain-text prompts (a grey "blocked"
  card on a harmless poster). Measured: 1 blocked card out of 2 in plain text, 0 out of
  2 with the same ideas in JSON. Rewording or a JSON caption is the fix; the filter
  itself is part of the weights and of the licence.
- Inpaint: describe the whole picture, not only the patch.

## Editing with references: crispz-flux3

Ideogram 4 does not edit from references. The **Edit in crispz-flux3** tab hands the
image (and up to 3 references) to crispz-flux3 on the same machine, through its own
protocol v1 command line:
- set the crispz-flux3 install folder in the tab (`crispz_flux3_dir`);
- if crispz-flux3's app is running, the edit goes through it (model warm); if not, it
  runs in crispz-flux3's own process after crispz-ideogram frees its VRAM;
- mind the ports: both apps start on 7860, and the second one started takes 7861.
  crispz-flux3 looks for its own app at its `cli_protocol.instance_url` (7860 by
  default), so **start crispz-flux3 first**, or point that setting at the port its app
  really uses. If it finds crispz-ideogram there instead, the hand-over says so and runs
  crispz-flux3 cold.

Nothing leaves the machine: the files go under the output folder (`handoff/`).

## Local and cloud

**Everything above runs locally.** The Ideogram cloud functions (Ideogram 4.5 precise
edit, replace background, remix, remove background, upscale, transparent, vectorize, SVG,
the cloud magic prompt and Describe, through api.ideogram.ai in the United States) exist
in the code but are **off and hidden**: no tab, no button, nothing sent. To turn them on:
`"ideogram_cloud_enabled": true` in `config.txt`, restart, then the API key in the
*Cloud (Ideogram)* tab. Every call is then logged (`CLOUD:` lines) and written to
`ideogram_cloud_journal.jsonl` in the output folder. They have never been called with a
real key: the endpoints are checked against Ideogram's published API v2 specification
only.

## Command line

`python app.py --help`, the cheat sheet in [README_CLI.md](README_CLI.md), and the
family protocol: `czp caps` / `czp gen --spec spec.json` (routes to the running app,
else runs locally). `caps` says what is emulated (`supports.emulated`) and whether the
cloud is on (`supports.cloud`).

## Tests

`.venv\Scripts\python tools\run_tests.py` (CPU, a tiny random Ideogram 4, pytest in one
process per file; green in CI). Every file opens with a `# scope:` line: `family` for the
code shared by the crispz studios, `ideogram` for what depends on the model, `flux3` for
the FLUX.3 path kept so that fixes merge from crispz-flux3.
On the GPU, app closed: `.venv\Scripts\python -m pytest tests\test_ideogram_e2e.py -s`
(gen 1024, gen 2048, img2img, inpaint, Stop; images in `tests/e2e_*.png`). The lot
scripts in `tools/` (`gpu_lot1_check.py`, `gpu_lot2_check.py`, `gpu_lot3_check.py`)
measure each part on the card.

## Credits

diffusers (Apache-2.0) for `Ideogram4Pipeline`; Ideogram's open caption verifier and
magic prompt (Apache-2.0, `third_party/ideogram4/`); vtracer (MIT); rembg (MIT); the
crispz family (crispz-klein, crispz-flux3) this studio is forked from. How and why:
[FORK.md](FORK.md); the full work log: [CLAUDE.md](CLAUDE.md).
