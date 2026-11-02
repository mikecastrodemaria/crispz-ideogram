# crispz-ideogram.pinokio

1-click [Pinokio](https://pinokio.computer) launcher for
**[crispz-ideogram](https://github.com/mikecastrodemaria/crispz-ideogram)**, the
**Ideogram 4.0** studio of the crispz family: text in images and design (posters,
logos, packaging, dense typography), Fooocus-style, 100% local.

## Licence first

Ideogram 4.0's open weights are under the **Ideogram 4 Non-Commercial licence**:
personal and research use only. **Any commercial use of the self-hosted weights needs a
licence negotiated with Ideogram.** Do not deliver client work made with this local model
without that licence. The licence also forbids circumventing or disabling the model's
built-in safety filter.

## What it does

- **Install** clones `mikecastrodemaria/crispz-ideogram` into `app/`, creates a
  **Python 3.11** venv, installs `requirements.txt` (diffusers 0.42.1 with
  `Ideogram4Pipeline`, transformers 5, bitsandbytes for the nf4 weights, outlines for the
  local magic prompt, vtracer for local vectorization), Pillow 13 apart, the extras
  (rembg for Remove BG), the Face Swap deps and PyTorch (torch 2.11 + CUDA cu128 on
  NVIDIA). Ends with a check that `Ideogram4Pipeline`, bitsandbytes, outlines and vtracer
  import.
- **Start** runs `python app.py` and opens the Gradio Web UI.
- **Update** force-updates the launcher and the app to their `origin/main`, refreshes
  deps, reasserts the torch build, re-runs the check.
- **Reset** removes `app/` (and its venv) to reinstall from scratch.

## Before the first render: the licence and the token

The weights are **gated** on Hugging Face. Once, with your HF account:

1. Open https://huggingface.co/ideogram-ai/ideogram-4-nf4-diffusers and accept the
   licence **on that exact repo** (accepting it on `ideogram-4-nf4` is not enough).
2. Create a READ token at https://huggingface.co/settings/tokens and put it in
   `app/config.txt` as `"hf_token": "hf_..."`, or run `hf auth login` in the venv.

The first render downloads about 15 GB (two nf4 transformers of 5.2 GB, the Qwen3-VL-8B
text encoder 5.5 GB, the VAE). The first magic prompt adds a 1.2 GB head, the first
Remove BG the 176 MB u2net model.

## Requirements

- An **NVIDIA GPU** with CUDA: the nf4 weights load through bitsandbytes, CUDA only.
  Other platforms install, but cannot load the model.
- **24 GB of VRAM** is comfortable up to 2048x2048; the app keeps both transformers and
  the VAE on the card and moves only the text encoder (offload `auto`).

Measured on an RTX 5090, 48 steps, guidance 7:

| | |
|---|---|
| Load from the cache | 20 to 23 s |
| VRAM at rest | 9.9 GB |
| 1024x1024 | 61 s, 19.2 GB peak |
| 2048x2048 | 394 s, 23.4 GB peak |

## What is local, what is not

Everything runs on your machine. The optional Ideogram cloud functions (Ideogram 4.5
precise edit and friends, through api.ideogram.ai in the United States) are off and
hidden by default; see the app's README.
