module.exports = {
  run: [
    // Force-update the launcher repo to exactly match its origin (main)
    {
      method: "shell.run",
      params: {
        message: [
          "git fetch origin",
          "git reset --hard origin/main"
        ]
      }
    },
    // Force-update crispz-ideogram (app) to exactly match its origin (main)
    {
      method: "shell.run",
      params: {
        path: "app",
        message: [
          "git fetch origin",
          "git reset --hard origin/main"
        ]
      }
    },
    // Refresh dependencies (in case requirements changed).
    // Note: this can pull a generic torch wheel that overrides the CUDA build,
    // so torch.js is re-run right after to restore the correct, consistent torch.
    {
      method: "shell.run",
      params: {
        venv: "env",
        venv_python: "3.11",
        // UTF-8 env: packages built from source read UTF-8 files in their
        // setup; the legacy cp1252 codec on Windows fails the build.
        env: {
          PYTHONUTF8: "1",
          PYTHONIOENCODING: "utf-8"
        },
        path: "app",
        message: [
          "uv pip install -r requirements.txt"
        ]
      }
    },
    // Pillow 13 apart, as in Install (gradio 5.50 declares pillow<12).
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install --no-deps pillow==13.0.0"
        ]
      }
    },
    // The extras (rembg for Remove BG, trustmark, c2pa, ultralytics) AFTER Pillow 13:
    // every rembg since 2.0.75 requires pillow>=12.1 while gradio 5.50 requires
    // pillow<12, so asking for both in one resolution fails (seen 2026-10-28). Installed
    // here, rembg resolves against the Pillow 13 already in place.
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: {
          PYTHONUTF8: "1",
          PYTHONIOENCODING: "utf-8"
        },
        path: "app",
        message: [
          "uv pip install -r requirements-extra.txt"
        ]
      }
    },
    // Face Swap tab dependencies (requirements-faceswap.txt).
    // insightface depends on the CPU 'onnxruntime', which shares its files with
    // 'onnxruntime-gpu': whichever is installed last wins. Installing both in one
    // resolution silently leaves the CPU build on top (no CUDAExecutionProvider),
    // so the runtime matching this machine is force-reinstalled last.
    // onnxruntime-gpu has no macOS/ROCm wheels -> plain onnxruntime elsewhere.
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: {
          PYTHONUTF8: "1",
          PYTHONIOENCODING: "utf-8"
        },
        path: "app",
        message: [
          "uv pip install \"insightface>=0.7\"",
          "uv pip install --force-reinstall --no-deps {{gpu === 'nvidia' ? 'onnxruntime-gpu' : 'onnxruntime'}}"
        ]
      }
    },
    // Reassert the correct torch build (CUDA cu128 / ROCm / MPS / CPU) so that
    // updating dependencies never leaves a mismatched torch (WinError 127).
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: "app"
        }
      }
    },
    // Same guard as Install: a dependency refresh must not silently land a
    // diffusers build without Ideogram4Pipeline, or lose the nf4 quantizer.
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "python -c \"from diffusers import Ideogram4Pipeline; import bitsandbytes, outlines, vtracer; print('Ideogram 4 pipeline, nf4, outlines, vtracer OK')\""
        ]
      }
    }
  ]
}
