module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        // Force UTF-8 stdout/stderr so tqdm download progress bars don't crash
        // on Windows consoles using the legacy cp1252 codec (UnicodeEncodeError).
        // No offload override here: the app's default_cpu_offload is 'auto', which
        // measures the free VRAM at load time and, under 40 GB, keeps the two Ideogram
        // transformers and the VAE on the card and moves only the text encoder (9.9 GB
        // resident, 19 GB peak at 1024, 23.4 GB at 2048, measured on an RTX 5090).
        env: {
          PYTHONUTF8: "1",
          PYTHONIOENCODING: "utf-8"
        },
        path: "app",
        message: [
          "python app.py"
        ],
        on: [{
          // Capture the local Gradio URL (e.g. http://127.0.0.1:7860)
          "event": "/(http:\\/\\/[0-9.:]+)/",
          "done": true
        }]
      }
    },
    {
      // Expose the captured URL to pinokio.js (Open Web UI menu item)
      method: "local.set",
      params: {
        url: "{{input.event[1]}}"
      }
    }
  ]
}
