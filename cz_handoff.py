"""crispz-ideogram - hand an image over to crispz-flux3 for editing (lot 5).

Ideogram 4 has no editing pipeline, and the cloud editor of Ideogram 4.5 is not used
here: reference editing is done by crispz-flux3, the sibling studio, on the same machine.
The hand-over goes through crispz-flux3's OWN command line, the family protocol v1
(`cz_protocol.py edit --spec -`), so crispz-flux3 decides how to run it:

  - its app is running  -> the edit is queued in that instance (model warm, its queue
                           serializes the GPU);
  - its app is closed   -> it loads FLUX.3 in a separate process (cold). crispz-ideogram
                           first frees its own VRAM (said in the log): two models of
                           15 to 20 GB do not share a 32 GB card.

Nothing leaves the machine: the image and the references are written as PNG files under
the output folder (`handoff/`) and crispz-flux3 reads them from there (the protocol is
machine-local). Setting: `crispz_flux3_dir`, the crispz-flux3 install folder (env
CRISPZ_FLUX3_DIR). Every hand-over is logged with a `HANDOFF:` line.
"""

import datetime
import json
import os
import subprocess
import sys

from PIL import Image

from cz_core import CONFIG, _log, _prefs

TARGET = "crispz-flux3"
MAX_REFS = 3                 # the image itself is reference 0 of the edit


class HandoffError(RuntimeError):
    """A hand-over that cannot run or failed, with a message to show as is."""


def flux3_dir():
    d = (os.environ.get("CRISPZ_FLUX3_DIR") or _prefs.get("crispz_flux3_dir")
         or CONFIG.get("crispz_flux3_dir", "") or "")
    return os.path.expanduser(str(d).strip().strip('"'))


def _python(d):
    for p in (os.path.join(d, ".venv", "Scripts", "python.exe"),
              os.path.join(d, ".venv", "bin", "python")):
        if os.path.isfile(p):
            return p
    return None


def status():
    """(usable, reason) from the setting and the folder, without running anything."""
    d = flux3_dir()
    if not d:
        return False, (f"{TARGET} folder not set (`crispz_flux3_dir`): set it below to "
                       f"edit with {TARGET}, on this machine.")
    if not os.path.isdir(d):
        return False, f"{TARGET} folder not found: {d}"
    if not os.path.isfile(os.path.join(d, "cz_protocol.py")):
        return False, f"{d} has no cz_protocol.py: is it a crispz install?"
    if not _python(d):
        return False, f"{d} has no .venv (run its install first)"
    return True, f"{TARGET} at {d}"


def _czp(args, spec=None, timeout=3600):
    """Runs crispz-flux3's protocol CLI -> (exit_code, result dict). Replaced in tests."""
    d = flux3_dir()
    cmd = [_python(d), "cz_protocol.py", *args]
    p = subprocess.run(cmd, cwd=d, input=(json.dumps(spec) if spec is not None else None),
                       capture_output=True, text=True, encoding="utf-8", timeout=timeout)
    lines = [ln for ln in (p.stdout or "").splitlines() if ln.strip().startswith("{")]
    if not lines:
        tail = (p.stderr or p.stdout or "").strip()[-300:]
        raise HandoffError(f"{TARGET} gave no answer (exit {p.returncode}): {tail}")
    try:
        return p.returncode, json.loads(lines[-1])
    except ValueError:
        raise HandoffError(f"{TARGET} answer unreadable: {lines[-1][:200]}")


def caps():
    code, c = _czp(["caps"], timeout=60)
    if c.get("tool") != TARGET:
        raise HandoffError(f"the folder runs {c.get('tool') or 'an unknown tool'}, not "
                           f"{TARGET}: check `crispz_flux3_dir`")
    return c


def _handoff_dir(output_dir=None):
    from cz_core import DEFAULT_OUTPUT_DIR, HERE
    root = output_dir or DEFAULT_OUTPUT_DIR
    root = root if os.path.isabs(root) else os.path.join(HERE, root)
    d = os.path.join(root, "handoff", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(d, exist_ok=True)
    return d


def _write(img, path):
    img = img if isinstance(img, Image.Image) else Image.open(img)
    img.save(path, "PNG")
    return os.path.abspath(path)


def build_spec(image, instruction, refs=(), seed=-1, output_dir=None):
    """Writes the image and the references as PNG files -> (spec, folder)."""
    instruction = (instruction or "").strip()
    if not instruction:
        raise HandoffError("the edit needs an instruction (what to change)")
    if image is None:
        raise HandoffError("drop the image to edit first")
    refs = [r for r in (refs or []) if r is not None]
    if len(refs) > MAX_REFS:
        _log(f"HANDOFF: {len(refs)} references, {MAX_REFS} kept (the image is the first)")
        refs = refs[:MAX_REFS]
    d = _handoff_dir(output_dir)
    inp = _write(image, os.path.join(d, "input.png"))
    ref_paths = [_write(r, os.path.join(d, f"ref{i + 1}.png")) for i, r in enumerate(refs)]
    if ref_paths:
        # crispz-flux3's `edit` op keeps the input as its ONLY reference (it overwrites
        # spec.refs): with references, `gen` with refs takes the same multi-reference
        # route and keeps them all; the size is pinned to the image's.
        w, h = (image if isinstance(image, Image.Image) else Image.open(image)).size
        spec = {"protocol": 1, "op": "gen", "prompt": instruction,
                "refs": [inp] + ref_paths, "width": int(w), "height": int(h), "out_dir": d}
    else:
        spec = {"protocol": 1, "op": "edit", "prompt": instruction, "input": inp,
                "out_dir": d}
    if seed is not None and int(seed) >= 0:
        spec["seed"] = int(seed)
    return spec, d


def edit(image, instruction, refs=(), seed=-1, output_dir=None, free_vram=None):
    """-> (images, result dict, route). Raises HandoffError."""
    ok, why = status()
    if not ok:
        raise HandoffError(why)
    spec, folder = build_spec(image, instruction, refs, seed, output_dir)
    c = caps()
    inst = c.get("instance") or {}
    running = bool(inst.get("running"))
    if running and inst.get("tool") not in (None, TARGET):
        # Both studios default to the same port: crispz-flux3's probe can land on THIS app.
        _log(f"HANDOFF: the instance at {inst.get('url')} is {inst.get('tool')}, not "
             f"{TARGET} (same port in both configs?): {TARGET} runs cold instead")
        running = False
    args = [spec["op"], "--spec", "-"]
    if running:
        route = "instance"
        _log(f"HANDOFF: image sent to {TARGET} (its running app, "
             f"{inst.get('url')}), files in {folder}")
    else:
        route = "cold"
        args.append("--local")
        _log(f"HANDOFF: {TARGET} is not running: it loads FLUX.3 in its own process "
             f"(cold, slower); crispz-ideogram frees its VRAM first. Files in {folder}")
        if free_vram is not None:
            free_vram()
    code, res = _czp(args, spec)
    if not res.get("ok"):
        raise HandoffError(f"{TARGET}: {res.get('error') or 'failed'} (exit {code})")
    paths = [p for p in res.get("images") or [] if os.path.isfile(p)]
    if not paths:
        raise HandoffError(f"{TARGET} answered without an image file: {str(res)[:200]}")
    imgs = [Image.open(p).copy() for p in paths]
    for w in res.get("warnings") or []:
        _log(f"HANDOFF: {TARGET} warning: {w}")
    _log(f"HANDOFF: {TARGET} edit done ({route}): {', '.join(paths)}")
    with open(os.path.join(folder, "handoff.json"), "w", encoding="utf-8") as f:
        json.dump({"target": TARGET, "route": route, "spec": spec, "result": res}, f,
                  indent=2, ensure_ascii=False)
    return imgs, res, route


def save_setting(folder):
    """The folder field of the tab -> preferences.json and the running app."""
    from cz_core import _save_prefs_keys
    folder = (folder or "").strip().strip('"')
    try:
        _save_prefs_keys({"crispz_flux3_dir": folder})
    except Exception as e:
        _log(f"crispz_flux3_dir not persisted: {e}")
    _prefs["crispz_flux3_dir"] = folder
    return status()


if __name__ == "__main__":       # quick check: python cz_handoff.py
    print(status())
    sys.exit(0)
