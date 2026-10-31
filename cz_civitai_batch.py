"""crispz-flux3 — batch CivitAI enrichment (standalone, no torch import → starts fast).

Scans the LoRA and/or checkpoint folders and, for every `.safetensors`, fetches the
missing CivitAI info (preview + trigger words + example prompts) and refreshes the
"newer version available" flag — the same per-model action as the Asset Browser's
🔎 button, but for the whole folder.

Two entry points share the SAME core (`enrich`):
  - the Asset Browser "🔄 Fetch all missing" button (cz_ui runs `enrich` in a thread);
  - this script, meant to be launched from `civitai_index.bat` / `.sh` (and in PARALLEL
    with `--shard i/m` to split the file list across processes).

CLI:
  python cz_civitai_batch.py --kind {loras,models,all} [--force] [--all]
         [--shard i/m] [--loras-dir DIR] [--checkpoints-dir DIR]
         [--api-key KEY] [--sleep 0.5]

Only stdlib + cz_core/cz_civitai. Defensive: a corrupt sidecar, an unknown hash or a
network error never aborts the batch — that file is counted as failed/skipped and the
loop continues.
"""

import os
import sys
import time
import argparse

import cz_core
from cz_core import CONFIG, _prefs, HERE
import cz_civitai

# .gguf included: the Asset Browser's Models tab lists the GGUF checkpoints (the quantized
# FLUX.3 builds people download from CivitAI are GGUF as often as safetensors), and
# CivitAI indexes them by SHA256 like any other file. Left out, the batch said "done" and
# the GGUF rows stayed without preview or examples, with nothing to explain why (2026-10-28).
_EXTS = (".safetensors", ".gguf", ".ckpt", ".pt", ".sft")
_BATCH_CFG = CONFIG.get("civitai_batch") if isinstance(CONFIG.get("civitai_batch"), dict) else {}
DEFAULT_SLEEP = float(_BATCH_CFG.get("sleep", 0.5))
DEFAULT_CHECK_UPDATES = bool(_BATCH_CFG.get("check_updates", True))


def _split_dirs(spec):
    """Folder list from a JSON list or from an 'a;b' string (os.pathsep or ';'), without
    duplicates. Same rule as cz_pipeline._split_dirs, repeated here on purpose: importing
    cz_pipeline would pull torch in and cost seconds at every launch of this script."""
    if not spec:
        return []
    parts = ([p for chunk in spec.split(os.pathsep) for p in chunk.split(";")]
             if isinstance(spec, str) else list(spec))
    out = []
    for p in parts:
        p = str(p or "").strip()
        if p and p not in out:
            out.append(p)
    return out


def resolve_dirs(loras_dir=None, checkpoints_dir=None):
    """Resolves ([loras_dirs...], [checkpoints_dirs...]) without importing cz_pipeline (so
    without loading torch). The same order of priority as cz_pipeline: the arg > the env >
    the prefs > the config > the default <HERE>/loras|checkpoints.

    With NO argument, the EXTRA folders count too ('loras_extra_dirs',
    'checkpoints_extra_dir'). Most libraries keep their models outside the app folder (a
    Civitai library shared with other tools): scanning the main folder alone found nothing
    at all and the batch exited with "no models found". An explicit argument names exactly
    what to scan -- one folder, or a LIST of folders (what the Asset Browser passes: its
    live list)."""
    def _folders(arg, env, pref, cfg, env_extra, pref_extra, default):
        if isinstance(arg, (list, tuple)):
            return list(arg)
        if arg:
            return [arg]                      # one folder asked = that one, nothing else
        main = (os.environ.get(env) or _prefs.get(pref) or CONFIG.get(cfg)
                or os.path.join(HERE, default))
        extra = (os.environ[env_extra] if env_extra in os.environ
                 else (_prefs.get(pref_extra) or CONFIG.get(pref_extra)))
        return [main] + _split_dirs(extra)

    loras = _folders(loras_dir, "LORAS_DIR", "loras_dir", "loras_dir",
                     "LORAS_EXTRA_DIRS", "loras_extra_dirs", "loras")
    cks = _folders(checkpoints_dir, "CHECKPOINTS_DIR", "checkpoints_dir", "checkpoints_dir",
                   "CHECKPOINTS_EXTRA_DIR", "checkpoints_extra_dir", "checkpoints")
    return ([os.path.abspath(d) for d in _split_dirs(loras)],
            [os.path.abspath(c) for c in _split_dirs(cks)])


def _list_safetensors(dirs):
    """A RECURSIVE list of the models (.safetensors/.ckpt/.pt) in the given folders.
    Deduplicated by file name (the 1st folder has the priority, as for the checkpoints)."""
    out, seen = [], set()
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        for root, sub, files in os.walk(d):
            sub[:] = [x for x in sub if x not in ("_index", ".cache", "recipes")]
            for f in sorted(files):
                if f.lower().endswith(_EXTS) and f not in seen:
                    seen.add(f)
                    out.append(os.path.join(root, f))
    return sorted(out, key=lambda p: os.path.basename(p).lower())


def _apply_shard(files, shard):
    """shard 'i/m' (1<=i<=m) -> the subset files[i-1::m] (a disjoint partition, to launch
    m processes in parallel). shard None/invalid -> the list unchanged."""
    if not shard:
        return files
    try:
        i, m = shard.split("/")
        i, m = int(i), int(m)
        if m >= 1 and 1 <= i <= m:
            return files[i - 1::m]
    except Exception:
        pass
    return files


def collect_files(kind, loras_dir=None, checkpoints_dir=None, shard=None):
    loras, cks = resolve_dirs(loras_dir, checkpoints_dir)
    files = []
    if kind in ("loras", "all"):
        files += _list_safetensors(loras)
    if kind in ("models", "all"):
        files += _list_safetensors(cks)
    return _apply_shard(files, shard)


def _needs_enrich(path, overwrite):
    """True when it has to be (re)downloaded: no CivitAI enrichment, OR no preview, OR
    overwrite was asked for. We test 'modelId' and not the mere presence of the sidecar:
    that one can hold nothing but our hash cache (a model unknown to CivitAI)."""
    if overwrite:
        return True
    if not cz_civitai.load_civitai_sidecar(path).get("modelId"):
        return True
    return not cz_civitai.has_preview(path)


def enrich(files, api_key=None, overwrite=False, only_missing=True, sleep=DEFAULT_SLEEP,
           check_updates=DEFAULT_CHECK_UPDATES, progress=None):
    """The shared core (the CLI + the UI button). For every file: it enriches when some
    info is missing (or overwrite), otherwise it only refreshes the 'new version' flag.
    progress is called as progress(i, n, name, phase, text) — i is 1-based. Returns a
    summary dict."""
    n = len(files)
    summary = {"total": n, "enriched": 0, "skipped": 0, "updated": 0, "failed": 0,
               "warnings": []}

    def _emit(i, name, phase, text):
        if progress:
            try:
                progress(i, n, name, phase, text)
            except Exception:
                pass

    for i, path in enumerate(files, 1):
        name = os.path.basename(path)
        try:
            if only_missing and not _needs_enrich(path, overwrite):
                # already enriched -> just (re)check the version, without re-downloading
                _emit(i, name, "update", f"{name}: checking version…")
                res = cz_civitai.refresh_update_flag(path, api_key) if check_updates else {}
                summary["skipped"] += 1
                if res.get("update_available"):
                    summary["updated"] += 1
                    summary["warnings"].append(f"{name}: newer version on CivitAI")
                if check_updates:
                    time.sleep(sleep)
                continue

            def _pf(phase, frac, text, _i=i, _name=name):
                _emit(_i, _name, phase, f"{_name}: {text}")

            _emit(i, name, "start", f"{name}: fetching…")
            res = cz_civitai.fetch_civitai_for_model(
                path, api_key=api_key, overwrite=overwrite, progress=_pf,
                check_update=check_updates)
            if res.get("success"):
                summary["enriched"] += 1
                if res.get("update_available"):
                    summary["updated"] += 1
                    summary["warnings"].append(f"{name}: newer version on CivitAI")
            else:
                summary["failed"] += 1
                summary["warnings"].append(f"{name}: {res.get('message', 'failed')}")
            time.sleep(sleep)
        except Exception as e:  # noqa: BLE001 - one file must never break the batch
            summary["failed"] += 1
            summary["warnings"].append(f"{name}: {e}")
            cz_core._dbg(f"batch enrich failed for {name}: {e}")
    return summary


def run(kind="all", loras_dir=None, checkpoints_dir=None, shard=None, api_key=None,
        overwrite=False, only_missing=True, sleep=DEFAULT_SLEEP,
        check_updates=DEFAULT_CHECK_UPDATES, progress=None):
    files = collect_files(kind, loras_dir, checkpoints_dir, shard)
    return enrich(files, api_key=api_key, overwrite=overwrite, only_missing=only_missing,
                  sleep=sleep, check_updates=check_updates, progress=progress)


def hashing_note(files):
    """One line saying how much has to be READ before CivitAI can even be asked, or ""
    when every hash is already cached.

    CivitAI only knows a file by its SHA256, and that means reading the whole file once:
    on 2026-10-28 an 11 GB bf16 checkpoint on a mechanical drive took about three minutes
    at 60 MB/s, five checkpoints a quarter of an hour, with nothing on screen but a
    percentage that moved slowly. People read that as "it does not work". The hash is
    cached in the sidecar afterwards, so this is paid once per file."""
    todo, size = 0, 0
    for p in files:
        try:
            if cz_civitai.model_sha256(p, allow_compute=False):
                continue
            size += os.path.getsize(p)
            todo += 1
        except Exception:
            continue
    if not todo:
        return ""
    gb = size / 1024 ** 3
    # 60 MB/s is the mechanical-drive figure measured above; an NVMe is 10 to 30 times faster
    slow_min = size / (60 * 1024 ** 2) / 60
    return (f"[civitai-batch] {todo} file(s) have no cached SHA256 yet: {gb:.1f} GB to read "
            f"before CivitAI can be asked (first time only, the hash is kept in the "
            f".civitai.json sidecar). Up to about {max(1, round(slow_min))} min on a hard "
            f"disk, well under a minute on an NVMe.")


def _cli_progress(i, n, name, phase, text):
    # a single line rewritten in place: [12/48] modelname: Downloading preview…
    sys.stderr.write(f"\r[{i}/{n}] {text}".ljust(90)[:90])
    sys.stderr.flush()


def main(argv=None):
    # Robust output even on a cp1252 console (the .bat can run outside the Pinokio env):
    # UTF-8 + replacement, so as never to crash on … / — / emojis.
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="Batch CivitAI enrichment for crispz-flux3 "
                                             "(previews / trigger words / example prompts / "
                                             "new-version warnings).")
    ap.add_argument("--kind", choices=["loras", "models", "all"], default="all")
    ap.add_argument("--force", action="store_true",
                    help="Re-download everything, overwriting existing previews/sidecars.")
    ap.add_argument("--all", dest="all_files", action="store_true",
                    help="Process every file (re-query metadata) instead of only the "
                         "ones missing info. Previews are NOT overwritten unless --force.")
    ap.add_argument("--shard", default=None, metavar="i/m",
                    help="Process only shard i of m (1-based) — run several in parallel.")
    ap.add_argument("--loras-dir", default=None,
                    help="Scan THIS folder only (default: the configured folder AND its "
                         "extras, 'loras_extra_dirs').")
    ap.add_argument("--checkpoints-dir", default=None,
                    help="Scan THIS folder only (default: the configured folder AND its "
                         "extra, 'checkpoints_extra_dir').")
    ap.add_argument("--api-key", default=None, help="CivitAI API key (else config/prefs).")
    ap.add_argument("--sleep", type=float, default=DEFAULT_SLEEP,
                    help="Seconds to wait between requests (rate-limit friendly).")
    ap.add_argument("--no-check-updates", action="store_true",
                    help="Skip the newer-version check (faster, one less request/model).")
    a = ap.parse_args(argv)

    files = collect_files(a.kind, a.loras_dir, a.checkpoints_dir, a.shard)
    if not files:
        print(f"[civitai-batch] no models found for kind={a.kind} "
              f"(shard={a.shard or 'all'}). Nothing to do.")
        return 0
    print(f"[civitai-batch] {len(files)} model(s) — kind={a.kind}, shard={a.shard or 'all'}, "
          f"force={a.force}, only_missing={not a.all_files}, sleep={a.sleep}s")
    note = hashing_note(files)
    if note:
        print(note)
    s = enrich(files, api_key=a.api_key, overwrite=a.force,
               only_missing=(not a.all_files and not a.force), sleep=a.sleep,
               check_updates=(not a.no_check_updates), progress=_cli_progress)
    sys.stderr.write("\n")
    print(f"[civitai-batch] done: enriched={s['enriched']} skipped={s['skipped']} "
          f"updated={s['updated']} failed={s['failed']}")
    # The Asset Browser reads _index/models.json and loras.json, rebuilt when the UI's own
    # "Fetch all missing" finishes but NOT by this script until now: after a .bat run the
    # browser kept showing placeholders for models whose sidecar and preview were on disk,
    # until the tab was closed and reopened (seen 2026-10-28). cz_assetbrowser imports no
    # torch, so the rebuild costs nothing here.
    try:
        import cz_assetbrowser
        from cz_core import DEFAULT_OUTPUT_DIR
        loras, cks = resolve_dirs(a.loras_dir, a.checkpoints_dir)
        cz_assetbrowser.ab_build_catalog(DEFAULT_OUTPUT_DIR, loras, cks)
        print("[civitai-batch] Asset Browser catalogue refreshed (reload the tab to see it).")
    except Exception as e:  # noqa: BLE001 - a catalogue miss must not fail the batch
        print(f"[civitai-batch] catalogue not refreshed ({e}); it rebuilds when the Asset "
              f"Browser is opened again.")
    for w in s["warnings"][:50]:
        print(f"  - {w}")
    # a non-zero exit code only when EVERYTHING failed (useful for a .bat)
    return 1 if (s["failed"] and not s["enriched"] and not s["skipped"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
