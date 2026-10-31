"""Runs the whole tests/test_*.py suite in separate processes and summarises.

Separate processes on purpose: the tests manipulate the modules' GLOBAL state
(CHECKPOINTS_DIR, FORCE_RATIO, the caches...) and would pollute one another in a single
interpreter. pytest for the FLUX.3 files (tests/test_flux3_*.py), plain scripts for the rest.

Usage:
    .venv/Scripts/python tools/run_tests.py            # everything
    .venv/Scripts/python tools/run_tests.py xyz quant  # the ones whose name matches
    .venv/Scripts/python tools/run_tests.py -v         # the full output of the failures

"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TESTS = os.path.join(ROOT, "tests")


def main(argv):
    verbose = "-v" in argv or "--verbose" in argv
    filters = [a for a in argv if not a.startswith("-")]
    if not os.path.isdir(TESTS):
        print(f"no tests directory at {TESTS}")
        return 0
    files = sorted(f for f in os.listdir(TESTS)
                   if f.startswith("test_") and f.endswith(".py")
                   and (not filters or any(k.lower() in f.lower() for k in filters)))
    if not files:
        print(f"no test file matches {filters}")
        return 1
    failed, t_all = [], time.time()
    for f in files:
        t0 = time.time()
        # UTF-8 forced: the tests print non-ASCII labels, and a Windows console in
        # cp1252 would make the print fail rather than the test itself.
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        path = os.path.join(TESTS, f)
        # The FLUX.3 test files are written for pytest (fixtures, monkeypatch, skipif) and
        # have no __main__ runner: run as a script they would import, do nothing and pass.
        # They go through pytest, which the CI installs; the klein-era files keep their
        # script runner (the module globals they poke stay isolated per process).
        with open(path, encoding="utf-8", errors="replace") as fh:
            uses_pytest = "import pytest" in fh.read()
        cmd = ([sys.executable, "-m", "pytest", path, "-q", "-p", "no:cacheprovider"]
               if uses_pytest else [sys.executable, path])
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", cwd=ROOT, env=env)
        # pytest exits 5 when every test of a file is skipped (the GPU e2e on a CPU runner)
        ok = p.returncode == 0 or (uses_pytest and p.returncode == 5)
        print(f"{'PASS' if ok else 'FAIL'}  {f:<34} {time.time() - t0:5.1f}s")
        if not ok:
            failed.append(f)
            tail = (p.stdout or "") + (p.stderr or "")
            print("\n".join(tail.strip().splitlines()[-(200 if verbose else 12):]))
            print()
    print(f"\n{len(files) - len(failed)}/{len(files)} passed "
          f"in {time.time() - t_all:.1f}s"
          + (f" | FAILED: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
