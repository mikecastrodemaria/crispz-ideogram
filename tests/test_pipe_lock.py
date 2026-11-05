"""free_vram() waits for the render in progress (_GPU_LOCK): it blocks while another thread
holds the GPU, a call from inside a generation does not deadlock, the Free VRAM button,
the encoder dropdown and the offload radio are serialised with Generate."""
# scope: family
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_pipeline as P  # noqa: E402

GRACE = 0.5      # lets the calling thread reach the lock before concluding
TIMEOUT = 10.0   # a wait longer than this is a deadlock, not slowness


class _State:
    """Saves and restores every global these setters write."""

    NAMES = ("_BASE_PIPE", "_DERIVED", "_LOADED_KEY", "_APPLIED_LORAS", "_APPLIED_LOKRS",
             "_ENCODER_TRIMMED", "_TEXT_ENCODER_ACTIVE", "LORAS", "ZIMAGE_TRANSFORMER",
             "OFFLOAD_MODE", "TEXT_ENCODER", "BASE_REPO")

    def __enter__(self):
        self.saved = {n: getattr(P, n) for n in self.NAMES if hasattr(P, n)}
        return self

    def __exit__(self, *exc):
        for n, v in self.saved.items():
            setattr(P, n, v)
        return False


class _Busy:
    """Another thread holding _GPU_LOCK, exactly like a render in progress."""

    def __enter__(self):
        self.held = threading.Event()
        self.release = threading.Event()

        def hold():
            with P._GPU_LOCK:
                self.held.set()
                self.release.wait(TIMEOUT)

        self.t = threading.Thread(target=hold, daemon=True)
        self.t.start()
        assert self.held.wait(TIMEOUT), "the holding thread never took the lock"
        return self

    def __exit__(self, *exc):
        self.release.set()
        self.t.join(TIMEOUT)
        return False


class _Caller:
    """Calls fn on its own thread and says when it has entered and when it has returned."""

    def __init__(self, fn, *args):
        self.entered = threading.Event()
        self.done = threading.Event()
        self.error = None

        def run():
            self.entered.set()
            try:
                fn(*args)
            except BaseException as e:      # noqa: BLE001 - reported, not swallowed
                self.error = e
            finally:
                self.done.set()

        self.t = threading.Thread(target=run, daemon=True)

    def start(self):
        self.t.start()
        assert self.entered.wait(TIMEOUT), "the calling thread never started"
        return self

    def finished(self, timeout=0.0):
        return self.done.wait(timeout)

    def join(self):
        assert self.done.wait(TIMEOUT), "the call never returned"
        self.t.join(TIMEOUT)
        assert self.error is None, self.error


def _waits(fn, *args):
    """fn must NOT go through while the GPU is held, and must go through once it is free."""
    with _State(), _Busy() as busy:
        call = _Caller(fn, *args).start()
        assert not call.finished(GRACE), f"{fn.__name__} did not wait for the render"
        busy.release.set()
        call.join()


def _does_not_wait(fn, *args):
    """fn must go through WHILE the GPU is held: it does not touch the loaded pipe."""
    with _State(), _Busy():
        call = _Caller(fn, *args).start()
        assert call.finished(TIMEOUT), f"{fn.__name__} waited although it need not"
        call.join()


def test_free_vram_waits_for_the_render_in_progress():
    _waits(P.free_vram)


def test_free_vram_from_inside_a_generation_does_not_deadlock():
    """retry_on_oom and _consume_vram_downgrade free the VRAM on the generation's own
    thread, which already holds the lock. An RLock must let that through -- a plain Lock
    here would hang every OOM retry."""
    with _State():
        with P._GPU_LOCK:
            P._BASE_PIPE = object()
            P.free_vram()
            assert P._BASE_PIPE is None, "free_vram did not run"


def test_changing_the_offload_mode_waits():
    with _State():
        P.OFFLOAD_MODE = "none"
    _waits(P.set_offload_mode, "model")


def _text_encoder_args():
    """crispz-krea drives TWO encoders (T5 + CLIP), so its setter takes (component, src);
    the other four take src alone. Read from the signature rather than hard-coded per
    fork -- this file is the same on the five."""
    import inspect
    n = len(inspect.signature(P.set_text_encoder).parameters)
    src = os.path.join("D:", "enc", "does-not-exist")
    return (P.TEXT_ENCODER_COMPONENTS[0], src) if n == 2 else (src,)


def test_changing_the_text_encoder_waits():
    _waits(P.set_text_encoder, *_text_encoder_args())


def test_changing_the_base_repo_waits():
    _waits(P.set_zimage_model, "someone/a-base-repo-that-does-not-exist")


def test_setting_the_loras_does_not_wait():
    """set_loras only writes LORAS; _ensure_base hot-swaps on the next run, under the lock.
    Making it wait would freeze the Apply button during a render for no benefit."""
    _does_not_wait(P.set_loras, [])


def test_setting_the_single_file_transformer_does_not_wait():
    """Same reason: a global read by the next _ensure_base, no touch to the loaded pipe."""
    _does_not_wait(P.set_zimage_transformer, "")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"OK {fn.__name__}")
    print(f"All {len(tests)} shared-pipe lock tests passed.")
