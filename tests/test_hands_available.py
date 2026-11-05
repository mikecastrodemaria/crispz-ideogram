"""Hands detailer availability (cz_detailer): an exported ONNX in cache/ is enough,
ultralytics alone is enough, no onnxruntime means no."""
# scope: family
import importlib.util
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_core
import cz_detailer
import cz_protocol as C


def _onnx_path():
    stem = os.path.splitext(os.path.basename(cz_detailer._HAND_MODEL))[0]
    return os.path.join(cz_core.HERE, "cache", stem + ".onnx")


def _with_find_spec(missing, fn, present=()):
    """Runs fn with the modules named in `missing` made to disappear."""
    # present: modules forced to LOOK installed. onnxruntime is optional (it comes
    # with the FaceSwap deps), so a bare install and the CI runner do not have it;
    # these tests are about the ultralytics gate and the cached .onnx, not about it.
    real = importlib.util.find_spec

    def fake(name, *a, **k):
        if name in missing:
            return None
        return object() if name in present else real(name, *a, **k)
    importlib.util.find_spec = fake
    try:
        return fn()
    finally:
        importlib.util.find_spec = real


def test_exported_onnx_is_enough():
    """Without ultralytics but with the .onnx: the feature IS available."""
    if not os.path.isfile(_onnx_path()):
        pytest.skip("no exported detector in cache/")
    assert _with_find_spec({"ultralytics"}, C._hands_available,
                           present={"onnxruntime"}) is True


def test_ultralytics_alone_is_enough():
    """With ultralytics, the export can happen on demand -> available."""
    # onnxruntime is faked present as well: this test is about the ultralytics
    # gate, and onnxruntime is an optional package that a bare install (or a CI
    # runner) does not have - its own gate is test_no_onnxruntime_means_no.
    real = importlib.util.find_spec

    def fake(name, *a, **k):
        return object() if name in ("ultralytics", "onnxruntime") else real(name, *a, **k)
    importlib.util.find_spec = fake
    try:
        assert C._hands_available() is True
    finally:
        importlib.util.find_spec = real


def test_no_onnxruntime_means_no():
    """onnxruntime is the only dependency that is really indispensable at run time."""
    assert _with_find_spec({"onnxruntime"}, C._hands_available) is False
