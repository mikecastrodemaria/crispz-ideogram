"""What a model IS, shown in the menu that offers it.

A library of 27 checkpoints and 105 LoRAs offered nothing but file names, and the one
thing that decides whether a file loads at all -- 4B or 9B -- was invisible until it
failed. The menus now carry the variant, the size and, for a checkpoint, the format.

The contract that matters: the VALUE behind each entry is unchanged. Presets, queued
jobs and image metadata store a file name, and relabelling the menu must not touch it.

Neither GPU nor model: tiny safetensors are written with the shapes that give the
variant away.

Run:  .venv/Scripts/python tests/test_model_labels.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import cz_pipeline as P  # noqa: E402


def _ckpt(path, dim=3072, dtype=torch.bfloat16, family="flux3"):
    """A transformer the header alone can identify: the modulation layer's shape is the
    signature klein read (out = 6 x hidden); crispz-flux3 reads the FAMILY off the block
    names instead, so the fixture carries FLUX.3 names by default and FLUX.2 ones on
    request."""
    key = ("early_stream_modulations.0.lin.weight" if family == "flux3"
           else "double_stream_modulation_img.linear.weight")
    save_file({key: torch.zeros(dim * 6, dim, dtype=dtype)}, path)
    return path


def _lora(path, dim=3072, rank=8):
    save_file({"transformer.transformer_blocks.0.attn.to_q.lora_A.weight":
               torch.zeros(rank, dim, dtype=torch.bfloat16),
               "transformer.transformer_blocks.0.attn.to_q.lora_B.weight":
               torch.zeros(dim, rank, dtype=torch.bfloat16)}, path)
    return path


class _Lib:
    def __init__(self):
        self.ck = tempfile.mkdtemp()
        self.lo = tempfile.mkdtemp()

    def __enter__(self):
        self.old = (P.CHECKPOINTS_DIR, P.CHECKPOINTS_EXTRA_DIR,
                    P.LORAS_DIR, list(P.LORAS_EXTRA_DIRS))
        P.CHECKPOINTS_DIR, P.CHECKPOINTS_EXTRA_DIR = self.ck, ""
        P.LORAS_DIR, P.LORAS_EXTRA_DIRS = self.lo, []
        P._FILE_INFO_CACHE.clear()
        return self

    def __exit__(self, *exc):
        (P.CHECKPOINTS_DIR, P.CHECKPOINTS_EXTRA_DIR,
         P.LORAS_DIR, P.LORAS_EXTRA_DIRS) = self.old
        for d in (self.ck, self.lo):
            shutil.rmtree(d, ignore_errors=True)
        return False


def test_a_checkpoint_label_carries_variant_size_and_format():
    with _Lib() as lib:
        p = _ckpt(os.path.join(lib.ck, "model.safetensors"), dim=3072)
        info = P.model_info(p)
        assert info["variant"] == "FLUX.3", info
        assert info["format"] == "bf16", info
        assert info["size_h"].endswith("MB"), info
        label = P.model_label("f3/model.safetensors", p)
        for bit in ("f3/model.safetensors", "FLUX.3", "bf16", info["size_h"]):
            assert bit in label, (bit, label)


def test_a_lora_label_carries_the_variant_that_decides_if_it_loads():
    with _Lib() as lib:
        # transformer_blocks keys = a FLUX.2 Klein LoRA: the label says so, which is what
        # decides whether it loads here (it does not)
        p = _lora(os.path.join(lib.lo, "style.safetensors"), dim=3072)
        info = P.model_info(p, "lora")
        assert info["variant"] == "FLUX.2", info
        assert "FLUX.2" in P.model_label("style.safetensors", p, "lora")


def test_the_value_behind_an_entry_is_still_the_plain_name():
    """The whole point: a preset stores a name, and relabelling must not touch it."""
    with _Lib() as lib:
        _ckpt(os.path.join(lib.ck, "a.safetensors"))
        _lora(os.path.join(lib.lo, "b.safetensors"))
        ck = P.checkpoint_choices()
        lo = P.lora_choices()
        assert [v for _lab, v in ck] == P.list_checkpoints(), ck
        assert [v for _lab, v in lo] == P.list_loras(), lo
        assert all(isinstance(lab, str) and lab != v for lab, v in ck + lo)


def test_the_header_is_read_once_per_file():
    """A menu is rebuilt at every refresh; re-reading a hundred headers off a mechanical
    disk each time would be felt."""
    with _Lib() as lib:
        p = _ckpt(os.path.join(lib.ck, "a.safetensors"))
        calls = []
        real = P._safetensors_header
        P._safetensors_header = lambda path: (calls.append(path), real(path))[1]
        try:
            P.model_info(p)
            P.model_info(p)
            P.model_info(p)
        finally:
            P._safetensors_header = real
        assert len(calls) <= 1, calls          # the variant read is cached with the rest


def test_a_file_that_changes_is_read_again():
    """The cache is stamped on size and mtime: a re-downloaded model must not keep the
    label of the one it replaced."""
    with _Lib() as lib:
        p = _ckpt(os.path.join(lib.ck, "a.safetensors"))
        assert P.model_info(p)["variant"] == "FLUX.3"
        os.remove(p)
        _ckpt(p, family="flux2")
        os.utime(p, (0, 0))                    # a different stamp, whatever the clock did
        assert P.model_info(p)["variant"] == "FLUX.2"


def test_an_unreadable_file_still_gets_a_line():
    """No variant, no format -- but a name and a size beat no entry at all."""
    with _Lib() as lib:
        p = os.path.join(lib.ck, "broken.safetensors")
        with open(p, "wb") as f:
            f.write(b"not a safetensors")
        info = P.model_info(p)
        assert info["variant"] is None and info["size_h"], info
        assert P.model_label("broken.safetensors", p).startswith("broken.safetensors")


def test_a_missing_file_says_nothing_rather_than_raising():
    assert P.model_info("D:/nope/ghost.safetensors") == {}
    assert P.model_label("ghost.safetensors", "D:/nope/ghost.safetensors") == "ghost.safetensors"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"OK {fn.__name__}")
    print(f"All {len(tests)} model label tests passed.")
