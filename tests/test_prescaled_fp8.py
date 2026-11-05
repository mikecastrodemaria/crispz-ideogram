"""FP8 / INT8 single files stored ALREADY scaled (cz_pipeline): the detector separates the two
layouts, the loader reads a file mixing both."""
# scope: family
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from safetensors.torch import save_file

import cz_pipeline as P

torch.manual_seed(0)
E4 = torch.float8_e4m3fn


def _pair(n=64):
    w = torch.randn(n, n) * 0.02
    return w, (w.abs().max() / 448.0).reshape(())


def test_the_detector_separates_the_two_layouts():
    w, s = _pair()
    regular = (w / s).to(E4)                 # weights / scale: the normal 'scaled' FP8
    prescaled = w.to(E4)                     # the weights as they are, the scale supplied on top
    assert not P._stored_at_scale(regular.float(), s, E4)
    assert P._stored_at_scale(prescaled.float(), s, E4)
    # an arbitrary scale on small values (synthetic test data):
    # a ratio far below 1, this is NOT the 'already scaled' case
    assert not P._stored_at_scale((w * 50).to(E4).float(), torch.tensor(0.5), E4)
    # INT8 normal
    s8 = (w.abs().max() / 127.0).reshape(())
    q8 = torch.round(w / s8).clamp(-127, 127).to(torch.int8)
    assert not P._stored_at_scale(q8.float(), s8, torch.int8)
    # a full INT8 (+-127) with a scale close to 1: a ratio of ~1 as well, but the
    # range is FILLED -- it is a normal file (the test_quant_formats case)
    full = torch.randint(-127, 128, (4, 3), dtype=torch.int8)
    full[0, 0] = 127
    assert not P._stored_at_scale(full.float(), torch.full((4, 1), 0.9), torch.int8)
    # MX scales (an E8M0 exponent in a uint8): never concerned
    assert not P._stored_at_scale(prescaled.float(), torch.tensor([120], dtype=torch.uint8), E4)
    assert not P._stored_at_scale(prescaled.float(), s, E4, {"format": "mxfp8"})


def _tiny(path, prescaled, both=False):
    """x_embedder already scaled when `prescaled`; context_embedder too when `both`
    (a real file is homogeneous: the cache key only samples one tensor)."""
    w1, s1 = _pair(32)
    w2, s2 = _pair(32)
    sd = {"x_embedder.weight": (w1 if prescaled else w1 / s1).to(E4),
          "x_embedder.weight_scale": s1.float(),
          "context_embedder.weight": (w2 if both else w2 / s2).to(E4),
          "context_embedder.weight_scale": s2.float(),
          # an Ideogram 4 block name: the loader checks the family before reading anything
          "llm_cond_proj.bias": torch.zeros(2, dtype=torch.bfloat16)}
    save_file(sd, path)
    return w1, w2


def test_the_loader_reads_a_file_mixing_both_layouts():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "mixed.safetensors")
    w1, w2 = _tiny(p, prescaled=True)
    out = P._load_dequant_state_dict(p)
    for k, w in (("x_embedder.weight", w1), ("context_embedder.weight", w2)):
        rel = ((out[k].float() - w).norm() / w.norm()).item()
        assert rel < 0.1, (k, rel)      # without the fix: ~1.0 on x_embedder
