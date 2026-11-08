"""Lot 8b: single-file Ideogram 4 transformers in the original / ComfyUI layout (CPU).

A tiny Ideogram 4 transformer is written the way the reference code and ComfyUI name it
(fused `attention.qkv`, `attention.o`, an exporter prefix), then read back: the converted
weights must be the original ones bit for bit, and the forward identical. Every refusal
names its cause (quantized export, missing / unknown / duplicated tensors, a fused qkv
that does not split, no profile, a base profile). The app lists, selects and loads it.
Run: python -m pytest tests/test_ideogram_single_file.py -q
"""
import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cz_ideogram as E  # noqa: E402
from ideogram_tiny import tiny_components  # noqa: E402


def _tiny():
    tr = tiny_components(0)["transformer"]
    return tr, dict(tr.config)


def _comfy_state(tr, prefix="model.diffusion_model."):
    """The diffusers state dict renamed / fused the reference-code way."""
    sd = {k: v.clone() for k, v in tr.state_dict().items()}
    out = {}
    for k in list(sd):
        if k.endswith(".attention.to_q.weight"):
            base = k[: -len("to_q.weight")]
            out[prefix + base + "qkv.weight"] = torch.cat(
                [sd[base + "to_q.weight"], sd[base + "to_k.weight"], sd[base + "to_v.weight"]], 0)
        elif k.endswith((".attention.to_k.weight", ".attention.to_v.weight")):
            continue
        elif k.endswith(".attention.to_out.0.weight"):
            out[prefix + k[: -len("to_out.0.weight")] + "o.weight"] = sd[k]
        else:
            out[prefix + k] = sd[k]
    return out


def _write(path, state, metadata=None):
    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in state.items()}, str(path),
              metadata=metadata or {"format": "pt"})
    return str(path)


@pytest.fixture
def comfy(tmp_path):
    tr, cfg = _tiny()
    path = _write(tmp_path / "Ideogram4-tiny_bf16.safetensors", _comfy_state(tr))
    return path, tr, cfg


def test_the_converted_weights_are_the_original_ones(comfy):
    path, tr, cfg = comfy
    plan = E.single_file_plan(path, cfg)
    assert sum(1 for d in plan.values() if len(d) == 3) == cfg["num_layers"]
    loaded = E.load_single_file(path, dtype=torch.float32, explicit_profile="instant",
                                expected_arch=cfg)
    ref = tr.state_dict()
    got = loaded.transformer.state_dict()
    assert set(got) == set(ref)
    for k in ref:
        assert torch.equal(got[k], ref[k]), k
    assert loaded.profile.key == "instant" and loaded.info["fused_qkv"] == cfg["num_layers"]
    # buffers that no checkpoint carries (rotary inv_freq) are real and equal too
    for (n, b), (_, rb) in zip(loaded.transformer.named_buffers(), tr.named_buffers()):
        assert not b.is_meta and torch.equal(b, rb), n


def test_the_forward_is_identical(comfy):
    path, tr, cfg = comfy
    loaded = E.load_single_file(path, dtype=torch.float32, explicit_profile="instant",
                                expected_arch=cfg)
    comps = tiny_components(0)
    from diffusers import Ideogram4Pipeline
    a = E.wrap_distilled(Ideogram4Pipeline(**comps), E.LoadedSource(
        tr, E.PROFILES["instant"], "ref", {}))
    b = E.wrap_distilled(Ideogram4Pipeline(**tiny_components(0)), loaded)
    kw = dict(prompt="a poster", width=256, height=256, num_inference_steps=2,
              output_type="latent")
    la = a(generator=torch.Generator().manual_seed(4), **kw).images
    lb = b(generator=torch.Generator().manual_seed(4), **kw).images
    assert torch.equal(la, lb)


def test_qkv_order_matters():
    """The fused rows are [q; k; v] (reference: view(B, L, 3, heads, head_dim)): a file
    fused in another order loads, but not to the same weights."""
    tr, cfg = _tiny()
    st = _comfy_state(tr)
    k = next(n for n in st if n.endswith("attention.qkv.weight"))
    q, kk, v = st[k].chunk(3, 0)
    assert torch.equal(q, tr.state_dict()[k.replace("model.diffusion_model.", "")
                                         .replace("qkv", "to_q")])


@pytest.mark.parametrize("prefix", ["", "diffusion_model.", "transformer."])
def test_exporter_prefixes_are_removed(tmp_path, prefix):
    tr, cfg = _tiny()
    path = _write(tmp_path / "x.safetensors", _comfy_state(tr, prefix))
    assert len(E.single_file_plan(path, cfg)) == len(_comfy_state(tr, prefix))


def _broken(tmp_path, edit):
    tr, cfg = _tiny()
    st = _comfy_state(tr)
    edit(st)
    return _write(tmp_path / "broken.safetensors", st), cfg


def test_quantized_exports_are_refused_by_the_header(tmp_path):
    def int8(st):
        k = next(n for n in st if n.endswith("attention.qkv.weight"))
        st[k] = st[k].to(torch.int8)
        st[k + "_scale"] = torch.ones(1)
    path, cfg = _broken(tmp_path, int8)
    with pytest.raises(E.SourceError, match="quantized export.*I8"):
        E.single_file_plan(path, cfg)

    def blob(st):
        st["layers.0.attention.o.comfy_quant"] = torch.zeros(4, dtype=torch.float32)
    path, cfg = _broken(tmp_path, blob)
    with pytest.raises(E.SourceError, match="quantization blobs"):
        E.single_file_plan(path, cfg)


def test_missing_unknown_duplicated_and_shapes_are_named(tmp_path):
    def edit(st):
        names = sorted(st)
        st.pop(next(n for n in names if n.endswith("feed_forward.w1.weight")))
        st["model.diffusion_model.extra.weight"] = torch.zeros(2)
        k = next(n for n in names if n.endswith("adaln_proj.weight"))
        st[k.replace("model.diffusion_model.", "")] = st[k].clone()  # same tensor twice
        q = next(n for n in names if n.endswith("attention.qkv.weight"))
        st[q] = st[q][:-1]                                           # 3n - 1 rows
    path, cfg = _broken(tmp_path, edit)
    with pytest.raises(E.SourceError) as e:
        E.single_file_plan(path, cfg)
    msg = str(e.value)
    for word in ("missing", "unknown", "duplicated", "extra.weight", "adaln_proj"):
        assert word in msg, word


def test_the_profile_is_never_read_from_the_name(comfy, tmp_path):
    path, tr, cfg = comfy
    with pytest.raises(E.SourceError, match="no inference profile"):
        E.resolve_file_profile(path)
    with pytest.raises(E.SourceError, match="needs the unconditional"):
        E.resolve_file_profile(path, "base")
    with open(path + E.SINGLE_FILE_SIDECAR, "w") as f:
        json.dump({"profile": "instant"}, f)
    assert E.resolve_file_profile(path).key == "instant"
    with pytest.raises(E.SourceError, match="documented as 'instant'"):
        E.resolve_file_profile(path, "base")


def test_a_refused_file_reads_no_weight(comfy, monkeypatch):
    path, tr, cfg = comfy
    import safetensors
    monkeypatch.setattr(safetensors, "safe_open",
                        lambda *a, **k: pytest.fail("weights read before the checks"))
    with pytest.raises(E.SourceError, match="no inference profile"):
        E.load_single_file(path, expected_arch=cfg)


# --- The app ---------------------------------------------------------------------------
def test_the_app_lists_selects_and_loads_it(comfy, monkeypatch):
    import cz_pipeline as P
    path, tr, cfg = comfy
    monkeypatch.setattr(E, "ARCH", cfg)
    monkeypatch.setattr(P, "IDEOGRAM_PROFILE", None)
    assert P._flux3_file_refusal(path) is None, "the header check passes"
    assert "no inference profile" in P.checkpoint_refusal(path), "selection needs a profile"
    monkeypatch.setattr(P, "IDEOGRAM_PROFILE", "instant")
    assert P.checkpoint_refusal(path) is None
    loaded = P._load_transformer(path)
    assert isinstance(loaded, E.LoadedSource) and loaded.profile.key == "instant"
    assert P._is_distilled_transformer(path)
    monkeypatch.setattr(P, "ZIMAGE_TRANSFORMER", path)
    need = P._base_vram_need_gb()
    assert need == pytest.approx(P._FLUX3_REST_GB + os.path.getsize(path) / 1024 ** 3)


def test_the_ui_takes_the_steps_from_the_profile(comfy, monkeypatch):
    import cz_pipeline as P
    import cz_ui
    path, tr, cfg = comfy
    monkeypatch.setattr(P, "IDEOGRAM_PROFILE", "instant")
    st, g, why = cz_ui._profile_for_checkpoint(path)
    assert (st, g) == (8, 1.0) and "instant" in why


def test_compare_against_a_diffusers_folder(comfy, tmp_path):
    path, tr, cfg = comfy
    folder = tmp_path / "fal" / "transformer"
    tr.save_pretrained(str(folder), max_shard_size="20KB")
    res = E.compare_single_file(path, str(folder), cfg)
    assert res["equal"] == len(tr.state_dict()) and not res["different"] and not res["missing"]
    # a file fused in the wrong order is caught
    st = _comfy_state(tr)
    k = next(n for n in st if n.endswith("attention.qkv.weight"))
    q, kk, v = st[k].chunk(3, 0)
    st[k] = torch.cat([kk, q, v], 0)
    bad = _write(tmp_path / "swapped.safetensors", st)
    res = E.compare_single_file(bad, str(folder), cfg)
    assert {d for d, _ in res["different"]} == {k.replace("model.diffusion_model.", "")
                                                .replace("qkv", f"to_{x}") for x in "qk"}
