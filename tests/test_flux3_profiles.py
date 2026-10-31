"""Lot B: the model profiles and presets of crispz-flux3.

The trap this guards: profile_for_model matches by substring, first match wins, and
crispz-klein's 'klein' key (4 steps) is a substring of 'flux-3-image-klein'. Through the
built-in defaults OR a config.txt copied from crispz-klein, FLUX.3 must still get its 16
steps (distilled) or 50 steps + CFG 3.5 (base).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_core  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DISTILLED = "black-forest-labs/flux-3-image-klein"
BASE = "black-forest-labs/flux-3-image-klein-base"


def _with_profiles(profiles, fn, default=None):
    old, old_d = cz_core.MODEL_PROFILES, cz_core.DEFAULT_MODEL_PROFILE
    cz_core.MODEL_PROFILES = profiles
    if default is not None:
        cz_core.DEFAULT_MODEL_PROFILE = default
    try:
        return fn()
    finally:
        cz_core.MODEL_PROFILES, cz_core.DEFAULT_MODEL_PROFILE = old, old_d


def test_the_built_in_profiles_know_both_flux3_checkpoints():
    assert cz_core.profile_for_model(DISTILLED) == (16, 1.0)
    assert cz_core.profile_for_model(BASE) == (50, 3.5)
    # mirrors and community single files
    assert cz_core.profile_for_model("unsloth/flux-3-image-klein-GGUF") == (16, 1.0)
    assert cz_core.profile_for_model("flux3-klein-Q4_K_M.gguf") == (16, 1.0)
    assert cz_core.profile_for_model("FLUX_3_image_klein_base_fp8.safetensors") == (50, 3.5)
    # an unknown name falls back to the distilled defaults, never to 4 steps
    assert cz_core.profile_for_model("mystery.safetensors") == (16, 1.0)


def test_a_klein_config_cannot_hand_flux3_the_four_step_profile(monkeypatch):
    lines = []
    monkeypatch.setattr(cz_core, "_log", lambda m: lines.append(str(m)))
    cz_core._FLUX2_PROFILE_SAID.clear()
    klein_cfg = {"klein": {"steps": 4, "guidance": 1.0}, "flux-2": {"steps": 4, "guidance": 1.0}}
    got = _with_profiles(klein_cfg, lambda: cz_core.profile_for_model(DISTILLED),
                         default={"steps": 16, "guidance": 1.0})
    assert got == (16, 1.0)
    assert sum("crispz-klein" in m for m in lines) == 1
    _with_profiles(klein_cfg, lambda: cz_core.profile_for_model(DISTILLED))
    assert sum("crispz-klein" in m for m in lines) == 1          # said once
    # the same key still serves a real FLUX.2 file name (a shared library)
    got = _with_profiles(klein_cfg, lambda: cz_core.profile_for_model("FLUX.2-klein-4B.safetensors"))
    assert got == (4, 1.0)


def test_config_sample_orders_base_before_distilled_and_matches_the_code():
    with open(os.path.join(ROOT, "config-sample.txt"), encoding="utf-8") as f:
        cfg = json.load(f)
    keys = list(cfg["model_profiles"])
    assert keys.index("flux-3-image-klein-base") < keys.index("flux-3-image-klein")
    assert "klein" not in keys and "flux-2" not in keys and "flux2" not in keys
    sample = {k: (int(v["steps"]), float(v["guidance"])) for k, v in cfg["model_profiles"].items()}
    built_in = {k: (int(v["steps"]), float(v["guidance"])) for k, v in cz_core.MODEL_PROFILES.items()}
    assert sample == built_in
    assert cfg["default_model_profile"] == {"steps": 16, "guidance": 1.0}
    presets = cfg["performance_presets"]
    assert cfg["default_performance"] in presets
    assert presets[cfg["default_performance"]][:2] == [16, 1.0]
    assert any(v[:2] == [50, 3.5] for v in presets.values())
    assert cfg["default_gen_steps"] == 16
    assert cfg["flux3_model"] == DISTILLED and "klein_model" not in cfg
    assert cfg["flux3_base_repos"] == [DISTILLED, BASE] and "klein_base_repos" not in cfg
    assert cfg["flux3_precision"] in ("bf16", "fp8r", "nvfp4")
    for k in cfg:
        assert "klein_" not in k, k        # no crispz-klein key survives in the sample
    with open(os.path.join(ROOT, "preferences.example.json"), encoding="utf-8") as f:
        prefs = json.load(f)
    assert prefs["flux3_model"] == DISTILLED and "klein_model" not in prefs


def test_the_ui_presets_mirror_the_sample_and_the_edit_catalogue_is_flux3_only():
    import cz_edit_loras
    # One shipped preset (the CivitAI Outpaint LoRA, 2026-10-28), no FLUX.2 leftover
    assert list(cz_edit_loras.EDIT_LORA_SPECS) == [cz_edit_loras.OUTPAINT_PRESET]
    assert all(isinstance(n, str) for n in cz_edit_loras.names())   # config may add some
    assert not any("klein" in str(s.get("base", "")) for s in cz_edit_loras.SPECS.values())
    with open(os.path.join(ROOT, "config-sample.txt"), encoding="utf-8") as f:
        cfg = json.load(f)
    # cz_ui's built-in PERFORMANCE is the fallback when config has none: the same table
    import cz_ui
    sample = {k: (int(v[0]), float(v[1])) for k, v in cfg["performance_presets"].items()}
    assert set(sample) >= set(cz_ui.PERFORMANCE) or cz_ui.PERFORMANCE == sample


def test_set_precision_changes_the_cache_key_and_is_said(monkeypatch):
    import cz_pipeline as P
    lines = []
    monkeypatch.setattr(P, "_log", lambda m: lines.append(str(m)))
    monkeypatch.setattr(P, "PRECISION", "bf16")
    monkeypatch.setattr(P, "ZIMAGE_TRANSFORMER", None)
    assert P.set_precision("fp8r") == "fp8r" and P.PRECISION == "fp8r"
    assert any("transformer swap" in m for m in lines)
    assert P.set_precision("fp8r") == "fp8r"                  # unchanged: nothing said
    assert sum("precision" in m for m in lines) == 1
    assert P.set_precision("int2") == "bf16"                  # unknown -> the safe build, said
    assert any("unknown" in m for m in lines)
    # with a single-file transformer picked, the choice is noted but announced as pending
    monkeypatch.setattr(P, "ZIMAGE_TRANSFORMER", "C:/m/x.safetensors")
    assert P.set_precision("nvfp4") == "nvfp4"
    assert any("single-file transformer" in m for m in lines)
    # the UI handler persists and reports, without touching the pipeline here
    import cz_ui
    saved = {}
    monkeypatch.setattr(cz_ui, "_save_prefs_keys", lambda d: saved.update(d))
    monkeypatch.setattr(P, "ZIMAGE_TRANSFORMER", None)
    msg = cz_ui._ui_set_precision("bf16")
    assert saved == {"flux3_precision": "bf16"} and "bf16" in msg
    assert cz_ui._variant_of_repo("black-forest-labs/flux-3-image-klein-base") == "Base"
    assert cz_ui._variant_of_repo("black-forest-labs/flux-3-image-klein") == "Distilled"
