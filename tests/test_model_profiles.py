"""The model profiles and presets of crispz-ideogram (rewritten from crispz-flux3's lot B).

The trap this guards: profile_for_model matches by substring, first match wins, and an
unknown name (a local diffusers folder, a renamed mirror) falls back on the default
profile. Inherited from FLUX.3, that default was 16 steps at guidance 1.0: on Ideogram 4
it means a guidance schedule peaking at 1.0, almost no CFG. Every path must give 48 / 7.0.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_core  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "ideogram-ai/ideogram-4-nf4-diffusers"


def _with_profiles(profiles, fn, default=None):
    old, old_d = cz_core.MODEL_PROFILES, cz_core.DEFAULT_MODEL_PROFILE
    cz_core.MODEL_PROFILES = profiles
    if default is not None:
        cz_core.DEFAULT_MODEL_PROFILE = default
    try:
        return fn()
    finally:
        cz_core.MODEL_PROFILES, cz_core.DEFAULT_MODEL_PROFILE = old, old_d


def test_the_built_in_profiles_give_ideogram_48_steps_at_7():
    assert cz_core.profile_for_model(REPO) == (48, 7.0)
    assert cz_core.profile_for_model("cocktailpeanut/ideogram-4-nf4") == (48, 7.0)
    assert cz_core.profile_for_model("D:/models/Ideogram_4_local") == (48, 7.0)
    # an unknown name (a renamed local folder) falls back on the same profile
    assert cz_core.profile_for_model("D:/models/my-poster-model") == (48, 7.0)


def test_an_empty_default_profile_still_gives_48_at_7():
    """A config.txt whose default_model_profile lacks a field: the code fallback is
    Ideogram's, not FLUX.3's 16 / 1.0."""
    got = _with_profiles({}, lambda: cz_core.profile_for_model("mystery"), default={})
    assert got == (48, 7.0)
    got = _with_profiles({"ideogram": {}}, lambda: cz_core.profile_for_model(REPO), default={})
    assert got == (48, 7.0)


def test_config_sample_matches_the_code_and_starts_on_ideogram():
    with open(os.path.join(ROOT, "config-sample.txt"), encoding="utf-8") as f:
        cfg = json.load(f)
    keys = list(cfg["model_profiles"])
    assert "klein" not in keys and "flux-2" not in keys and "flux2" not in keys
    sample = {k: (int(v["steps"]), float(v["guidance"])) for k, v in cfg["model_profiles"].items()}
    built_in = {k: (int(v["steps"]), float(v["guidance"])) for k, v in cz_core.MODEL_PROFILES.items()}
    assert sample == built_in
    assert cfg["default_model_profile"] == {"steps": 48, "guidance": 7.0}
    presets = cfg["performance_presets"]
    assert cfg["default_performance"] in presets
    assert presets[cfg["default_performance"]][:2] == [48, 7.0]
    assert cfg["default_gen_steps"] == 48
    assert cfg["ideogram_model"] == REPO and "klein_model" not in cfg and "flux3_model" not in cfg
    assert cfg["ideogram_base_repos"][0] == REPO and "flux3_base_repos" not in cfg
    assert cfg["ideogram_precision"] == "nf4"
    # crispz_flux3_dir is this studio's own setting (the Edit in crispz-flux3 tab, lot 5),
    # not a FLUX.3 leftover
    for k in cfg:
        if k.lstrip("_").startswith("crispz_flux3_dir"):
            continue
        assert "klein_" not in k and "flux3_" not in k, k
    with open(os.path.join(ROOT, "preferences.example.json"), encoding="utf-8") as f:
        prefs = json.load(f)
    assert prefs["ideogram_model"] == REPO and "flux3_model" not in prefs


def test_the_ui_presets_mirror_the_sample():
    with open(os.path.join(ROOT, "config-sample.txt"), encoding="utf-8") as f:
        cfg = json.load(f)
    import cz_ui
    sample = {k: (int(v[0]), float(v[1])) for k, v in cfg["performance_presets"].items()}
    assert set(sample) >= set(cz_ui.PERFORMANCE) or cz_ui.PERFORMANCE == sample


def test_set_precision_knows_only_nf4_and_persists_the_ideogram_key(monkeypatch):
    import cz_pipeline as P
    lines = []
    monkeypatch.setattr(P, "_log", lambda m: lines.append(str(m)))
    monkeypatch.setattr(P, "PRECISION", "nf4")
    monkeypatch.setattr(P, "ZIMAGE_TRANSFORMER", None)
    assert P.set_precision("nf4") == "nf4"                    # unchanged: nothing said
    assert not lines
    assert P.set_precision("fp8r") == "nf4"                   # unknown -> nf4, said
    assert any("unknown" in m for m in lines)
    # the (hidden) UI handler persists under the key the engine READS back
    import cz_ui
    saved = {}
    monkeypatch.setattr(cz_ui, "_save_prefs_keys", lambda d: saved.update(d))
    msg = cz_ui._ui_set_precision("nf4")
    assert saved == {"ideogram_precision": "nf4"} and "nf4" in msg


def test_refine_steps_follow_ideogram_not_flux3_distilled():
    """Lot 7: the upscale / img2img presets and the refine default were sized for FLUX.3
    distilled (16 steps): 12 to 16. On Ideogram 4 (48) that left 3 or 4 steps actually
    run at denoise 0.3. They are scaled 3x; the config sample and the code agree."""
    import cz_ui
    for name, p in cz_ui.PRESETS.items():
        if "steps" in p:
            assert p["steps"] >= 24, (name, p["steps"])
    with open(os.path.join(ROOT, "config-sample.txt"), encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["default_refine_steps"] == 36
    assert cz_core.DEFAULT_STEPS in (36, int(cz_core.CONFIG.get("default_refine_steps", 36)))
