"""Lot 8a in the app (cz_pipeline / cz_core / cz_cli, CPU): a transformer source is a
transformer change (components kept, pinned revision), the profile's guidance replaces
the slider (said, recorded in the metadata as applied), the UI profile gives 8 / 1.0,
the source route is the checked one, and a contradiction is refused before loading.
Run: python -m pytest tests/test_ideogram_instant_wiring.py -q
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cz_core  # noqa: E402
import cz_ideogram as E  # noqa: E402
import cz_pipeline as P  # noqa: E402
from ideogram_tiny import tiny_components  # noqa: E402

FAL = "fal/ideogram-v4-instant"
CALLS = []


@pytest.fixture
def instant(monkeypatch):
    """The tiny pipeline under the Instant profile, installed as the loaded model."""
    from diffusers import Ideogram4Pipeline
    comps = tiny_components(0)
    comps["unconditional_transformer"] = E.ZeroUnconditionalTransformer(torch.float32)
    pipe = E.IdeogramPipe(Ideogram4Pipeline(**comps), repo=E.DEFAULT_REPO,
                          profile=E.PROFILES["instant"], source=FAL)
    real = E.IdeogramPipe.__call__

    def spy(self, *a, **kw):
        CALLS.append(dict(kw))
        return real(self, *a, **kw)
    monkeypatch.setattr(E.IdeogramPipe, "__call__", spy)
    monkeypatch.setattr(P, "ZIMAGE_TRANSFORMER", FAL)
    monkeypatch.setattr(P, "_BASE_PIPE", pipe)
    monkeypatch.setattr(P, "_DERIVED", {"txt2img": pipe})
    monkeypatch.setattr(P, "_LOADED_KEY", (P.BASE_REPO, FAL, P.OFFLOAD_MODE, P.PRECISION))
    monkeypatch.setattr(P, "_BASE_SCHED_CONFIG", dict(pipe.scheduler.config))
    monkeypatch.setattr(P, "LORAS", [])
    monkeypatch.setattr(P, "EDIT_LORAS", [])
    CALLS.clear()
    yield pipe


def test_the_slider_is_not_applied_and_that_is_said(instant, monkeypatch):
    lines = []
    monkeypatch.setattr(P, "_log", lambda m: lines.append(str(m)))
    monkeypatch.setattr(P, "GUIDANCE", 7.0)
    img = P.generate("a sign that says OPEN", 256, 256, 8, 1)
    assert img.size == (256, 256)
    assert CALLS[-1]["guidance_scale"] == 1.0
    assert any("guidance 7 not applied" in m and "Instant" in m for m in lines)


def test_metadata_carries_the_applied_recipe(instant):
    m = P._gen_meta("txt2img", "x", seed=1, steps=8, guidance=7.0, size=(256, 256))
    assert m["guidance"] == 1.0 and m["guidance_not_applied"] == 7.0
    assert m["ideogram_profile"]["profile"] == "instant"
    assert m["ideogram_profile"]["std"] == 1.75
    assert m["model"] == FAL and m["base_repo"] == P.BASE_REPO
    assert m["components_revision"] == E.COMPONENTS_REVISION


def test_ui_profile_is_8_steps_without_cfg():
    assert cz_core.profile_for_model(FAL) == (8, 1.0)
    assert cz_core.profile_for_model("ideogram-ai/ideogram-4-nf4-diffusers") == (48, 7.0)


def test_a_source_is_a_transformer_change(monkeypatch):
    monkeypatch.setattr(P, "BASE_REPO", E.DEFAULT_REPO)
    monkeypatch.setattr(P, "ZIMAGE_TRANSFORMER", None)
    freed = []
    monkeypatch.setattr(P, "free_vram", lambda: freed.append(1))
    P.set_zimage_model(FAL)
    assert P.ZIMAGE_TRANSFORMER == FAL and P.BASE_REPO == E.DEFAULT_REPO and not freed


def test_the_source_route_is_the_checked_loader(monkeypatch):
    seen = {}

    def fake(src, dtype=None, explicit_profile=None, token=None, expected=None):
        seen.update(src=src, profile=explicit_profile)
        return "loaded"
    monkeypatch.setattr(E, "load_source", fake)
    monkeypatch.setattr(P, "IDEOGRAM_PROFILE", None)
    assert P._load_transformer(FAL) == "loaded"
    assert seen == {"src": FAL, "profile": None}


def test_instant_profile_on_the_base_weights_is_refused(monkeypatch):
    monkeypatch.setattr(P, "IDEOGRAM_PROFILE", "instant")
    monkeypatch.setattr(E, "load_transformer",
                        lambda *a, **k: pytest.fail("loaded the base transformer"))
    with pytest.raises(E.SourceError, match="distilled recipe"):
        P._load_transformer(None)


def test_vram_estimate_counts_one_bf16_transformer(monkeypatch):
    monkeypatch.setattr(P, "ZIMAGE_TRANSFORMER", FAL)
    need = P._base_vram_need_gb()
    assert 22.0 < need < 23.0          # 5.3 (encoder + VAE) + 17.3 GiB


def test_load_wires_components_at_the_pinned_revision(monkeypatch):
    """engine.load with a LoadedSource: the unconditional transformer is not loaded,
    the components come at COMPONENTS_REVISION, the zero branch is registered."""
    from diffusers import Ideogram4Pipeline
    seen = {}
    comps = tiny_components(0)

    def fake_from_pretrained(repo, **kw):
        seen.update(kw, repo=repo)
        c = dict(comps)
        c["transformer"] = kw["transformer"]
        c["unconditional_transformer"] = comps["unconditional_transformer"]
        return Ideogram4Pipeline(**c)
    monkeypatch.setattr(Ideogram4Pipeline, "from_pretrained",
                        staticmethod(fake_from_pretrained))
    src = E.LoadedSource(comps["transformer"], E.PROFILES["instant"], FAL, {})
    pipe = E.load(E.DEFAULT_REPO, dtype=torch.float32, transformer=src)
    assert seen["revision"] == E.COMPONENTS_REVISION
    assert seen["unconditional_transformer"] is None
    assert isinstance(pipe._pipe.unconditional_transformer, E.ZeroUnconditionalTransformer)
    assert pipe.profile.key == "instant" and pipe.source == FAL


def test_the_ui_offers_instant_and_applies_it_as_a_transformer(monkeypatch):
    import cz_ui
    assert FAL in cz_ui.ZIMAGE_BASE_REPOS and FAL in cz_ui.ZIMAGE_TRANSFORMER_SOURCES
    assert "fal/ideogram-v4-fast" not in cz_ui.ZIMAGE_BASE_REPOS      # refused, not offered
    saved = {}
    monkeypatch.setattr(cz_ui, "_save_prefs_keys", lambda d: saved.update(d))
    monkeypatch.setattr(P, "BASE_REPO", E.DEFAULT_REPO)
    monkeypatch.setattr(P, "ZIMAGE_TRANSFORMER", None)
    monkeypatch.setattr(P, "free_vram", lambda: None)
    out = cz_ui._apply_checkpoint(FAL)
    assert P.ZIMAGE_TRANSFORMER == FAL and P.BASE_REPO == E.DEFAULT_REPO
    assert out[1]["value"] == 8 and out[2]["value"] == 1.0
    assert "8 steps, no CFG" in out[0] and "NON-COMMERCIAL" in out[0]
    assert saved == {P.CFG_MODEL_KEY: FAL}
    # back to the base model: the source is dropped, the base saved
    cz_ui._apply_checkpoint(E.DEFAULT_REPO)
    assert P.ZIMAGE_TRANSFORMER is None and saved == {P.CFG_MODEL_KEY: E.DEFAULT_REPO}


def test_xyz_checkpoint_axis_keeps_the_base_for_a_source():
    import cz_ui
    key = next(k for k, v in cz_ui._XYZ_AXES.items() if (v or {}).get("kind") == "checkpoint")
    ms = {"base_repo": E.DEFAULT_REPO, "transformer": None}
    cz_ui._xyz_apply(key, FAL, [], ms)
    assert ms == {"base_repo": P.DEFAULT_BASE_REPO, "transformer": FAL}
