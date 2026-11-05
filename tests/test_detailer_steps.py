"""Detailer steps (cz_detailer): follow the model profile by default (48 on Ideogram), never
raise the slider, a config value overrides."""
# scope: ideogram
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_detailer as D
import cz_pipeline as P


def _with_model(base, transformer, fn):
    old = (P.BASE_REPO, P.ZIMAGE_TRANSFORMER, D.CONFIG.get("detailer_steps"))
    P.BASE_REPO, P.ZIMAGE_TRANSFORMER = base, transformer
    try:
        return fn()
    finally:
        P.BASE_REPO, P.ZIMAGE_TRANSFORMER = old[0], old[1]
        if old[2] is None:
            D.CONFIG.pop("detailer_steps", None)
        else:
            D.CONFIG["detailer_steps"] = old[2]


def test_follows_the_model_profile_by_default():
    """Ideogram 4's profile is 48 steps: a detailer pass never runs more than the model's
    own budget, and never more than the slider."""
    repo = P.engine.DEFAULT_REPO

    def check():
        D.CONFIG.pop("detailer_steps", None)
        assert D._detailer_steps(60) == 48, "Ideogram 4 runs 48 steps"
        assert D._detailer_steps(40) == 40
    _with_model(repo, None, check)
    print("OK test_follows_the_model_profile_by_default")


def test_never_raises_the_slider():
    """Taking the slider below the profile stays the user's choice."""
    def check():
        D.CONFIG.pop("detailer_steps", None)
        assert D._detailer_steps(3) == 3
        assert D._detailer_steps(1) == 1
    _with_model(P.engine.DEFAULT_REPO, None, check)
    print("OK test_never_raises_the_slider")


def test_config_overrides():
    def check():
        D.CONFIG["detailer_steps"] = 8
        assert D._detailer_steps(12) == 8, "un entier positif force la valeur"
        D.CONFIG["detailer_steps"] = -1
        assert D._detailer_steps(12) == 12, "-1 hands control back to the slider"
        D.CONFIG["detailer_steps"] = "anything at all"
        assert D._detailer_steps(60) == 48, "an unreadable value falls back on the profile (48)"
    _with_model(P.engine.DEFAULT_REPO, None, check)
    print("OK test_config_overrides")


if __name__ == "__main__":
    test_follows_the_model_profile_by_default()
    test_never_raises_the_slider()
    test_config_overrides()
    print("ALL OK")
