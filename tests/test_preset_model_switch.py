"""Loading a preset switches the model for real (cz_ui + cz_core + cz_pipeline): one base
repo, every single file refused with its own reason (a FLUX.2 / FLUX.3 file names its
studio), a preset naming a foreign file never pushes a value outside the dropdown choices."""
# scope: ideogram
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from safetensors.torch import save_file

import cz_core
import cz_pipeline as P
import cz_ui as U

TMP = os.path.join(os.environ.get("TEMP") or "/tmp", "cz_preset_switch")
os.makedirs(TMP, exist_ok=True)


def _ckpt(name, family="flux3"):
    """A fake transformer: only the header is read, the weights are empty. The block
    names carry the family."""
    p = os.path.join(TMP, name)
    if family == "ideogram":
        sd = {"llm_cond_proj.weight": torch.zeros(2, 2),
              "embed_image_indicator.weight": torch.zeros(2, 2)}
    elif family == "flux3":
        sd = {"early_stream_modulations.0.lin.weight": torch.zeros(2, 2),
              "mode_blocks.0.img_attn.qkv.weight": torch.zeros(2, 2)}
    else:
        sd = {"double_stream_modulation_img.lin.weight": torch.zeros(2, 2),
              "double_blocks.0.img_attn.qkv.weight": torch.zeros(2, 2)}
    save_file(sd, p)
    return p


def _with_lib(fn):
    """Runs fn with TMP as the checkpoints AND presets folder, on the distilled base
    repo, with no preference written. The install's presets/ folder must see nothing of
    these fixtures: _apply_checkpoint and _refresh_checkpoints create presets per local
    model."""
    old = (P.CHECKPOINTS_DIR, P.CHECKPOINTS_EXTRA_DIR, P.BASE_REPO,
           P.ZIMAGE_TRANSFORMER, U._PRESETS_DIR, U._save_prefs_keys)
    P.CHECKPOINTS_DIR, P.CHECKPOINTS_EXTRA_DIR = TMP, ""
    P.BASE_REPO = U.FLUX3_DISTILLED
    P.ZIMAGE_TRANSFORMER = None
    U._PRESETS_DIR = os.path.join(TMP, "presets")
    saved = {}
    U._save_prefs_keys = saved.update
    shutil.rmtree(U._PRESETS_DIR, ignore_errors=True)
    os.makedirs(U._PRESETS_DIR, exist_ok=True)
    P._FILE_INFO_CACHE.clear()
    try:
        return fn(saved)
    finally:
        (P.CHECKPOINTS_DIR, P.CHECKPOINTS_EXTRA_DIR, P.BASE_REPO,
         P.ZIMAGE_TRANSFORMER, U._PRESETS_DIR, U._save_prefs_keys) = old


def test_refusal_names_the_reason_and_the_studio_that_runs_the_file():
    _ckpt("ideo.safetensors", family="ideogram")
    _ckpt("flux3.safetensors", family="flux3")
    _ckpt("klein2.safetensors", family="flux2")

    def check(_saved):
        why = P.checkpoint_refusal("ideo.safetensors")
        assert why and "single-file Ideogram 4" in why, why
        why = P.checkpoint_refusal("flux3.safetensors")
        assert why and "FLUX.3" in why and "crispz-flux3" in why, why
        why = P.checkpoint_refusal("klein2.safetensors")
        assert why and "FLUX.2" in why and "crispz-klein" in why, why
        # no setting of THIS build loads them: the refusal must not send anyone to the
        # config file or the dropdown for a fix that does not exist
        assert P.CFG_MODEL_KEY not in why and "dropdown" not in why, why
        # an HF repo / diffusers folder does not go through that filter
        assert P.checkpoint_refusal(U.FLUX3_BASE) is None
        # a name that no longer exists on the disk must be said, not kept quiet
        gone = P.checkpoint_refusal("moved-away.safetensors")
        assert gone and "no such file" in gone, gone
    _with_lib(check)


def test_apply_checkpoint_refuses_a_single_file():
    """Without the guard, a file of the other family became the current transformer and
    died on the next run, after reading gigabytes, on an unreadable diffusers message."""
    _ckpt("flux3.safetensors", family="flux3")
    _ckpt("ideo.safetensors", family="ideogram")

    def check(_saved):
        for name, needle in (("flux3.safetensors", "crispz-flux3"),
                             ("ideo.safetensors", "single-file Ideogram 4")):
            msg = U._apply_checkpoint(name)[0]
            assert not P.ZIMAGE_TRANSFORMER, f"{name} was applied"
            assert P.BASE_REPO == U.FLUX3_DISTILLED
            assert "NOT applied" in msg and needle in msg, msg
            assert P.BASE_REPO in msg, f"the message must say what stays in place: {msg}"
    _with_lib(check)


def _preset(name, ckpt, steps=7, base=None):
    os.makedirs(U._PRESETS_DIR, exist_ok=True)
    p = os.path.join(U._PRESETS_DIR, name + ".json")
    d = {"prompt": "a lighthouse", "steps": steps, "checkpoint": ckpt,
         "transformer": "", "loras": []}
    if base:
        d["base_repo"] = base
    with open(p, "w", encoding="utf-8") as f:
        json.dump(d, f)
    return p


def _ckpt_values(upd):
    """The menu carries (label, value) pairs since the family/size labels: what a
    preset, a queued job or the frontend matches on is the VALUE."""
    return [v if isinstance(v, str) else v[1] for v in (upd.get("choices") or [])]


def _ckpt_update(outs):
    """The checkpoint dropdown's gr.update in _ui_preset_load's output."""
    return outs[U._PRESET_KEYS.index("checkpoint")]


def test_preset_never_pushes_a_checkpoint_the_dropdown_refuses():
    _ckpt("klein2.safetensors", family="flux2")

    def check(_saved):
        _preset("zz-test-ok", "")
        _preset("zz-test-bad", "klein2.safetensors")

        outs = U._ui_preset_load("zz-test-ok")
        assert "NOT switched" not in outs[-1], outs[-1]
        assert outs[U._PRESET_KEYS.index("steps")].get("value") == 7

        outs = U._ui_preset_load("zz-test-bad")
        # nothing must be pushed: a value outside 'choices' is rejected by the
        # frontend, and it is that silent rejection that left the model in place.
        assert "value" not in _ckpt_update(outs), _ckpt_update(outs)
        status = outs[-1]
        assert "NOT switched" in status and "klein2.safetensors" in status, status
        assert "FLUX.2" in status, status
        assert P.BASE_REPO in status, f"it must say which model stays active: {status}"
        assert outs[U._PRESET_KEYS.index("steps")].get("value") == 7, \
            "the rest of the preset must apply anyway"
    _with_lib(check)


def test_a_preset_switches_its_own_base_repo():
    """A preset that knows its base restores it, and the base's note (the NON-COMMERCIAL
    licence, the gated repo) follows the swap. Here from a mirror back to the official
    repo."""
    def check(saved):
        _preset("zz-base", "", base=U.FLUX3_BASE)
        P.BASE_REPO = "someone/ideogram-4-mirror"

        outs = U._ui_preset_load("zz-base")
        status = outs[-1]
        assert P.BASE_REPO == U.FLUX3_BASE, P.BASE_REPO
        assert "NOT switched" not in status, status
        assert "Base model switched" in status, status
        assert "next **Generate**" in status and "released" in status, status
        assert "NON-COMMERCIAL" in status, "the base repo's note must follow the swap"
        assert saved.get(P.CFG_MODEL_KEY) == U.FLUX3_BASE, saved
    _with_lib(check)


def test_the_refusal_leads_with_the_action():
    """Someone who has just clicked Load is looking for the action, not the diagnosis.
    For a FLUX.2 file the action is NOT a base switch (none of this build's bases loads
    it): it is to pick a FLUX.3 checkpoint and repair the preset."""
    _ckpt("good3.safetensors")
    _ckpt("klein2.safetensors", family="flux2")

    def check(_saved):
        # a preset WITHOUT base_repo (written before the base was tracked)
        _preset("zz-old", "klein2.safetensors")
        status = U._ui_preset_load("zz-old")[-1]
        head = status.split("Why:")[0]
        assert "crispz-klein" in head and "FLUX.2" in head, f"the action must come first: {head}"
        assert "Pick `" not in head, f"no base of this build loads a FLUX.2 file: {head}"
        assert "Why:" in status, "the diagnosis must follow, not precede"
        # and the lasting repair of the old preset
        assert "Update selected" in status, status
    _with_lib(check)
    print("OK test_the_refusal_leads_with_the_action")


def test_saving_records_the_base_repo():
    """Without that, every preset created today reproduces tomorrow's bug."""
    _ckpt("good3.safetensors")

    def check(_saved):
        vals = [""] * len(U._PRESET_KEYS)
        vals[U._PRESET_KEYS.index("checkpoint")] = "good3.safetensors"
        U._ui_preset_save("zz-saved", *(vals + ["None"] * U.MAX_LORA_SLOTS
                                        + [1.0] * U.MAX_LORA_SLOTS))
        assert U._load_preset_file("zz-saved").get("base_repo") == U.FLUX3_DISTILLED
        # and an auto-created preset as well
        U._ensure_model_presets(["good3.safetensors"])
        assert U._load_preset_file("good3").get("base_repo") == U.FLUX3_DISTILLED
    _with_lib(check)
    print("OK test_saving_records_the_base_repo")


def test_the_base_repo_is_announced_before_the_run():
    """One Ideogram 4 checkpoint (the official nf4 repo): it is offered, it is the
    default, and it says what it costs BEFORE the run: the NON-COMMERCIAL licence, the
    gated repo, the steps."""
    repo = U.FLUX3_DISTILLED
    assert repo == U.FLUX3_BASE == P.DEFAULT_BASE_REPO
    assert repo in U.ZIMAGE_BASE_REPOS
    note = U.BASE_REPO_NOTES[repo]
    assert "NON-COMMERCIAL" in note and "gated" in note and "48 steps" in note, note
    assert repo in note


def test_choosing_a_base_repo_retries_its_dimension():
    """A repo's dimension is cached, failures included. A gated repo refused before the
    licence is accepted left a None stuck there for the whole session. Choosing it counts
    as "retry"."""
    def check(_saved):
        P._BASE_DIM_CACHE[U.FLUX3_BASE] = None     # a previous failure (403 gated)
        P.set_zimage_model(U.FLUX3_BASE)
        assert U.FLUX3_BASE not in P._BASE_DIM_CACHE, \
            "the cached failure must be purged when the repo is chosen again"
        # and an Ideogram repo never needs the Hub for its width anyway: 4608
        assert P._base_hidden_dim(U.FLUX3_BASE) == 4608
        P._BASE_DIM_CACHE.pop(U.FLUX3_BASE, None)
    _with_lib(check)


def test_offload_is_forced_when_the_base_cannot_fit():
    """klein-9B set whole on a 32 GB card loaded, then died on the first diffusion step
    on 'CUDA error: unknown error', which does not even name the VRAM. A config that
    CANNOT work is corrected beforehand. On FLUX.3 the weight depends on the precision."""
    old = (P.DEVICE, P._total_vram_gb, P.OFFLOAD_MODE, P.BASE_REPO, P.ZIMAGE_TRANSFORMER,
           P.PRECISION)
    try:
        P.DEVICE = "cuda"
        P.OFFLOAD_MODE = "none"
        P.BASE_REPO = U.FLUX3_DISTILLED
        P.ZIMAGE_TRANSFORMER = None
        P.PRECISION = "bf16"
        P._total_vram_gb = lambda: 31.8        # RTX 5090
        assert P._effective_offload() == "none", "bf16 fits a 32 GB card whole"
        P._total_vram_gb = lambda: 16.0        # RTX 5080 / 4080
        assert P._effective_offload() == "model", "bf16 does not fit 16 GB: offload forced"
        P.PRECISION = "nvfp4"
        assert P._effective_offload() == "model", "13.5 + 4 GB of margin: still not 16"
        # a GGUF override is always streamed (it does not move onto the GPU with .to)
        P._total_vram_gb = lambda: 80.0
        P.PRECISION = "bf16"
        assert P._effective_offload() == "none"
        P.ZIMAGE_TRANSFORMER = os.path.join(TMP, "flux3-Q4.gguf")
        assert P._effective_offload() == "model"
    finally:
        (P.DEVICE, P._total_vram_gb, P.OFFLOAD_MODE, P.BASE_REPO, P.ZIMAGE_TRANSFORMER,
         P.PRECISION) = old
    print("OK test_offload_is_forced_when_the_base_cannot_fit")


def test_gated_repo_error_says_what_to_do():
    """A 401/403 from the Hub must become an instruction, not a trace. Both FLUX.3 repos
    are gated."""
    hint = P._hf_access_hint(U.FLUX3_DISTILLED,
                             RuntimeError("401 Client Error: Access to model ... is restricted"))
    assert hint and U.FLUX3_DISTILLED in hint, hint
    assert "accept its licence" in hint and "token" in hint, hint
    # a token set by 'huggingface-cli login' counts: telling someone who has one
    # that there is "no token" sends them looking for the wrong cause.
    real, cz_core.hf_token_is_set = cz_core.hf_token_is_set, lambda: True
    try:
        with_tok = P._hf_access_hint(U.FLUX3_BASE, RuntimeError("403 gated"))
    finally:
        cz_core.hf_token_is_set = real
    assert "licence itself" in with_tok, with_tok
    # an ordinary failure must NOT be dressed up as a licence problem
    assert P._hf_access_hint(U.FLUX3_BASE, OSError("disk full")) is None
    print("OK test_gated_repo_error_says_what_to_do")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL OK")
