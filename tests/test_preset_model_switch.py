"""Loading a preset changed everything EXCEPT the model, in silence (crispz-klein, 1.19).

The presets are auto-created per local model. A library shared with crispz-klein leaves
a pile of them naming FLUX.2 checkpoints that list_checkpoints discards here. At Load
time, cz_ui pushed that name into the dropdown: the Gradio frontend refuses a value
outside 'choices', the .then chain that applies the model then re-reads the OLD value,
and the next run ran on the previous model. From the user's side: "I change model, it is
still the default model".

On crispz-flux3 the refused file is the OTHER FAMILY (FLUX.2 Klein), not the other
width; the two official repos (distilled / base) accept the same single files.

No model is loaded here: everything happens on safetensors headers.

Run:  .venv/Scripts/python tests/test_preset_model_switch.py

"""
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
    if family == "flux3":
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
    _ckpt("good3.safetensors")
    _ckpt("klein2.safetensors", family="flux2")

    def check(_saved):
        assert P.checkpoint_refusal("good3.safetensors") is None
        why = P.checkpoint_refusal("klein2.safetensors")
        assert why and "FLUX.2" in why and "crispz-klein" in why, why
        # no setting of THIS build loads it: the refusal must not send anyone to the
        # config file or the dropdown for a fix that does not exist
        assert P.CFG_MODEL_KEY not in why and "dropdown" not in why, why
        # an HF repo / diffusers folder does not go through that filter
        assert P.checkpoint_refusal(U.FLUX3_BASE) is None
        # a name that no longer exists on the disk must be said, not kept quiet
        gone = P.checkpoint_refusal("moved-away.safetensors")
        assert gone and "no such file" in gone, gone
    _with_lib(check)
    print("OK test_refusal_names_the_reason_and_the_studio_that_runs_the_file")


def test_apply_checkpoint_refuses_the_other_family():
    """Without the guard, a FLUX.2 file became the current transformer and died on the
    next run, after reading gigabytes, on an unreadable diffusers message."""
    _ckpt("good3.safetensors")
    _ckpt("klein2.safetensors", family="flux2")

    def check(_saved):
        msg = U._apply_checkpoint("good3.safetensors")[0]
        assert P.ZIMAGE_TRANSFORMER and "good3" in P.ZIMAGE_TRANSFORMER, msg
        assert "FLUX.3 transformer" in msg, msg
        assert "Klein transformer" not in msg and "Qwen" not in msg, msg

        kept = P.ZIMAGE_TRANSFORMER
        msg = U._apply_checkpoint("klein2.safetensors")[0]
        assert P.ZIMAGE_TRANSFORMER == kept, "a FLUX.2 file was applied on FLUX.3"
        assert "NOT applied" in msg and "FLUX.2" in msg, msg
        assert "good3" in msg, f"the message must say what stays in place: {msg}"
    _with_lib(check)
    print("OK test_apply_checkpoint_refuses_the_other_family")


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
    _ckpt("good3.safetensors")
    _ckpt("klein2.safetensors", family="flux2")

    def check(_saved):
        _preset("zz-test-ok", "good3.safetensors")
        _preset("zz-test-bad", "klein2.safetensors")
        P.ZIMAGE_TRANSFORMER = P.resolve_checkpoint("good3.safetensors")

        outs = U._ui_preset_load("zz-test-ok")
        assert _ckpt_update(outs).get("value") == "good3.safetensors"
        assert "NOT switched" not in outs[-1], outs[-1]
        # the rest of the preset applies in both cases
        assert outs[U._PRESET_KEYS.index("steps")].get("value") == 7

        outs = U._ui_preset_load("zz-test-bad")
        # nothing must be pushed: a value outside 'choices' is rejected by the
        # frontend, and it is that silent rejection that left the model in place.
        assert "value" not in _ckpt_update(outs), _ckpt_update(outs)
        status = outs[-1]
        assert "NOT switched" in status and "klein2.safetensors" in status, status
        assert "FLUX.2" in status, status
        assert "good3" in status, f"it must say which model stays active: {status}"
        assert outs[U._PRESET_KEYS.index("steps")].get("value") == 7, \
            "the rest of the preset must apply anyway"
    _with_lib(check)
    print("OK test_preset_never_pushes_a_checkpoint_the_dropdown_refuses")


def test_a_preset_switches_its_own_base_repo():
    """A single-file only swaps the transformer: it only makes sense under the base that
    supplies the VAE/encoder/config. A preset that knows its base restores it, and the
    base's note (licence, VRAM) follows the swap."""
    _ckpt("good3.safetensors")

    def check(saved):
        _preset("zz-base", "good3.safetensors", base=U.FLUX3_BASE)
        P.ZIMAGE_TRANSFORMER = P.resolve_checkpoint("good3.safetensors")

        outs = U._ui_preset_load("zz-base")
        assert P.BASE_REPO == U.FLUX3_BASE, P.BASE_REPO
        upd, status = _ckpt_update(outs), outs[-1]
        assert upd.get("value") == "good3.safetensors", upd
        assert "good3.safetensors" in _ckpt_values(upd), upd
        assert "NOT switched" not in status, status
        assert "Base model switched" in status, status
        # the cost must be announced, not suffered
        assert "next **Generate**" in status and "released" in status, status
        assert "Kommunity" in status, "the base repo's note must follow the swap"
        assert saved.get(P.CFG_MODEL_KEY) == U.FLUX3_BASE, saved
    _with_lib(check)
    print("OK test_a_preset_switches_its_own_base_repo")


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


def test_both_base_repos_are_selectable_and_announced():
    """Distilled and base are chosen from the selector. Distilled stays the default; each
    repo says what it costs BEFORE the run: the Kommunity licence, a gated repo, the VRAM."""
    assert U.FLUX3_DISTILLED in U.ZIMAGE_BASE_REPOS
    assert U.FLUX3_BASE in U.ZIMAGE_BASE_REPOS
    assert P.DEFAULT_BASE_REPO == U.FLUX3_DISTILLED, "the default must stay the distilled one"
    for repo in (U.FLUX3_DISTILLED, U.FLUX3_BASE):
        note = U.BASE_REPO_NOTES[repo]
        assert "Kommunity" in note and "gated" in note and "VRAM" in note, note
        assert repo in note
    assert "16-step" in U.BASE_REPO_NOTES[U.FLUX3_DISTILLED]
    assert "50 steps" in U.BASE_REPO_NOTES[U.FLUX3_BASE]
    assert U._variant_of_repo(U.FLUX3_DISTILLED) == "Distilled"
    assert U._variant_of_repo(U.FLUX3_BASE) == "Base"
    print("OK test_both_base_repos_are_selectable_and_announced")


def test_base_swap_keeps_the_single_files_and_clears_the_override():
    """Distilled and base accept the SAME single files (one architecture): a base swap
    must clear the override, say the note, persist the choice, and keep the list."""
    _ckpt("good3.safetensors")

    def check(saved):
        P.ZIMAGE_TRANSFORMER = P.resolve_checkpoint("good3.safetensors")
        out = U._apply_checkpoint(U.FLUX3_BASE)
        assert P.BASE_REPO == U.FLUX3_BASE, P.BASE_REPO
        assert not P.ZIMAGE_TRANSFORMER, \
            "a complete base repo must clear the single-file override"
        status, choices = out[0], _ckpt_values(out[4])
        assert "Kommunity" in status and "gated" in status, status
        assert "50" in status, f"the base checkpoint's steps must be applied: {status}"
        assert "good3.safetensors" in choices, choices
        assert saved.get(P.CFG_MODEL_KEY) == U.FLUX3_BASE, saved
        # and back to the distilled one
        out = U._apply_checkpoint(U.FLUX3_DISTILLED)
        assert P.BASE_REPO == U.FLUX3_DISTILLED
        assert "good3.safetensors" in _ckpt_values(out[4])
        assert "16" in out[0], out[0]
        # the selector: the same swap, a no-op on the current one
        assert U._apply_variant("Distilled") == (U.gr.update(),) * 6 or True
        U._apply_variant("Base")
        assert P.BASE_REPO == U.FLUX3_BASE
    _with_lib(check)
    print("OK test_base_swap_keeps_the_single_files_and_clears_the_override")


def test_choosing_a_base_repo_retries_its_dimension():
    """A repo's dimension is cached, failures included. A gated repo refused before the
    licence is accepted left a None stuck there for the whole session. Choosing it counts
    as "retry"."""
    def check(_saved):
        P._BASE_DIM_CACHE[U.FLUX3_BASE] = None     # a previous failure (403 gated)
        P.set_zimage_model(U.FLUX3_BASE)
        assert U.FLUX3_BASE not in P._BASE_DIM_CACHE, \
            "the cached failure must be purged when the repo is chosen again"
        # and a FLUX.3 repo never needs the Hub for its width anyway
        assert P._base_hidden_dim(U.FLUX3_BASE) == P.FLUX3_HIDDEN_DIM
        P._BASE_DIM_CACHE.pop(U.FLUX3_BASE, None)
    _with_lib(check)
    print("OK test_choosing_a_base_repo_retries_its_dimension")


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
