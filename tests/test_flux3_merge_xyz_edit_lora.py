"""The 'Edit LoRA weight' X/Y/Z axis (cz_ui), FLUX.3 path kept for merges from crispz-flux3:
accepted on an edit run, refused with the reason on a txt2img grid, without a selected
edit LoRA or with the checkbox off; a value rewrites only the edit set."""
# scope: flux3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_ui as U        # noqa: E402


def _vals(edit=False):
    """Stand-in for _gen_inputs. edit=True -> a real edit job."""
    v = [None] * 36
    v[U._Q_IDX["prompt"]] = "a cat"
    v[U._Q_IDX["use_input"]] = bool(edit)
    v[U._Q_IDX["input_mode"]] = U.EDIT_MODE if edit else "Image to image"
    if edit:
        v[U._Q_REF_IDX[0]] = object()          # one reference is enough
    return v


def _ms(edit_loras=(("/x/consistence-edit.safetensors", 0.6),), enabled=True):
    return {"base_repo": "r", "transformer": None, "loras": [],
            "edit_loras": [tuple(t) for t in edit_loras],
            "edit_loras_enabled": enabled,
            "sampler": "euler", "schedule": "sgm_uniform"}


# --------------------------------------------------------------------- axe ---

def test_the_axis_exists_and_is_calibrated():
    assert "Edit LoRA weight" in U._XYZ_AXES
    assert U._XYZ_AXES["Edit LoRA weight"]["kind"] == "edit_lora_weight"
    # the calibration offers a sweep centred on 0: negative weights invert
    # the LoRA's effect, and 0 serves as the reference with no effect.
    assert "0" in U._XYZ_CALIB["Edit LoRA weight"]
    assert "-" in U._XYZ_CALIB["Edit LoRA weight"]
    print("OK test_the_axis_exists_and_is_calibrated")


def test_on_an_edit_run_the_weights_are_accepted():
    vals, err = U._xyz_validate_axis("Edit LoRA weight", ["-0.4", "0", "0.4"],
                                     _vals(edit=True), _ms())
    assert err is None, err
    assert vals == [-0.4, 0.0, 0.4], vals
    print("OK test_on_an_edit_run_the_weights_are_accepted")


def test_a_txt2img_grid_is_refused_not_silently_useless():
    """The heart of the matter: on a grid with no editing, the edit set is
    never applied. Rendering N identical images would be worse than an error."""
    _v, err = U._xyz_validate_axis("Edit LoRA weight", ["0.4"], _vals(edit=False), _ms())
    assert err and "EDIT run" in err, err
    assert U.EDIT_MODE in err and "identical" in err, err
    print("OK test_a_txt2img_grid_is_refused_not_silently_useless")


def test_no_edit_lora_selected_is_refused():
    _v, err = U._xyz_validate_axis("Edit LoRA weight", ["0.4"], _vals(edit=True),
                                   _ms(edit_loras=()))
    assert err and "no active edit LoRA" in err, err
    print("OK test_no_edit_lora_selected_is_refused")


def test_the_edit_loras_checkbox_being_off_is_refused():
    """The box unticked = a set remembered but never applied: the same silent trap."""
    _v, err = U._xyz_validate_axis("Edit LoRA weight", ["0.4"], _vals(edit=True),
                                   _ms(enabled=False))
    assert err and "checkbox" in err, err
    print("OK test_the_edit_loras_checkbox_being_off_is_refused")


def test_applying_a_value_rewrites_only_the_edit_set():
    """The weight replaces the one of EVERY edit slot, and the BASE set does not
    move: the two sets are distinct, the axis must not spill onto the other."""
    ms = _ms(edit_loras=(("/a.safetensors", 0.6), ("/b.safetensors", 1.0)))
    ms["loras"] = [("/base.safetensors", 0.9)]
    vals = _vals(edit=True)
    U._xyz_apply("Edit LoRA weight", -0.25, vals, ms)
    assert ms["edit_loras"] == [("/a.safetensors", -0.25), ("/b.safetensors", -0.25)]
    assert ms["loras"] == [("/base.safetensors", 0.9)], ms["loras"]
    print("OK test_applying_a_value_rewrites_only_the_edit_set")


def test_the_base_lora_axis_still_ignores_the_edit_set():
    """The symmetric control: the historical axis must not touch the edit set."""
    ms = _ms()
    ms["loras"] = [("/base.safetensors", 0.9)]
    U._xyz_apply("LoRA weight", 0.3, _vals(), ms)
    assert ms["loras"] == [("/base.safetensors", 0.3)]
    assert ms["edit_loras"] == [("/x/consistence-edit.safetensors", 0.6)]
    print("OK test_the_base_lora_axis_still_ignores_the_edit_set")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("All XYZ edit-LoRA tests passed.")
