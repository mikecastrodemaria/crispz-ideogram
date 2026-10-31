"""Edit-LoRA presets on crispz-flux3.

The catalogue ships ONE preset, the CivitAI "FLUX 3 Image Klein Outpaint LoRA"
(2026-10-28); crispz-klein's two Consistence-Edit files are FLUX.2 (refused by family).
What these tests lock down:
  - the shipped preset: its trigger sentence, the author's weight, a CivitAI source that
    downloads through cz_civitai (SHA256 carried) and lands under the stable name;
  - a preset declared in config.txt ('edit_loras') is picked up, by name and adapter name;
  - a FLUX.2 edit LoRA is refused by the family guard with one sentence;
  - picking a preset in the UI applies it at ITS advised weight and moves the slider.
"""
import os
import sys
import shutil
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import cz_edit_loras  # noqa: E402

SPEC = {"repo": "someone/flux3-edit-lora", "weights": "f3_relight.safetensors",
        "adapter_name": "relight-f3", "prompt": "Relight the scene with a warm sunset.",
        "inputs": 1, "base": "flux3", "weight": 0.7}


class _Declared:
    """A preset declared through config 'edit_loras', the way a user would."""

    def __enter__(self):
        self.old = cz_edit_loras.SPECS
        cz_edit_loras.SPECS = cz_edit_loras._apply_config_overrides({})
        cz_edit_loras.CONFIG["edit_loras"] = {"Relight F3": dict(SPEC)}
        cz_edit_loras.SPECS = cz_edit_loras._apply_config_overrides({})
        return self

    def __exit__(self, *exc):
        cz_edit_loras.CONFIG.pop("edit_loras", None)
        cz_edit_loras.SPECS = self.old
        return False


def test_the_shipped_catalogue_is_the_outpaint_lora():
    assert list(cz_edit_loras.EDIT_LORA_SPECS) == [cz_edit_loras.OUTPAINT_PRESET]
    s = cz_edit_loras.spec("Outpaint")
    assert s["prompt"] == cz_edit_loras.OUTPAINT_PROMPT and "solid gray" in s["prompt"]
    assert float(s["weight"]) == 1.5                  # the author's figure
    assert s["civitai"]["versionId"] == 3407881 and len(s["civitai"]["sha256"]) == 64
    assert s["weights"] in s["local_names"]           # a hand-downloaded copy is found
    assert cz_edit_loras.local_path("Outpaint").endswith("flux3-outpaint.safetensors")
    assert cz_edit_loras.speed_names() == []          # no Lightning LoRA either


def test_a_civitai_preset_downloads_through_cz_civitai():
    """The CivitAI source: cz_civitai.download_model_file gets the spec's candidate (the
    version id and the SHA256 to check), and the file is renamed to the stable
    <adapter_name>.safetensors so local_path() finds it next time."""
    import cz_civitai
    tmp = tempfile.mkdtemp()
    seen = {}

    def fake_download(cand, dest_dir, api_key=None, progress=None):
        seen["cand"] = dict(cand)
        p = os.path.join(dest_dir, cand["fileName"])
        open(p, "wb").close()
        return {"success": True, "message": "ok", "path": p}

    real = (cz_civitai.download_model_file, cz_edit_loras.lora_dirs, cz_edit_loras.edit_loras_dir)
    cz_civitai.download_model_file = fake_download
    cz_edit_loras.lora_dirs = lambda: [os.path.join(tmp, "loras")]
    cz_edit_loras.edit_loras_dir = lambda: os.path.join(tmp, "loras", "_hf-edit")
    try:
        assert not cz_edit_loras.is_downloaded("Outpaint")
        got = cz_edit_loras.resolve("Outpaint")
        assert os.path.samefile(got, cz_edit_loras.local_path("Outpaint"))
        assert seen["cand"]["versionId"] == 3407881 and seen["cand"]["sha256"]
        assert cz_edit_loras.is_downloaded("Outpaint")
        # a failed download is a clean error with CivitAI's message, no half file
        cz_civitai.download_model_file = lambda *a, **k: {"success": False, "message": "401"}
        os.remove(got)
        try:
            cz_edit_loras.resolve("Outpaint")
        except RuntimeError as e:
            assert "401" in str(e)
        else:
            raise AssertionError("a failed download must raise")
    finally:
        (cz_civitai.download_model_file, cz_edit_loras.lora_dirs,
         cz_edit_loras.edit_loras_dir) = real
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_config_declared_preset_is_picked_up():
    with _Declared():
        assert "Relight F3" in cz_edit_loras.names()
        s = cz_edit_loras.spec("Relight F3")
        assert s["repo"] == SPEC["repo"] and float(s["weight"]) == 0.7
        assert cz_edit_loras.spec("relight-f3") is s      # by adapter_name too
        assert cz_edit_loras.spec("nope") is None
        assert cz_edit_loras.catalog()                  # it is announced to the protocol


def test_a_local_copy_is_used_without_downloading():
    tmp = tempfile.mkdtemp()
    main, extra = os.path.join(tmp, "loras"), os.path.join(tmp, "civitai")
    os.makedirs(main)
    os.makedirs(extra)
    copy = os.path.join(extra, SPEC["weights"])
    open(copy, "wb").close()
    real = (cz_edit_loras.lora_dirs, cz_edit_loras.edit_loras_dir, cz_edit_loras._download)

    def no_download(*a, **kw):
        raise AssertionError("a local copy exists: nothing must be downloaded")

    cz_edit_loras.lora_dirs = lambda: [main, extra]
    cz_edit_loras.edit_loras_dir = lambda: os.path.join(main, "_hf-edit")
    cz_edit_loras._download = no_download
    try:
        with _Declared():
            assert cz_edit_loras.is_downloaded("Relight F3")
            assert os.path.samefile(cz_edit_loras.resolve("Relight F3"), copy)
            assert cz_edit_loras.status_label("Relight F3").endswith("✓")
    finally:
        cz_edit_loras.lora_dirs, cz_edit_loras.edit_loras_dir, cz_edit_loras._download = real
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_flux2_edit_lora_is_refused_by_family():
    import cz_pipeline
    tmp = tempfile.mkdtemp()
    try:
        l2 = os.path.join(tmp, "consistence_edit_flux2.safetensors")
        save_file({"diffusion_model.double_blocks.0.img_attn.proj.lora_A.weight": torch.zeros(4, 3072),
                   "diffusion_model.double_blocks.0.img_attn.proj.lora_B.weight": torch.zeros(3072, 4)}, l2)
        why = cz_pipeline._lora_unsupported(l2)
        assert why and "FLUX.2" in why and "crispz-klein" in why, why
        l3 = os.path.join(tmp, "relight_flux3.safetensors")
        save_file({"transformer.mode_blocks.0.attn.to_q.lora_A.weight": torch.zeros(4, 3072),
                   "transformer.mode_blocks.0.attn.to_q.lora_B.weight": torch.zeros(3072, 4)}, l3)
        assert cz_pipeline._lora_unsupported(l3) is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_picking_a_preset_sets_its_weight_on_the_slider():
    import cz_ui
    import cz_pipeline
    applied = []
    real = (cz_pipeline.set_edit_loras, cz_edit_loras.is_downloaded, cz_ui._edit_lora_choices)
    cz_pipeline.set_edit_loras = lambda slots: applied.append(list(slots))
    cz_edit_loras.is_downloaded = lambda name, index=None: True
    cz_ui._edit_lora_choices = lambda: ["None", "Relight F3 ✓"]
    try:
        with _Declared():
            dd, note, slider = cz_ui._ui_edit_lora_pick("Relight F3 ✓", 1.0,
                                                        progress=lambda *a, **k: None)
            assert applied == [[("Relight F3", 0.7)]], applied
            assert slider.get("value") == 0.7, slider
            assert "0.7" in note, note
            # "None": nothing to apply, the slider does not move.
            dd, note, slider = cz_ui._ui_edit_lora_pick("None", 0.8, progress=lambda *a, **k: None)
            assert applied[-1] == [] and "value" not in slider, (applied, slider)
    finally:
        cz_pipeline.set_edit_loras, cz_edit_loras.is_downloaded, cz_ui._edit_lora_choices = real


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"OK {fn.__name__}")
    print(f"All {len(tests)} edit-LoRA preset tests passed.")
