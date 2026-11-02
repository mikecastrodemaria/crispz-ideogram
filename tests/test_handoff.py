"""Lot 5: Edit in crispz-flux3 (cz_handoff). The image goes to crispz-flux3's own protocol
v1 CLI on this machine; nothing leaves it. A fake crispz-flux3 folder answers like the
real one (a tiny cz_protocol.py run by a real subprocess)."""
import json
import os
import sys
import textwrap

import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_handoff as H  # noqa: E402

FAKE_CZP = textwrap.dedent('''
    import json, os, sys
    from PIL import Image
    args = sys.argv[1:]
    mode = open("mode.txt").read().strip() if os.path.exists("mode.txt") else "running"
    if args[0] == "caps":
        tool = "crispz-flux3" if mode != "wrong" else "crispz-klein"
        inst = {"running": mode == "running", "url": "http://127.0.0.1:7861",
                "tool": "crispz-flux3"}
        if mode == "clash":
            inst = {"running": True, "url": "http://127.0.0.1:7860", "tool": "crispz-ideogram"}
        print(json.dumps({"ok": True, "tool": tool, "instance": inst}))
        sys.exit(0)
    spec = json.loads(sys.stdin.read())
    open("last_call.json", "w").write(json.dumps({"args": args, "spec": spec}))
    if mode == "fail":
        print(json.dumps({"ok": False, "tool": "crispz-flux3", "error": "out of memory"}))
        sys.exit(1)
    src = Image.open(spec.get("input") or spec["refs"][0])
    out = os.path.join(spec["out_dir"], "edited.png")
    src.transpose(Image.FLIP_LEFT_RIGHT).save(out)
    print("loading...")
    print(json.dumps({"ok": True, "tool": "crispz-flux3", "images": [out],
                      "warnings": ["guidance ignored"]}))
''')


@pytest.fixture
def flux3(tmp_path, monkeypatch):
    d = tmp_path / "crispz-flux3"
    d.mkdir()
    (d / "cz_protocol.py").write_text(FAKE_CZP)
    # its .venv is this interpreter (no symlink: Windows would need privileges for one)
    monkeypatch.setattr(H, "_python", lambda folder: sys.executable)
    monkeypatch.setenv("CRISPZ_FLUX3_DIR", str(d))
    return d


@pytest.fixture
def logs(monkeypatch):
    lines = []
    monkeypatch.setattr(H, "_log", lambda m: lines.append(str(m)))
    return lines


def _img():
    im = Image.new("RGB", (32, 16), (200, 0, 0))
    im.paste((0, 0, 200), (0, 0, 8, 16))
    return im


def _last(d):
    return json.loads((d / "last_call.json").read_text())


def test_status_names_what_is_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("CRISPZ_FLUX3_DIR", raising=False)
    monkeypatch.setitem(H._prefs, "crispz_flux3_dir", "")
    assert "not set" in H.status()[1]
    monkeypatch.setenv("CRISPZ_FLUX3_DIR", str(tmp_path / "nope"))
    assert "not found" in H.status()[1]
    monkeypatch.setenv("CRISPZ_FLUX3_DIR", str(tmp_path))
    assert "cz_protocol.py" in H.status()[1]
    (tmp_path / "cz_protocol.py").write_text("")
    assert ".venv" in H.status()[1]


def test_a_running_flux3_gets_the_edit_through_its_instance(flux3, tmp_path, logs):
    freed = []
    imgs, res, route = H.edit(_img(), "make it blue", refs=[_img()], seed=5,
                              output_dir=str(tmp_path / "out"), free_vram=lambda: freed.append(1))
    assert route == "instance" and freed == [], "a warm instance needs no VRAM freed"
    call = _last(flux3)
    assert call["args"] == ["gen", "--spec", "-"], "references go through gen + refs"
    spec = call["spec"]
    assert spec["op"] == "gen" and spec["prompt"] == "make it blue" and spec["seed"] == 5
    assert len(spec["refs"]) == 2 and (spec["width"], spec["height"]) == (32, 16)
    assert all(os.path.isfile(p) for p in spec["refs"])
    assert imgs[0].getpixel((31, 8)) == (0, 0, 200), "the result image comes back"
    assert any(m.startswith("HANDOFF:") for m in logs)
    assert any("guidance ignored" in m for m in logs), "flux3 warnings are relayed"
    assert os.path.isfile(os.path.join(spec["out_dir"], "handoff.json"))


def test_a_closed_flux3_runs_cold_after_freeing_our_vram(flux3, tmp_path, logs):
    (flux3 / "mode.txt").write_text("cold")
    freed = []
    _, _, route = H.edit(_img(), "x", output_dir=str(tmp_path),
                         free_vram=lambda: freed.append(1))
    assert route == "cold" and freed == [1]
    call = _last(flux3)
    assert call["args"] == ["edit", "--spec", "-", "--local"]
    assert call["spec"]["op"] == "edit" and os.path.isfile(call["spec"]["input"])
    assert "refs" not in call["spec"], "edit keeps the input as its only reference"
    assert any("not running" in m and "frees its VRAM" in m for m in logs)


def test_an_instance_that_is_this_app_is_not_taken_for_flux3(flux3, tmp_path, logs):
    (flux3 / "mode.txt").write_text("clash")
    _, _, route = H.edit(_img(), "x", output_dir=str(tmp_path), free_vram=lambda: None)
    assert route == "cold" and any("same port" in m for m in logs)


def test_another_tool_in_the_folder_is_refused(flux3, tmp_path):
    (flux3 / "mode.txt").write_text("wrong")
    with pytest.raises(H.HandoffError, match="crispz-klein, not crispz-flux3"):
        H.edit(_img(), "x", output_dir=str(tmp_path))


def test_a_flux3_failure_is_reported(flux3, tmp_path):
    (flux3 / "mode.txt").write_text("fail")
    with pytest.raises(H.HandoffError, match="out of memory"):
        H.edit(_img(), "x", output_dir=str(tmp_path))


def test_no_instruction_or_no_image_sends_nothing(flux3, tmp_path):
    with pytest.raises(H.HandoffError, match="instruction"):
        H.edit(_img(), "  ", output_dir=str(tmp_path))
    with pytest.raises(H.HandoffError, match="image"):
        H.edit(None, "x", output_dir=str(tmp_path))
    assert not (flux3 / "last_call.json").exists()


def test_extra_references_are_cut_and_said(flux3, tmp_path, logs):
    H.edit(_img(), "x", refs=[_img()] * 5, output_dir=str(tmp_path), free_vram=lambda: None)
    assert len(_last(flux3)["spec"]["refs"]) == 1 + H.MAX_REFS
    assert any("3 kept" in m for m in logs)


def test_the_tab_handler_uses_the_main_prompt_when_the_field_is_empty(flux3, tmp_path,
                                                                      monkeypatch):
    import cz_ui
    monkeypatch.setattr(cz_ui.cz_pipeline, "free_vram", lambda: None)
    imgs, msg, hist, _ = cz_ui._ui_handoff(_img(), None, None, None, "", "turn it blue", -1,
                                           [], str(tmp_path))
    assert len(imgs) == 1 and "Edited by crispz-flux3 (its running app)" in msg
    assert _last(flux3)["spec"]["prompt"] == "turn it blue" and hist[0] is imgs[0]
    imgs, msg, _, _ = cz_ui._ui_handoff(None, None, None, None, "x", "", -1, [],
                                        str(tmp_path))
    assert imgs == [] and "drop the image" in msg
