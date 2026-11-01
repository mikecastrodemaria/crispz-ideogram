"""cz_magic: plain idea -> JSON caption, three engines (lot 2).

No network, no GPU: Ollama and the Ideogram API are mocked at their transport, the local
engine runs the real IdeogramPipe.magic_prompt on the tiny Ideogram 4 (random weights, so
the text is garbage: what is checked is the plumbing, the encoder offload and the head).

Run:  .venv/Scripts/python -m pytest tests/test_magic_prompt.py -q
"""
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest  # noqa: E402
import torch  # noqa: E402

import cz_caption  # noqa: E402
import cz_magic as M  # noqa: E402

GOOD = {"high_level_description": "A jazz poster.",
        "compositional_deconstruction": {"background": "Teal.", "elements": [
            {"type": "text", "text": "JAZZ", "desc": "Big cream headline."}]}}


@pytest.fixture
def settings(monkeypatch):
    vals = {}
    monkeypatch.setattr(M, "_setting", lambda k, d=None: vals.get(k, d))
    monkeypatch.delenv("IDEOGRAM_API_KEY", raising=False)
    return vals


def test_clean_reads_what_llms_actually_return():
    v1 = dict(GOOD, aspect_ratio="1:1")
    raw = "<think>planning...</think>\n```json\n" + json.dumps(v1) + "\n```"
    text, warnings = M.clean(raw)
    assert warnings == [] and "aspect_ratio" not in text and json.loads(text) == GOOD
    scrambled = {"compositional_deconstruction": {"elements": [
        {"desc": "d", "text": "T", "type": "text"}], "background": "b"},
        "high_level_description": "h"}
    text, warnings = M.clean("Sure! " + json.dumps(scrambled) + " Enjoy.")
    assert warnings == [] and list(json.loads(text)) == ["high_level_description",
                                                         "compositional_deconstruction"]
    with pytest.raises(M.MagicPromptError, match="no JSON object"):
        M.clean("I cannot do that")
    with pytest.raises(M.MagicPromptError, match="invalid JSON"):
        M.clean('{"high_level_description": "unterminated}')
    text, _ = M.clean(json.dumps({**GOOD, "compositional_deconstruction": {
        "background": "b", "elements": [{"type": "obj", "bbox": [0, 0, 9, 9], "desc": "x"}]}}),
        strip_bboxes=True)
    assert "bbox" not in text


def test_engine_setting(settings):
    assert M.engine() == "local"
    settings["magic_prompt_engine"] = "OLLAMA"
    assert M.engine() == "ollama"
    settings["magic_prompt_engine"] = "nonsense"
    assert M.engine() == "local"


def test_cloud_is_off_by_default_and_needs_both_switches(settings, monkeypatch):
    called = []
    monkeypatch.setattr(M.urllib.request, "urlopen", lambda *a, **k: called.append(1))
    with pytest.raises(M.MagicPromptError, match="off"):
        M.expand("a poster", 1024, 1024, which="ideogram_api")
    settings["ideogram_cloud_enabled"] = True
    with pytest.raises(M.MagicPromptError, match="No Ideogram API key"):
        M.expand("a poster", 1024, 1024, which="ideogram_api")
    assert called == []                                   # nothing ever left


def test_cloud_call_is_logged_and_cleaned(settings, monkeypatch):
    settings.update(ideogram_cloud_enabled=True, ideogram_api_key="k-123")
    sent, logs = {}, []
    monkeypatch.setattr(M, "_log", lambda m: logs.append(m))

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(req, timeout=0):
        sent["url"], sent["headers"] = req.full_url, dict(req.headers)
        sent["body"] = json.loads(req.data)
        scrambled = {"compositional_deconstruction": {"elements": [
            {"desc": "Big headline.", "text": "OPEN", "type": "text"}],
            "background": "Red."}, "high_level_description": "A sign."}
        return Resp(json.dumps({"aspect_ratio": "16x9", "json_prompt": scrambled}).encode())
    monkeypatch.setattr(M.urllib.request, "urlopen", urlopen)
    text, warnings, used = M.expand("a sign that says OPEN", 2048, 1152, which="ideogram_api")
    assert used == "ideogram_api" and warnings == []
    assert sent["url"] == M.IDEOGRAM_MAGIC_PROMPT_URL
    assert sent["headers"].get("Api-key") == "k-123"
    assert sent["body"] == {"text_prompt": "a sign that says OPEN", "aspect_ratio": "16x9"}
    assert any(m.startswith("CLOUD:") and "United States" in m for m in logs)
    assert cz_caption.validate(text) == []


def test_ollama_gets_the_system_prompt_and_the_schema(settings, monkeypatch):
    import prompt_improve
    calls = []

    def http(path, payload=None, base=None, timeout=8):
        calls.append((path, payload))
        if path == "/api/tags":
            return {"models": [{"name": "qwen3:8b"}]}
        return {"response": json.dumps(GOOD)}
    monkeypatch.setattr(prompt_improve, "http", http)
    text, warnings, used = M.expand("a jazz poster", 1920, 320, seed=5, which="ollama")
    assert used == "ollama" and warnings == [] and json.loads(text) == GOOD
    path, payload = calls[-1]
    assert path == "/api/generate" and payload["model"] == "qwen3:8b"
    assert "structured JSON caption" in payload["system"]
    assert "6:1" in payload["prompt"] and "a jazz poster" in payload["prompt"]
    assert payload["format"] == M.OLLAMA_FORMAT and payload["options"]["seed"] == 5
    # Ideogram's v1 system prompt instead, no format constraint (it adds keys)
    settings["magic_prompt_ollama_system"] = "ideogram_v1"
    settings["magic_prompt_ollama_model"] = "llama3"
    M.expand("a jazz poster", 1024, 1024, which="ollama")
    payload = calls[-1][1]
    assert payload["model"] == "llama3" and "format" not in payload
    assert "OUTPUT CONTRACT" in payload["system"] and "1:1" in payload["prompt"]


def test_expand_refuses_empty_and_json(settings):
    with pytest.raises(M.MagicPromptError, match="empty"):
        M.expand("  ", 1024, 1024)
    with pytest.raises(M.MagicPromptError, match="already a JSON caption"):
        M.expand(json.dumps(GOOD), 1024, 1024)


def test_local_engine_goes_through_the_pipeline(settings, monkeypatch):
    import cz_pipeline
    seen = []
    monkeypatch.setattr(cz_pipeline, "magic_prompt_local",
                        lambda p, w, h, seed=None: seen.append((p, w, h, seed)) or json.dumps(GOOD))
    text, warnings, used = M.expand("a jazz poster", 1024, 1536, seed=9)
    assert used == "local" and warnings == [] and seen == [("a jazz poster", 1024, 1536, 9)]


def test_local_magic_prompt_plumbing_on_the_tiny_pipe():
    """The real IdeogramPipe.magic_prompt: a head registered (no download here), the
    encoder brought in for the call and sent back after under the encoder-only offload,
    the head too; seeded, so the same call gives the same text."""
    from diffusers import Ideogram4PromptEnhancerHead
    from ideogram_tiny import tiny_pipe
    pipe = tiny_pipe()
    cfg = pipe._pipe.text_encoder.config.text_config
    torch.manual_seed(0)
    head = Ideogram4PromptEnhancerHead(hidden_size=cfg.hidden_size, vocab_size=cfg.vocab_size)
    pipe._pipe.register_modules(prompt_enhancer_head=head)
    assert pipe.ensure_prompt_enhancer() is head                     # no reload
    pipe.enable_model_cpu_offload(device="cpu")
    moves = []
    te_to = pipe._pipe.text_encoder.to
    pipe._pipe.text_encoder.to = lambda *a, **k: moves.append(str(a[0])) or te_to(*a, **k)
    try:
        a = pipe.magic_prompt("a poster", 256, 256, seed=3, max_new_tokens=8)
        b = pipe.magic_prompt("a poster", 256, 256, seed=3, max_new_tokens=8)
    finally:
        del pipe._pipe.text_encoder.to
    assert isinstance(a, str) and a == b
    assert moves == ["cpu", "cpu", "cpu", "cpu"]        # in + out, twice (dev = cpu here)
    assert not pipe._te_on_gpu


def test_describe_converts_qwen_boxes_to_ideogram_order(monkeypatch):
    """Qwen3-VL emits bbox_2d = [x1, y1, x2, y2]; the caption wants [y0, x0, y1, x1]."""
    import cz_pipeline
    from PIL import Image
    raw = json.dumps({"high_level_description": "A sign.",
                      "compositional_deconstruction": {"background": "A wall.", "elements": [
                          {"type": "text", "bbox_2d": [100, 200, 900, 400], "text": "OPEN",
                           "desc": "Red letters."},
                          {"type": "obj", "bbox_2d": [700, 800, 300, 950], "desc": "A lamp."}]}})
    monkeypatch.setattr(cz_pipeline, "describe_local", lambda img: raw)
    text, warnings = M.describe(Image.new("RGB", (64, 64)))
    els = json.loads(text)["compositional_deconstruction"]["elements"]
    assert els[0]["bbox"] == [200, 100, 400, 900] and "bbox_2d" not in els[0]
    assert els[1]["bbox"] == [800, 300, 950, 700]                 # corners given reversed
    assert list(els[0]) == ["type", "bbox", "text", "desc"] and warnings == []
    with pytest.raises(M.MagicPromptError, match="no image"):
        M.describe(None)


def test_describe_plumbing_on_the_tiny_pipe():
    """The real IdeogramPipe.describe_image on the tiny Qwen3-VL (vision tower of depth 1,
    random weights): the image goes through the real Qwen3-VL processor and generate, the
    encoder comes and goes under the encoder-only offload. Greedy: deterministic."""
    from diffusers import Ideogram4PromptEnhancerHead
    from PIL import Image
    from ideogram_tiny import tiny_pipe
    pipe = tiny_pipe()
    cfg = pipe._pipe.text_encoder.config.text_config
    torch.manual_seed(0)
    pipe._pipe.register_modules(prompt_enhancer_head=Ideogram4PromptEnhancerHead(
        hidden_size=cfg.hidden_size, vocab_size=cfg.vocab_size))
    pipe.enable_model_cpu_offload(device="cpu")
    img = Image.new("RGB", (1600, 900), (200, 30, 30))
    a = pipe.describe_image(img, max_new_tokens=6)
    b = pipe.describe_image(img, max_new_tokens=6)
    assert isinstance(a, str) and a == b
    assert not pipe._te_on_gpu
