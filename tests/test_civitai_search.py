"""CivitAI LoRA search + download, ported from crispz-studio.

The part that is NOT a copy is the ranking. The single-base forks compare the CivitAI
base label for EQUALITY ('Z-Image'); that would match nothing here, because CivitAI labels
these LoRAs 'Flux.2 Klein 4B-base' -- read from a real sidecar on this machine. So klein
matches on a normalised PREFIX, and the VARIANT matters: a 4B LoRA does not load on a 9B
base (cz_pipeline refuses it with its reason), so the versions of the LOADED variant come
first and 'Klein only' filters on the family.

No network: _api_get and the download stream are stubbed. What is checked:
  - the ranking puts the loaded variant first, the other Klein ones next, the rest last;
  - 'Klein only' keeps both Klein variants and drops the foreign bases;
  - an unreadable variant degrades to the family ranking instead of breaking the search;
  - search_loras flattens one entry per model VERSION and survives a network failure;
  - a download whose SHA256 does not match is refused AND the file is removed;
  - an existing file is never overwritten.

Run:  .venv/Scripts/python tests/test_civitai_search.py

"""
import hashlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_civitai  # noqa: E402
import cz_ui  # noqa: E402


def _model(mid, name, versions):
    return {"id": mid, "name": name, "creator": {"username": "someone"},
            "nsfw": False, "modelVersions": versions}


def _ver(vid, base, fname="a.safetensors", sha="", size=2048):
    return {"id": vid, "name": f"v{vid}", "baseModel": base,
            "files": [{"name": fname, "primary": True, "sizeKB": size,
                       "downloadUrl": f"https://civitai.com/api/download/models/{vid}",
                       "hashes": {"SHA256": sha}}],
            "images": [{"url": "https://img/x.jpg"}]}


class _Api:
    """Stubs cz_civitai._api_get with a fixed payload (or None = network failure)."""

    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def __enter__(self):
        self.real = cz_civitai._api_get

        def fake(endpoint, params=None, api_key=None):
            self.calls.append((endpoint, params))
            return self.payload

        cz_civitai._api_get = fake
        return self

    def __exit__(self, *exc):
        cz_civitai._api_get = self.real
        return False


class _Variant:
    """Forces what _klein_base_prefixes() reads as the loaded variant."""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        import cz_pipeline
        self.p = cz_pipeline
        self.dim, self.vn = cz_pipeline._base_hidden_dim, cz_pipeline._variant_name
        if self.name is None:
            cz_pipeline._base_hidden_dim = lambda *a, **k: (_ for _ in ()).throw(
                RuntimeError("gated repo"))
        else:
            cz_pipeline._base_hidden_dim = lambda *a, **k: 1
            cz_pipeline._variant_name = lambda _d: self.name
        return self

    def __exit__(self, *exc):
        self.p._base_hidden_dim, self.p._variant_name = self.dim, self.vn
        return False


# CivitAI's label for FLUX.3 is not known yet (the model is days old): cz_ui matches a
# 'flux3' PREFIX on the normalised label, which both spellings below satisfy.
_PAYLOAD = {"items": [
    _model(1, "Foreign", [_ver(11, "SDXL 1.0")]),
    _model(2, "Klein4B", [_ver(22, "Flux.2 Klein 4B-base")]),
    _model(3, "Flux3", [_ver(33, "FLUX.3 Image Klein")]),
    _model(4, "Flux3b", [_ver(44, "Flux 3 Klein")]),
]}


def _bases(dd_update):
    """The base model of each candidate, read back from the dropdown labels."""
    return [lbl.split("[", 1)[1].split("]", 1)[0] for lbl in dd_update["choices"]]


def test_search_flattens_one_entry_per_version():
    with _Api({"items": [_model(1, "Two", [_ver(11, "SDXL 1.0"), _ver(12, "SDXL 1.0")])]}):
        out = cz_civitai.search_loras("x")
    assert len(out) == 2, out
    assert [c["versionId"] for c in out] == [11, 12]
    assert out[0]["modelName"] == "Two" and out[0]["url"].endswith("/models/1")


def test_a_network_failure_is_an_empty_list_not_an_exception():
    with _Api(None):
        assert cz_civitai.search_loras("x") == []
    assert cz_civitai.search_loras("   ") == []          # empty query: no call at all


def test_the_flux3_versions_come_first():
    """The ranking: the FLUX.3 LoRAs first (they load here), then the rest in CivitAI's
    own order. A FLUX.2 Klein one ranks with the foreign bases: same name, other family,
    it does not load in this build."""
    with _Api(_PAYLOAD):
        _q, dd, state, status = cz_ui._ui_civitai_lora_search("x", False)
    assert _bases(dd) == ["FLUX.3 Image Klein", "Flux 3 Klein", "SDXL 1.0",
                          "Flux.2 Klein 4B-base"], _bases(dd)
    assert len(state) == 4


def test_flux3_only_drops_the_other_families_flux2_included():
    with _Api(_PAYLOAD):
        _q, dd, _s, _st = cz_ui._ui_civitai_lora_search("x", True)
    assert _bases(dd) == ["FLUX.3 Image Klein", "Flux 3 Klein"], _bases(dd)


def test_the_family_prefix_needs_no_hub_access():
    """klein read the loaded variant off the Hub (and a gated repo answered 401). FLUX.3
    has one width, so the prefix is a constant: no network, nothing to degrade."""
    fam, cur = cz_ui._klein_base_prefixes()
    assert fam == cur == "flux3", (fam, cur)


def test_no_result_names_the_filter_as_the_likely_cause():
    with _Api({"items": [_model(1, "Foreign", [_ver(11, "SDXL 1.0")])]}):
        _q, dd, state, status = cz_ui._ui_civitai_lora_search("x", True)
    assert dd["choices"] == [] and state == {}
    assert "FLUX.3 only" in status, status


class _Stream:
    """Stubs urlopen with a fixed body, so the download never touches the network."""

    def __init__(self, body):
        self.body = body

    def __enter__(self):
        import urllib.request
        self.real = urllib.request.urlopen
        body = self.body

        class _R:
            headers = {"Content-Length": str(len(body))}

            def __init__(self):
                self._b = io.BytesIO(body)

            def read(self, n):
                return self._b.read(n)

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        urllib.request.urlopen = lambda *a, **k: _R()
        return self

    def __exit__(self, *exc):
        import urllib.request
        urllib.request.urlopen = self.real
        return False


def _no_enrich():
    """fetch_civitai_for_model would hit the network after a download."""
    real = cz_civitai.fetch_civitai_for_model
    cz_civitai.fetch_civitai_for_model = lambda *a, **k: None
    return real


def test_a_sha256_mismatch_refuses_and_removes_the_file():
    """The whole point of checking during the stream: a corrupted LoRA must not be left on
    disk looking valid."""
    body = b"not-the-announced-bytes"
    real = _no_enrich()
    try:
        with _Stream(body), tempfile.TemporaryDirectory() as d:
            cand = {"downloadUrl": "https://x/y", "fileName": "bad.safetensors",
                    "sha256": "0" * 64, "sizeKB": 1}
            res = cz_civitai.download_model_file(cand, d)
            assert res["success"] is False, res
            assert "SHA256 mismatch" in res["message"], res
            assert os.listdir(d) == [], os.listdir(d)      # no .part left either
    finally:
        cz_civitai.fetch_civitai_for_model = real


def test_a_matching_sha256_lands_the_file():
    body = b"the-real-bytes"
    real = _no_enrich()
    try:
        with _Stream(body), tempfile.TemporaryDirectory() as d:
            cand = {"downloadUrl": "https://x/y", "fileName": "ok.safetensors",
                    "sha256": hashlib.sha256(body).hexdigest(), "sizeKB": 1}
            res = cz_civitai.download_model_file(cand, d)
            assert res["success"] is True, res
            assert "verified" in res["message"], res
            # The file AND its hash sidecar: _cache_sha256 writes '<stem>.civitai.json' so
            # the next scan does not re-read the whole file to recompute the hash.
            assert sorted(os.listdir(d)) == ["ok.civitai.json", "ok.safetensors"], \
                os.listdir(d)
            assert res["path"] == os.path.join(d, "ok.safetensors"), res
    finally:
        cz_civitai.fetch_civitai_for_model = real


def test_an_existing_file_is_never_overwritten():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "there.safetensors")
        with open(p, "wb") as f:
            f.write(b"mine")
        res = cz_civitai.download_model_file(
            {"downloadUrl": "https://x/y", "fileName": "there.safetensors"}, d)
        assert res["success"] is True and "already exists" in res["message"], res
        with open(p, "rb") as f:
            assert f.read() == b"mine", "the existing file was overwritten"


def test_a_video_example_gives_a_poster_frame_as_preview():
    """The video LoRAs (Wan, LTX...) illustrate themselves with .mp4 'images': the search
    card shows the CDN's first frame (a JPEG), never a URL an <img> cannot render."""
    vid = ("https://image.civitai.com/sig/uuid/original=true/uuid.mp4")
    v = _ver(7, "Wan Video 14B t2v")
    v["images"] = [{"url": vid, "type": "video"}, {"url": "https://img/x.jpg"}]
    with _Api({"items": [_model(1, "Vid", [v])]}):
        out = cz_civitai.search_loras("vid")
    assert out[0]["previewUrl"] == "https://image.civitai.com/sig/uuid/anim=false,transcode=true/uuid.jpeg"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"OK {fn.__name__}")
    print(f"All {len(tests)} CivitAI search/download tests passed.")
