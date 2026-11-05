"""Latent -> RGB projection of the live preview: the packed-latent geometry, the FLUX.2 helpers
of diffusers, and the Ideogram projection (token order and bn stats folded into one
matrix, checked against the exact inverse of the decode). No GPU, no model."""
# scope: ideogram
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import types  # noqa: E402

import torch  # noqa: E402

import cz_pipeline as P  # noqa: E402


def _lat(w, h, ch=128):
    return torch.zeros(1, (h // 16) * (w // 16), ch)


def test_the_grid_comes_from_the_asked_resolution():
    img = P.latent_preview_image(_lat(1024, 768), 1024, 768)
    assert img is not None
    assert img.width > img.height, img.size              # landscape stays landscape
    assert abs(img.width / img.height - 1024 / 768) < 0.01, img.size


def test_a_portrait_render_stays_portrait():
    img = P.latent_preview_image(_lat(768, 1152), 768, 1152)
    assert abs(img.width / img.height - 768 / 1152) < 0.01, img.size


def test_the_preview_is_scaled_up_to_the_configured_side():
    img = P.latent_preview_image(_lat(768, 768), 768, 768)
    assert max(img.size) == P.LIVE_PREVIEW_MAX_SIDE, img.size


def test_an_unknown_resolution_falls_back_to_a_square():
    """The grid cannot be guessed from the token count alone; a square is the only safe
    reading, and it is better than no preview at all."""
    img = P.latent_preview_image(_lat(768, 768), 0, 0)
    assert img is not None and img.width == img.height, img


def test_a_token_count_that_fits_nothing_gives_no_preview():
    """Better nothing than a scrambled image: 3000 tokens is neither 1024x768's grid nor
    a square."""
    assert P.latent_preview_image(torch.zeros(1, 3000, 128), 1024, 768) is None


def test_unpacked_latents_are_refused():
    """A 4D tensor is another pipeline's layout, not FLUX.2's: no guessing."""
    assert P.latent_preview_image(torch.zeros(1, 32, 96, 96), 768, 768) is None


def test_the_matrix_matches_the_packed_channel_count():
    assert len(P._LATENT_RGB) == 128, len(P._LATENT_RGB)
    assert all(len(row) == 3 for row in P._LATENT_RGB)
    assert len(P._LATENT_RGB_BIAS) == 3


def test_a_preview_failure_never_reaches_the_render():
    """_store_preview swallows everything: it is a courtesy, not a result."""
    P.preview_begin()
    try:
        P._store_preview(object(), 1, 8, 768, 768)       # not a tensor at all
        assert P.preview_snapshot()["img"] is None
        P._store_preview(_lat(768, 768), 2, 8, 768, 768)
        snap = P.preview_snapshot()
        assert snap["img"] is not None and snap["step"] == 2 and snap["seq"] == 1, snap
    finally:
        P.preview_end()


# --- Ideogram 4: same VAE, (p, p, c) tokens normalized with the FLUX-ordered bn stats ----
def _flux_and_ideogram_tokens(r, mean, std):
    """The same raw VAE latent in both layouts. FLUX.2 through diffusers' own helpers;
    Ideogram as the exact inverse of pipeline_ideogram4's decode (bn, view, permute)."""
    from diffusers import Flux2Pipeline
    B, C, H, W = r.shape
    pf = Flux2Pipeline._patchify_latents(r)
    pf = (pf - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)
    flux = Flux2Pipeline._pack_latents(pf)
    ide = r.view(B, C, H // 2, 2, W // 2, 2).permute(0, 2, 4, 3, 5, 1)
    ide = ide.reshape(B, (H // 2) * (W // 2), 4 * C)
    ide = (ide - mean.view(1, 1, -1)) / std.view(1, 1, -1)
    # The pipeline's own decode must give r back, or the inverse above is wrong.
    z = ide * std.view(1, 1, -1) + mean.view(1, 1, -1)
    z = z.view(B, H // 2, W // 2, 2, 2, C).permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)
    assert torch.allclose(z, r, atol=1e-5)
    return flux, ide


class _FakeVae:
    def __init__(self, mean, var, eps=1e-4):
        self.bn = types.SimpleNamespace(running_mean=mean, running_var=var)
        self.config = types.SimpleNamespace(batch_norm_eps=eps)


def test_ideogram_projection_equals_flux_on_the_same_latent():
    """The folded matrix applied to Ideogram tokens == the FLUX matrix applied to the FLUX
    tokens of the same latent. Random bn stats, so neither the permutation nor the affine
    correction can pass by luck (the real VAE was measured: MAE 52-61 -> 13-24 / 255)."""
    g = torch.Generator().manual_seed(0)
    mean = torch.randn(128, generator=g)
    var = torch.rand(128, generator=g) * 3 + 0.2
    vae = _FakeVae(mean, var)
    std = torch.sqrt(var + 1e-4)
    r = torch.randn(1, 32, 24, 32, generator=g) * 2
    flux, ide = _flux_and_ideogram_tokens(r, mean, std)
    m, b = P._latent_rgb_on(torch.device("cpu"), torch.float32)
    ref = flux[0] @ m + b
    pm, pb = P._ideogram_preview_proj(vae)
    got = ide[0] @ pm + pb
    assert torch.allclose(got, ref, atol=1e-4), (got - ref).abs().max()
    # ... and the FLUX matrix as is on Ideogram tokens is NOT that (the bug fixed).
    assert not torch.allclose(ide[0] @ m + b, ref, atol=1e-2)


def test_ideogram_pipe_gets_its_projection_others_do_not():
    vae = _FakeVae(torch.zeros(128), torch.ones(128))
    Ideogram4Pipeline = type("Ideogram4Pipeline", (), {})
    Flux2KleinPipeline = type("Flux2KleinPipeline", (), {})
    ide = Ideogram4Pipeline()
    ide.vae = vae
    fl = Flux2KleinPipeline()
    fl.vae = vae
    assert P._preview_proj_for(ide) is not None
    assert P._preview_proj_for(types.SimpleNamespace(_pipe=ide)) is not None   # wrapper
    assert P._preview_proj_for(fl) is None
    assert P._preview_proj_for(None) is None
    img = P.latent_preview_image(_lat(512, 256), 512, 256, proj=P._preview_proj_for(ide))
    assert img is not None and img.width == 2 * img.height
