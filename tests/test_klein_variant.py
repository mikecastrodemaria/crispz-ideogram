"""crispz-klein told its 4B and 9B apart by width. crispz-flux3 cannot: FLUX.3 Image
Klein is 3072 wide like the 4B, so the FAMILY is read from the tensor names instead, and
the width helpers below only serve to NAME a FLUX.2 file in a message.

crispz-klein's original note: the 4B and the 9B share the architecture AND the tensor names.

So the architecture guard lets both through, and a 9B loaded in a 4B pipeline
blew up after reading gigabytes, on an unreadable diffusers
message: "expected shape [18432, 3072], but got [24576, 4096]".
Only the HIDDEN DIMENSION separates them -- readable from the header, without loading a
single weight.

"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from safetensors.torch import save_file

import cz_pipeline as P

TMP = os.path.join(os.environ.get("TEMP") or "/tmp", "cz_variant")
os.makedirs(TMP, exist_ok=True)

# (the hidden dim, the name) — 3072 = klein-4B, 4096 = klein-9B
DIMS = {"4B": 3072, "9B": 4096}


def _ckpt(name, dim, key="double_stream_modulation_img.lin.weight"):
    """A fake transformer: only the header counts, the weights are empty."""
    p = os.path.join(TMP, name)
    save_file({key: torch.zeros(dim * 6, dim),
               "single_transformer_blocks.0.attn.to_q.weight": torch.zeros(2, 2),
               "x_embedder.weight": torch.zeros(2, 2)}, p)
    return p


def test_hidden_dim_read_from_header():
    for label, dim in DIMS.items():
        p = _ckpt(f"{label}.safetensors", dim)
        assert P._flux2_hidden_dim(p) == dim, (label, P._flux2_hidden_dim(p))
    # the diffusers layout writes '.linear.' instead of '.lin.'
    p = _ckpt("diffusers_layout.safetensors", 3072,
              key="double_stream_modulation_img.linear.weight")
    assert P._flux2_hidden_dim(p) == 3072
    # the ComfyUI prefix
    p = _ckpt("comfy.safetensors", 4096,
              key="model.diffusion_model.double_stream_modulation_img.lin.weight")
    assert P._flux2_hidden_dim(p) == 4096
    print("OK test_hidden_dim_read_from_header")


def _with_base(repo, dim, fn):
    old_repo, old_cache = P.BASE_REPO, dict(P._BASE_DIM_CACHE)
    P.BASE_REPO = repo
    P._BASE_DIM_CACHE[repo] = dim
    try:
        return fn()
    finally:
        P.BASE_REPO = old_repo
        P._BASE_DIM_CACHE.clear()
        P._BASE_DIM_CACHE.update(old_cache)


def _ckpt3(name):
    """A fake FLUX.3 transformer: the block NAMES are what the family guard reads."""
    p = os.path.join(TMP, name)
    save_file({"early_stream_modulations.0.lin.weight": torch.zeros(2, 2),
               "mode_blocks.0.img_attn.qkv.weight": torch.zeros(2, 2)}, p)
    return p


def test_a_flux2_file_is_refused_and_a_flux3_one_passes_whatever_the_base():
    """crispz-flux3's reading of the same trap: FLUX.3 is as wide as FLUX.2-klein-4B, so
    the width decides nothing any more. The FAMILY does, from the tensor names, and it
    does not depend on which base repo is loaded."""
    p2 = _ckpt("m4.safetensors", 3072)          # FLUX.2 names, FLUX.3's width
    p3 = _ckpt3("m3.safetensors")
    for base, dim in (("base-flux3", 3072), ("mystere", None)):
        assert _with_base(base, dim, lambda: P._safetensors_unsupported(p3)) is None
        msg = _with_base(base, dim, lambda: P._safetensors_unsupported(p2))
        assert msg and "FLUX.2" in msg and "crispz-klein" in msg, msg
        # the per-file reason stays SHORT: it is repeated as many times as there are
        # files discarded.
        assert len(msg) < 130, f"reason too long ({len(msg)} chars): {msg}"
        assert "NON-COMMERCIAL" not in msg
    print("OK test_a_flux2_file_is_refused_and_a_flux3_one_passes_whatever_the_base")


def test_summary_names_the_studio_that_runs_the_file_once():
    """The summary of the discarded files says where they DO run, once, and never
    pretends a setting of this build could load them."""
    line = _with_base("base-flux3", 3072, lambda: P._variant_skip_summary(11, 4096))
    assert "11 checkpoint" in line, line
    assert "FLUX.2-klein-9B" in line and "FLUX.3 Image Klein" in line, line
    assert "crispz-klein" in line, line
    assert "zimage_model" not in line, "the old name must no longer appear"
    assert "NON-COMMERCIAL" not in line and "switch the base" not in line, line
    print("OK test_summary_names_the_studio_that_runs_the_file_once")


def test_unknown_base_never_discards():
    """A house rule: we NEVER discard a model on a doubt. A file with no family marker
    at all (another layout) passes the family guard, whatever the base; a POSITIVE
    FLUX.2 reading is not a doubt."""
    p = os.path.join(TMP, "nofamily.safetensors")
    save_file({"x_embedder.weight": torch.zeros(2, 2)}, p)
    assert _with_base("mystere", None, lambda: P._safetensors_unsupported(p)) is None
    assert _with_base("mystere", None,
                      lambda: P._safetensors_unsupported(_ckpt("u9.safetensors", 4096)))
    print("OK test_unknown_base_never_discards")


def test_bogus_shape_is_not_taken_for_a_hidden_dim():
    """The name alone is not enough: the structural out/in ratio must be 6 (a double
    stream) or 3 (a single stream), otherwise it is not a hidden dimension. Without that,
    a 2x2 fixture carrying the right name made a valid file be refused."""
    p = os.path.join(TMP, "bogus.safetensors")
    save_file({"double_stream_modulation_img.lin.weight": torch.zeros(2, 2)}, p)
    assert P._flux2_hidden_dim(p) is None
    # the right ratio for the single stream (3) -> recognised
    p = os.path.join(TMP, "single.safetensors")
    save_file({"single_stream_modulation.lin.weight": torch.zeros(9216, 3072)}, p)
    assert P._flux2_hidden_dim(p) == 3072
    print("OK test_bogus_shape_is_not_taken_for_a_hidden_dim")


def test_file_without_signature_is_not_filtered():
    """A checkpoint without the modulation keys (another layout) must not be
    discarded for a variant reason: the dimension is simply unknown."""
    p = os.path.join(TMP, "nosig.safetensors")
    save_file({"transformer_blocks.0.attn.to_q.weight": torch.zeros(2, 2),
               "x_embedder.weight": torch.zeros(2, 2)}, p)
    assert P._flux2_hidden_dim(p) is None
    assert _with_base("base-4b", 3072, lambda: P._flux2_variant_mismatch(None)) is None
    print("OK test_file_without_signature_is_not_filtered")


# ---------------------------------------------------------------------------
# On the LoRA side. A LoRA holds no weight of the model -- but its two matrices keep
# a trace of it: lora_A has the shape [rank, in], lora_B [out, rank]. On a
# projection whose input IS the hidden dimension, the shape therefore gives it.
# Without that guard, a 4B LoRA set on a 9B base made peft pour out forty
# lines of "size mismatch ... torch.Size([27648, 128]) ... torch.Size([36864, 128])",
# where nothing says that 27648 = 9 x 3072, so a 4B. Caught for real on an EDIT LoRA,
# which made the whole edit fail without ever naming the cause.
# ---------------------------------------------------------------------------

# The three dialects CAUGHT in a real library of 70 klein LoRAs. The
# first version of the guard knew only the first one -- and so recognised
# only ONE file out of 70. The others are in the original FLUX layout, and one inserts
# peft's adapter name ('default') between the matrix and '.weight'.
_LORA_LAYOUTS = {
    "diffusers": ("transformer.transformer_blocks.0.attn.to_q.lora_{ab}.weight",
                  "transformer.single_transformer_blocks.0.attn.to_out.lora_{ab}.weight"),
    "flux-original": ("diffusion_model.double_blocks.0.img_attn.proj.lora_{ab}.weight",
                      "diffusion_model.single_blocks.0.linear2.lora_{ab}.weight"),
    "peft-adapter-name": ("single_transformer_blocks.0.attn.to_out.lora_{ab}.default.weight",
                          "transformer_blocks.0.attn.to_q.lora_{ab}.default.weight"),
}


def _lora(name, dim, rank=128, layout="diffusers"):
    """A fake FLUX.2 LoRA: only the shapes count. lora_A = [rank, in],
    lora_B = [out, rank]."""
    sd = {}
    for tmpl in _LORA_LAYOUTS[layout]:
        sd[tmpl.format(ab="A")] = torch.zeros(rank, dim)
        sd[tmpl.format(ab="B")] = torch.zeros(dim, rank)
    p = os.path.join(TMP, name)
    save_file(sd, p)
    return p


def test_a_lora_declares_its_variant_through_its_shapes():
    for label, dim in DIMS.items():
        p = _lora(f"lora_{label}.safetensors", dim)
        assert P._flux2_lora_hidden_dim(p) == dim, (label, P._flux2_lora_hidden_dim(p))
    print("OK test_a_lora_declares_its_variant_through_its_shapes")


def test_a_flux2_lora_is_refused_in_one_sentence():
    """The same intention as klein's 4B-on-9B refusal: peft would otherwise dump forty
    lines of key mismatches. Here the other family IS the wrong model, and a FLUX.3 LoRA
    (block names mode_blocks / early_stream_modulations) goes through."""
    why = P._lora_unsupported(_lora("lora_flux2.safetensors", 3072, layout="flux-original"))
    assert why, "a FLUX.2 Klein LoRA must be refused on FLUX.3"
    assert "FLUX.2" in why and "FLUX.3" in why and "crispz-klein" in why, why
    assert "\n" not in why.strip(), "one sentence"
    p = os.path.join(TMP, "lora_flux3.safetensors")
    save_file({"transformer.mode_blocks.0.attn.to_q.lora_A.weight": torch.zeros(8, 3072),
               "transformer.mode_blocks.0.attn.to_q.lora_B.weight": torch.zeros(3072, 8)}, p)
    assert P._lora_unsupported(p) is None
    print("OK test_a_flux2_lora_is_refused_in_one_sentence")


def test_a_lora_without_a_signature_is_never_filtered():
    """We do not know every LoRA in the world: with no recognised signature, we do NOT
    filter. Discarding a valid LoRA would be worse than the message we replace."""
    p = os.path.join(TMP, "lora_exotic.safetensors")
    save_file({"some.other.arch.lora_A.weight": torch.zeros(8, 999),
               "some.other.arch.lora_B.weight": torch.zeros(999, 8)}, p)
    assert P._flux2_lora_hidden_dim(p) is None
    assert P._lora_unsupported(p) is None
    print("OK test_a_lora_without_a_signature_is_never_filtered")


def test_a_lokr_has_no_lora_signature():
    """A LoKr has neither lora_A nor lora_B: the variant guard must let it through,
    it is the merging (_merge_lokr) that takes care of it."""
    p = os.path.join(TMP, "lokr_no_sig.safetensors")
    save_file({f"diffusion_model.double_blocks.{i}.img_attn.proj.lokr_w{j}":
               torch.zeros(4, 4) for i in range(3) for j in (1, 2)}, p)
    assert P._flux2_lora_hidden_dim(p) is None
    print("OK test_a_lokr_has_no_lora_signature")


def test_every_lora_dialect_is_recognised():
    """The bug that made the guard inert: it only read the diffusers layout,
    whereas the real library is in the original FLUX layout. Recognising the
    matrix by SEGMENT ('lora_A' somewhere in the key) covers all three."""
    for layout in _LORA_LAYOUTS:
        for label, dim in DIMS.items():
            p = _lora(f"dialect_{layout}_{label}.safetensors", dim, layout=layout)
            got = P._flux2_lora_hidden_dim(p)
            assert got == dim, (layout, label, got)
    print("OK test_every_lora_dialect_is_recognised")


def test_derived_widths_do_not_fool_the_detector():
    """A fused qkv is 3x wide, an mlp 4x. Those values are not in
    _FLUX2_VARIANTS and must therefore be ignored, not taken for a dimension."""
    p = os.path.join(TMP, "derived.safetensors")
    save_file({
        # only that pair carries the real dimension (4096)
        "diffusion_model.double_blocks.0.img_attn.proj.lora_A.weight": torch.zeros(16, 4096),
        "diffusion_model.double_blocks.0.img_attn.proj.lora_B.weight": torch.zeros(4096, 16),
        # these carry 3x and 4x: to be ignored
        "diffusion_model.double_blocks.0.img_attn.qkv.lora_B.weight": torch.zeros(12288, 16),
        "diffusion_model.double_blocks.0.img_mlp.0.lora_B.weight": torch.zeros(16384, 16),
    }, p)
    assert P._flux2_lora_hidden_dim(p) == 4096, P._flux2_lora_hidden_dim(p)
    print("OK test_derived_widths_do_not_fool_the_detector")


if __name__ == "__main__":
    for fn in (test_hidden_dim_read_from_header,
               test_a_flux2_file_is_refused_and_a_flux3_one_passes_whatever_the_base,
               test_summary_names_the_studio_that_runs_the_file_once,
               test_unknown_base_never_discards, test_bogus_shape_is_not_taken_for_a_hidden_dim,
               test_file_without_signature_is_not_filtered,
               test_a_lora_declares_its_variant_through_its_shapes,
               test_a_flux2_lora_is_refused_in_one_sentence,
               test_a_lora_without_a_signature_is_never_filtered,
               test_a_lokr_has_no_lora_signature,
               test_every_lora_dialect_is_recognised,
               test_derived_widths_do_not_fool_the_detector):
        fn()
    print("All 4B/9B variant tests passed.")
