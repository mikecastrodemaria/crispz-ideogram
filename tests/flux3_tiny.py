"""Miniature FLUX.3 components for the CPU tests (no download, random weights).

The real model is 5.6B parameters behind a gated repo; the tests need the SHAPES and the
control flow, not the weights. So: a 1+1 block transformer 16 wide, a VAE with 8-channel
blocks, a stub text encoder that returns random embeddings of the right width (8 hidden
states x TEXT_HIDDEN, the way FLUX.3 stacks Qwen3-VL's), all wired into the real diffusers
modular blocks through cz_flux3.build_blocks, so the adapter is exercised against the real
block chain and not a mock of it.

Shared by tests/test_flux3_adapter.py and tests/test_flux3_wiring.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import cz_flux3  # noqa: E402

TEXT_HIDDEN = 16            # the stub encoder's width; FLUX.3 stacks 8 layers of it
SEQ = 80                    # one padding block of Qwen3-VL (TEXT_PAD_MULTIPLE)


def tiny_vae():
    from diffusers import AutoencoderKLFlux2
    # 4 levels (so vae_scale_factor stays 8, as the real one) and 32 latent channels (the
    # 2x2 packing gives the transformer's 128 input channels). Everything else is small.
    return AutoencoderKLFlux2(block_out_channels=(8, 8, 8, 8), latent_channels=32,
                              norm_num_groups=8, layers_per_block=1, sample_size=64,
                              mid_block_add_attention=False).eval()


def tiny_transformer(base=False):
    from diffusers import Flux3Transformer2DModel
    # flow_map=True is the distilled checkpoint's shape (the transformer also takes the
    # NEXT timestep); the base one is a plain flow-matching model.
    return Flux3Transformer2DModel(in_channels=128, cond_in_channels=128, num_layers=1,
                                   num_single_layers=1, attention_head_dim=8,
                                   num_attention_heads=2,
                                   joint_attention_dim=8 * TEXT_HIDDEN, timestep_channels=32,
                                   axes_dims_rope=(2, 2, 2, 2), flow_map=not base).eval()


class StubPipe(cz_flux3.Flux3Pipe):
    """A Flux3Pipe whose text encoding is random: the same shapes as the real one, no
    Qwen3-VL. A SUBCLASS and not an instance attribute on purpose: Flux3Pipe forwards
    unknown attribute writes to the modular pipeline, so `pipe.encode_prompt = ...` would
    land on the wrong object."""
    encode_calls = 0

    @torch.no_grad()
    def encode_prompt(self, prompt, device=None, max_sequence_length=None, **_):
        type(self).encode_calls += 1
        g = torch.Generator().manual_seed(abs(hash(prompt or "")) % 10_000)
        return (torch.randn(1, SEQ, 8 * TEXT_HIDDEN, generator=g),)


def tiny_pipe(base=False, seed=0):
    """A ready Flux3Pipe on the CPU, float32, distilled or base variant."""
    torch.manual_seed(seed)
    blocks = cz_flux3.build_blocks(base=base)
    mp = blocks.init_pipeline()
    mp.update_components(transformer=tiny_transformer(base), vae=tiny_vae())
    return StubPipe(mp, None, None, repo="tiny/flux3" + ("-base" if base else ""),
                    variant="base" if base else "distilled")


def tiny_qwen3_vl(seed=0):
    """A random Qwen3VLForConditionalGeneration with 33 layers of width 16: enough for
    the 8 hidden states FLUX.3 reads (the last one is layer 32). The REAL vocabulary
    size, so the real processor's token ids fit."""
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    torch.manual_seed(seed)
    cfg = Qwen3VLConfig(
        text_config=dict(hidden_size=TEXT_HIDDEN, intermediate_size=32, num_hidden_layers=33,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                         vocab_size=151936, max_position_embeddings=4096),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
                           out_hidden_size=16, patch_size=16, spatial_merge_size=2,
                           temporal_patch_size=2, in_channels=3))
    return Qwen3VLForConditionalGeneration(cfg).eval()
