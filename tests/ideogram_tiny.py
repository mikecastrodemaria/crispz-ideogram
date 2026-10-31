"""A tiny random Ideogram 4 pipeline for CPU tests (no gated download).

Real architecture, toy sizes: Ideogram4Transformer2DModel x2 (1 layer, 2 heads of 16),
a 36-layer Qwen3-VL text encoder 16 wide (the pipeline taps layers 0..35, so the depth
must stay), the tiny AutoencoderKLFlux2 of the flux3 tests, and the real Qwen3-VL-8B
tokenizer (public, a few MB). llm_features_dim = 13 taps x 16 = 208.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TOKENIZER_REPO = "Qwen/Qwen3-VL-8B-Instruct"


def tiny_components(seed=0):
    from diffusers import (AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler,
                           Ideogram4Transformer2DModel)
    from transformers import AutoTokenizer, Qwen3VLConfig, Qwen3VLModel
    torch.manual_seed(seed)
    hid = 16
    tcfg = dict(in_channels=128, num_layers=1, attention_head_dim=16, num_attention_heads=2,
                intermediate_size=32, adaln_dim=16, llm_features_dim=13 * hid,
                mrope_section=(2, 1, 1))
    tr = Ideogram4Transformer2DModel(**tcfg)
    utr = Ideogram4Transformer2DModel(**tcfg)
    enc = Qwen3VLModel(Qwen3VLConfig(
        text_config=dict(hidden_size=hid, intermediate_size=32, num_hidden_layers=36,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                         vocab_size=151936,
                         rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2],
                                       "mrope_interleaved": True}),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
                           out_hidden_size=hid, deepstack_visual_indexes=[0]))).eval()
    vae = AutoencoderKLFlux2(block_out_channels=[32, 32, 32, 32], latent_channels=32,
                             norm_num_groups=8, layers_per_block=1)
    tok = AutoTokenizer.from_pretrained(TOKENIZER_REPO)
    return dict(scheduler=FlowMatchEulerDiscreteScheduler(), vae=vae, text_encoder=enc,
                tokenizer=tok, transformer=tr, unconditional_transformer=utr)


def tiny_pipe(seed=0):
    from diffusers import Ideogram4Pipeline
    import cz_ideogram
    return cz_ideogram.IdeogramPipe(Ideogram4Pipeline(**tiny_components(seed)), repo="tiny")
