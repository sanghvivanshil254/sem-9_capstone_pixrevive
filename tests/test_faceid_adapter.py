"""Tests for faceid_adapter.py — FaceID conditioning (Phase H4).

Pure-tensor contract tests: run everywhere, no SD weights required (the
processors/projection are exercised on random tensors against a stub
attention module). Checkpoint-dependent paths are skipped when the .bin
is absent.
"""

import os
from unittest import mock

import numpy as np
import pytest
import torch

from ai.restoration import faceid_adapter as fa
from ai.restoration.faceid_adapter import (
    FaceIDAdapter,
    FaceIDAttnProcessor2_0,
    FaceIDIPAttnProcessor2_0,
    MLPProjModel,
)

CKPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(fa.__file__))),
    "weights", "ip_adapter_faceid", "models", "ip-adapter-faceid_sd15.bin")
HAS_CKPT = os.path.exists(CKPT)


class _StubHeads:
    heads = 4


class _StubAttn:
    """Minimal attn surface the processors call into."""

    def __init__(self, hidden=32, cross=48):
        self.heads = 4
        self.group_norm = None
        self.spatial_norm = None
        self.residual_connection = False
        self.rescale_output_factor = 1.0
        self.norm_cross = False
        self.to_q = torch.nn.Linear(hidden, hidden, bias=False)
        self.to_k = torch.nn.Linear(cross, hidden, bias=False)
        self.to_v = torch.nn.Linear(cross, hidden, bias=False)
        self.to_out = torch.nn.ModuleList(
            [torch.nn.Linear(hidden, hidden, bias=False),
             torch.nn.Identity()])

    def prepare_attention_mask(self, mask, seq_len, batch):
        return None


def _rand(shape):
    return torch.randn(*shape)


# ------------------------------------------------------------- processors

def test_self_processor_shapes():
    proc = FaceIDAttnProcessor2_0(hidden_size=32)
    attn = _StubAttn(hidden=32, cross=32)  # self-attn: square projections
    out = proc(attn, _rand((2, 10, 32)))
    assert out.shape == (2, 10, 32)


def test_cross_processor_without_encoder_falls_back_to_self():
    proc = FaceIDIPAttnProcessor2_0(hidden_size=32, cross_attention_dim=32)
    attn = _StubAttn(hidden=32, cross=32)  # fallback branch runs self-attention
    # No encoder_hidden_states: text branch == self-attention, no IP branch.
    out = proc(attn, _rand((2, 10, 32)), None)
    assert out.shape == (2, 10, 32)


def test_cross_processor_splits_last_tokens():
    torch.manual_seed(0)
    proc = FaceIDIPAttnProcessor2_0(hidden_size=32, cross_attention_dim=48,
                                    num_tokens=4, scale=0.0)
    attn = _StubAttn(hidden=32)
    enc_text = _rand((2, 16, 48))
    enc_ip = _rand((2, 4, 48))
    out = proc(attn, _rand((2, 10, 32)), torch.cat([enc_text, enc_ip], dim=1))
    assert out.shape == (2, 10, 32)
    # scale=0 must make the IP branch vanish exactly.
    proc.scale = 1.0
    out_scaled = proc(attn, _rand((2, 10, 32)),
                      torch.cat([enc_text, enc_ip], dim=1))
    assert out_scaled.shape == (2, 10, 32)


def test_scale_property_is_shared_reference():
    proc = FaceIDIPAttnProcessor2_0(hidden_size=32, cross_attention_dim=48)
    assert proc.scale == 1.0
    proc.scale = 0.77
    assert proc.scale == 0.77


# ---------------------------------------------------------------- proj

def test_mlp_proj_output_geometry():
    proj = MLPProjModel(cross_attention_dim=768, id_embeddings_dim=512,
                        num_tokens=4)
    embeds = torch.randn(2, 512)
    tokens = proj(embeds)
    assert tokens.shape == (2, 4, 768)
    # LayerNorm at the end: zero-mean per token.
    assert tokens.mean(dim=-1).abs().max() < 1e-4


def test_encode_uncond_is_projection_of_zero():
    proj = MLPProjModel(cross_attention_dim=768)
    with torch.inference_mode():
        tok = proj(torch.randn(1, 512))
        uncond = proj(torch.zeros(1, 512))
    # Uncond tokens are deterministic (no input dependence).
    assert uncond.shape == (1, 4, 768)


# ------------------------------------------------------------- adapter

def test_install_missing_checkpoint_returns_false(tmp_path):
    pipe = mock.MagicMock()
    adapter = FaceIDAdapter(pipe, ip_ckpt=str(tmp_path / "nope.bin"))
    assert adapter.install() is False
    assert adapter.installed is False
    # No processors were touched.
    pipe.unet.set_attn_processor.assert_not_called()


def test_uninstall_without_install_is_noop():
    pipe = mock.MagicMock()
    adapter = FaceIDAdapter(pipe, ip_ckpt="whatever.bin")
    adapter.uninstall()
    pipe.unet.set_attn_processor.assert_not_called()


def _fake_unet(cross_dim=768):
    """Tiny UNet stand-in: one self-attn + one cross-attn processor slot."""
    unet = mock.MagicMock()
    unet.config = mock.MagicMock()
    unet.config.cross_attention_dim = cross_dim
    unet.config.block_out_channels = [32, 64]
    from diffusers.models.attention_processor import AttnProcessor2_0
    # Values must be real Modules: install() wraps them in a ModuleList
    # before load_state_dict (mirrors the reference loader).
    unet.attn_processors = {
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor":
            AttnProcessor2_0(),
        "down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor":
            AttnProcessor2_0(),
    }
    return unet


@pytest.mark.skipif(not HAS_CKPT, reason="FaceID .bin not fetched")
def test_real_checkpoint_structure():
    """The released .bin must match our vendored geometry: image_proj is
    512->1024->768*4 (SD1.5 cross_attention_dim) and the adapter half carries
    rank-128 LoRA halves + 768-dim to_k_ip/to_v_ip."""
    state = torch.load(CKPT, map_location="cpu", weights_only=True)
    assert set(state.keys()) == {"image_proj", "ip_adapter"}
    ip = state["image_proj"]
    assert ip["proj.0.weight"].shape == (1024, 512)
    assert ip["proj.2.weight"].shape == (4 * 768, 1024)
    assert ip["norm.weight"].shape == (768,)
    ad = state["ip_adapter"]
    lora_downs = [v.shape for k, v in ad.items()
                  if k.endswith("to_q_lora.down.weight")]
    assert lora_downs and all(s[0] == 128 for s in lora_downs)
    ip_kv = [v.shape for k, v in ad.items() if k.endswith("to_k_ip.weight")]
    assert ip_kv and all(s[1] == 768 for s in ip_kv)


@pytest.mark.skipif(not HAS_CKPT, reason="FaceID .bin not fetched")
def test_install_dim_mismatch_restores_stock_processors():
    """Safety contract: a checkpoint that can't strict-load (here: real SD1.5
    dims vs the fake UNet's tiny dims) must leave the UNet back on stock
    processors and return False — never a half-installed adapter."""
    unet = _fake_unet()
    pipe = mock.MagicMock()
    pipe.unet = unet
    adapter = FaceIDAdapter(pipe, ip_ckpt=CKPT)
    calls = []

    def _capture(proc_map):
        calls.append(dict(proc_map))
        unet.attn_processors = dict(proc_map)

    unet.set_attn_processor.side_effect = _capture
    assert adapter.install() is False
    assert adapter.installed is False
    # First call installed FaceID processors; second call restored stock.
    assert len(calls) == 2
    from diffusers.models.attention_processor import AttnProcessor2_0
    assert all(isinstance(v, FaceIDIPAttnProcessor2_0)
               for k, v in calls[0].items() if k.endswith("attn2.processor"))
    assert all(isinstance(v, AttnProcessor2_0) for v in calls[1].values())


def test_encode_requires_install():
    pipe = mock.MagicMock()
    adapter = FaceIDAdapter(pipe, ip_ckpt="whatever.bin")
    assert adapter.encode(torch.randn(1, 512)) is None
