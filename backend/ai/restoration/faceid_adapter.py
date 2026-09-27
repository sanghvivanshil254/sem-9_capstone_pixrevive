"""faceid_adapter.py — FaceID-class identity conditioning on the SD1.5 pipeline.

Phase H4 (docs/HYBRID_DIFFUSION_DESIGN.md, H4 addendum). Loads
h94/IP-Adapter-FaceID (ip-adapter-faceid_sd15.bin) onto the already-loaded
StableDiffusionInpaintPipeline and converts a 512-d ArcFace identity
embedding (faceid_embedder.py) into 4 cross-attention tokens that are
CONCATENATED to the text embeddings — unlike the CLIP IP-Adapter, whose
image tokens ride a parallel path and are keyed to a *damaged crop*.

Vendored reimplementation (project rule: never pip-install IP-Adapter — it
would drag a second diffusers/basicsr dependency tree). Reference:
tencent-ailab/IP-Adapter ip_adapter_faceid.py + attention_processor_faceid.py
(Apache-2.0), adjusted for diffusers 0.40:

  * SDPA (torch 2.x) attention, matching LoRAIPAttnProcessor2_0.
  * LoRA rank 128 — the released SD1.5 FaceID adapter is trained at that
    rank, so ``ip_layers.load_state_dict`` stays strict (no missing keys).
  * Self-attention processors get only the LoRA halves; cross-attention
    processors additionally carry to_k_ip / to_v_ip (768 -> hidden).
  * The cond/uncond split is positional (last num_tokens of
    encoder_hidden_states), which is why the SAME processors must serve
    both halves of classifier-free guidance — a split batch arrives as one
    tensor and is sliced by sequence position. This is also why the
    adapter MUST be uninstalled when there is no identity embedding: with
    no trailing tokens the split misaligns and K/V shapes mismatch.

Adapter-state rules mirror the CLIP-adapter sync in reimaginer._generate:
FaceID replaces the CLIP adapter (the processor slot is exclusive), and the
adapters are removed whenever an image yields no identity embedding.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers.models.lora import LoRALinearLayer

logger = logging.getLogger(__name__)

NUM_TOKENS = 4
LORA_RANK = 128


class MLPProjModel(torch.nn.Module):
    """image_proj: 512-d ArcFace embed -> 4 x cross_attention_dim tokens.

    Verbatim from the reference implementation (the .bin's ``image_proj``
    state dict matches this exactly).
    """

    def __init__(self, cross_attention_dim: int = 768,
                 id_embeddings_dim: int = 512, num_tokens: int = NUM_TOKENS):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim * 2,
                            cross_attention_dim * num_tokens),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, id_embeds: torch.Tensor) -> torch.Tensor:
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        return self.norm(x)


class FaceIDAttnProcessor2_0(torch.nn.Module):
    """Self-attention: only the FaceID LoRA halves touch the computation."""

    def __init__(self, hidden_size: int, rank: int = LORA_RANK,
                 network_alpha: int | None = None, lora_scale: float = 1.0):
        super().__init__()
        self.rank = rank
        self.lora_scale = lora_scale
        self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, rank,
                                         network_alpha)
        self.to_k_lora = LoRALinearLayer(hidden_size, hidden_size, rank,
                                         network_alpha)
        self.to_v_lora = LoRALinearLayer(hidden_size, hidden_size, rank,
                                         network_alpha)
        self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, rank,
                                           network_alpha)

    def __call__(self, attn: Any, hidden_states: torch.Tensor,
                 encoder_hidden_states: torch.Tensor | None = None,
                 attention_mask: torch.Tensor | None = None,
                 temb: torch.Tensor | None = None,
                 *args, **kwargs) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None
            else encoder_hidden_states.shape)
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size)
        if attn.group_norm is not None:
            hidden_states = hidden_states.transpose(1, 2)
            hidden_states = attn.group_norm(hidden_states).transpose(1, 2)

        query = attn.to_q(hidden_states) \
            + self.lora_scale * self.to_q_lora(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        key = attn.to_k(encoder_hidden_states) \
            + self.lora_scale * self.to_k_lora(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states) \
            + self.lora_scale * self.to_v_lora(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        hidden_states = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask,
            dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states) \
            + self.lora_scale * self.to_out_lora(hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


class FaceIDIPAttnProcessor2_0(torch.nn.Module):
    """Cross-attention: LoRA halves + the identity-token K/V projections.

    The cond/uncond split is positional: the LAST ``num_tokens`` entries of
    encoder_hidden_states are the identity tokens, the rest is text.
    """

    def __init__(self, hidden_size: int,
                 cross_attention_dim: int | None = None,
                 rank: int = LORA_RANK, network_alpha: int | None = None,
                 lora_scale: float = 1.0, scale: float = 1.0,
                 num_tokens: int = NUM_TOKENS):
        super().__init__()
        self.rank = rank
        self.lora_scale = lora_scale
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, rank,
                                         network_alpha)
        self.to_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size,
                                         hidden_size, rank, network_alpha)
        self.to_v_lora = LoRALinearLayer(cross_attention_dim or hidden_size,
                                         hidden_size, rank, network_alpha)
        self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, rank,
                                           network_alpha)
        self.to_k_ip = torch.nn.Linear(
            cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = torch.nn.Linear(
            cross_attention_dim or hidden_size, hidden_size, bias=False)

    def __call__(self, attn: Any, hidden_states: torch.Tensor,
                 encoder_hidden_states: torch.Tensor | None = None,
                 attention_mask: torch.Tensor | None = None,
                 temb: torch.Tensor | None = None,
                 *args, **kwargs) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None
            else encoder_hidden_states.shape)
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size)
        if attn.group_norm is not None:
            hidden_states = hidden_states.transpose(1, 2)
            hidden_states = attn.group_norm(hidden_states).transpose(1, 2)

        query = attn.to_q(hidden_states) \
            + self.lora_scale * self.to_q_lora(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
            ip_hidden_states = None
        else:
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states, ip_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:, :],
            )

        key = attn.to_k(encoder_hidden_states) \
            + self.lora_scale * self.to_k_lora(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states) \
            + self.lora_scale * self.to_v_lora(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        hidden_states = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask,
            dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if ip_hidden_states is not None:
            ip_key = self.to_k_ip(ip_hidden_states)
            ip_value = self.to_v_ip(ip_hidden_states)
            ip_key = ip_key.view(
                batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_value = ip_value.view(
                batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_hidden = torch.nn.functional.scaled_dot_product_attention(
                query, ip_key, ip_value, attn_mask=None,
                dropout_p=0.0, is_causal=False)
            ip_hidden = ip_hidden.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_dim).to(query.dtype)
            hidden_states = hidden_states + self.scale * ip_hidden

        hidden_states = attn.to_out[0](hidden_states) \
            + self.lora_scale * self.to_out_lora(hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


def _hidden_size_for(name: str, unet: Any) -> int:
    """block_out_channels lookup, identical to the reference set_ip_adapter."""
    boc = unet.config.block_out_channels
    if name.startswith("mid_block"):
        return boc[-1]
    block_id = int(name.split(".")[1])
    if name.startswith("up_blocks"):
        return list(reversed(boc))[block_id]
    return boc[block_id]


class FaceIDAdapter:
    """Install/uninstall ip-adapter-faceid_sd15.bin on a diffusers SD1.5 pipe.

    install() replaces every attention processor (CLIP-adapter included —
    the processor slot is exclusive) with FaceID LoRA variants and loads
    the .bin strictly. encode() projects identity embeddings to the 4
    trailing tokens the processors split off. uninstall() restores the
    stock AttnProcessor2_0 set; reimaginer then re-arms the CLIP adapter
    through its existing two-way sync when a face reference exists.
    """

    def __init__(self, pipe: Any, ip_ckpt: str, lora_path: str | None = None,
                 rank: int = LORA_RANK, num_tokens: int = NUM_TOKENS) -> None:
        self.pipe = pipe
        self.ip_ckpt = ip_ckpt
        self.lora_path = lora_path
        self.rank = rank
        self.num_tokens = num_tokens
        self._proj: MLPProjModel | None = None
        self._installed = False

    # -- lifecycle -----------------------------------------------------------

    @property
    def installed(self) -> bool:
        return self._installed

    def install(self) -> bool:
        """Swap processors + load the .bin. False = weights absent/broken."""
        if self._installed:
            return True
        if not os.path.exists(self.ip_ckpt):
            logger.warning("FaceID: adapter checkpoint missing (%s)", self.ip_ckpt)
            return False
        unet = self.pipe.unet
        cross_dim = unet.config.cross_attention_dim
        procs: dict[str, torch.nn.Module] = {}
        for name in unet.attn_processors:
            hidden = _hidden_size_for(name, unet)
            is_cross = not name.endswith("attn1.processor")
            if is_cross:
                procs[name] = FaceIDIPAttnProcessor2_0(
                    hidden, cross_dim, rank=self.rank,
                    num_tokens=self.num_tokens)
            else:
                procs[name] = FaceIDAttnProcessor2_0(hidden, rank=self.rank)
        unet.set_attn_processor(procs)

        state = torch.load(self.ip_ckpt, map_location="cpu", weights_only=True)
        proj = MLPProjModel(cross_dim, 512, self.num_tokens)
        proj.load_state_dict(state["image_proj"])
        ip_layers = torch.nn.ModuleList(unet.attn_processors.values())
        try:
            ip_layers.load_state_dict(state["ip_adapter"])  # strict on purpose
        except RuntimeError as exc:
            # Rank/dim mismatch or foreign checkpoint: restore stock processors.
            unet.set_attn_processor({n: AttnProcessor2_0()
                                     for n in unet.attn_processors})
            logger.warning("FaceID: checkpoint incompatible — %s", exc)
            return False
        self._proj = proj
        self._installed = True
        # The CLIP loader marks the UNet to REQUIRE image embeds through
        # TWO config attributes (smoke-measured 2026-09-27, crash site
        # unet_2d_condition.process_encoder_hidden_states):
        #   * config.encoder_hid_dim_type = "ip_image_proj" + the
        #     encoder_hid_proj module — UNet.forward then demands
        #     added_cond_kwargs["image_embeds"] and crashes ("argument of
        #     type 'NoneType' is not iterable")
        #   * config.addition_embed_type = "ip_adapter" (same class of
        #     demand, different branch)
        # FaceID conditions through pre-concatenated prompt_embeds instead.
        # Clear both; the CLIP re-arm path (pipe.load_ip_adapter) restores
        # them when it re-arms.
        if getattr(unet.config, "encoder_hid_dim_type", None) == "ip_image_proj":
            unet.config.encoder_hid_dim_type = None
            unet.encoder_hid_proj = None
            logger.info("FaceID: cleared UNet encoder_hid_dim_type 'ip_image_proj'")
        if getattr(unet.config, "addition_embed_type", None) == "ip_adapter":
            unet.config.addition_embed_type = None
            logger.info("FaceID: cleared UNet addition_embed_type 'ip_adapter'")
        logger.info("FaceID: adapter installed (rank %d, %d tokens)",
                    self.rank, self.num_tokens)
        return True

    def uninstall(self) -> None:
        if not self._installed:
            return
        self.pipe.unet.set_attn_processor(
            {n: AttnProcessor2_0() for n in self.pipe.unet.attn_processors})
        self._proj = None
        self._installed = False
        logger.info("FaceID: adapter uninstalled (stock processors restored)")

    def set_scale(self, scale: float) -> None:
        for p in self.pipe.unet.attn_processors.values():
            if isinstance(p, FaceIDIPAttnProcessor2_0):
                p.scale = float(scale)

    # -- inference -------------------------------------------------------------

    @torch.inference_mode()
    def encode(self, faceid_embed: torch.Tensor,
               device: torch.device | str | None = None,
               dtype: torch.dtype | None = None,
               ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """(B,512) ArcFace embeds -> (image_tokens, uncond_tokens).

        Each is (B, num_tokens, cross_attention_dim). Returns None when the
        adapter is not installed. Under sequential CPU offload pass
        device="cpu" (accelerate's hooks stream the weights); otherwise pass
        the pipe's execution device so the concat with prompt embeds is
        device-consistent.
        """
        if not self._installed or self._proj is None:
            return None
        if faceid_embed.dim() == 1:
            faceid_embed = faceid_embed.unsqueeze(0)
        if device is not None:
            faceid_embed = faceid_embed.to(device)
        if dtype is not None:
            faceid_embed = faceid_embed.to(dtype)
        tokens = self._proj(faceid_embed)
        uncond = self._proj(torch.zeros_like(faceid_embed))
        return tokens, uncond
