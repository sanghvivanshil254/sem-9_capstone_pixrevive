"""deblurrer.py — T-3 deblur stage (Phase 4B.6), vendored minimal Restormer.

Restormer: Efficient Transformer for High-Resolution Image Restoration
(Syed Waqas Zamir et al., CVPR 2022, https://arxiv.org/abs/2111.09881).
Architecture re-typed from the official restormer_arch.py with einops
rearrange calls replaced by equivalent torch reshape/permute (PixRevive
carries no einops dependency). Weights: motion_deblurring.pth from the
official GitHub release (swz30/Restormer v1.0), fetched by
scripts/fetch_deblur_weights.py into backend/ai/weights/restormer/ —
per the runtime rules (GitHub source, urllib, weights under
backend/ai/weights/, never a user cache).

Inference is tiled with reflect padding + seam blending so any input size
works on the 4 GB GPU. The stage is STRICTLY gated (see estimate_optical_blur
and the pipeline wiring): it must never fire on sharp inputs.
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "weights", "restormer", "motion_deblurring.pth",
)

# Restormer defaults (identical to the released checkpoint).
_DIM = 48
_BLOCKS = [4, 6, 6, 8]
_HEADS = [1, 2, 4, 8]
_FFN_EXPANSION = 2.66
_REFINEMENT_BLOCKS = 4

_TILE = 256          # tile size for tiled inference
_TILE_OVERLAP = 32   # reflect pad overlap so seams have context


# --------------------------------------------------------------------------
# Vendored architecture (einops-free)
# --------------------------------------------------------------------------
class _BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class _WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class _LayerNorm(nn.Module):
    """Channel layer-norm over (h, w) — replaces the einops 3d/4d round trip."""

    def __init__(self, dim, layer_norm_type):
        super().__init__()
        if layer_norm_type == "BiasFree":
            self.body = _BiasFreeLayerNorm(dim)
        else:
            self.body = _WithBiasLayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        b, c = x.shape[0], x.shape[1]
        flat = x.reshape(b, c, h * w).permute(0, 2, 1)
        out = self.body(flat)
        return out.permute(0, 2, 1).reshape(b, c, h, w)


class _FeedForward(nn.Module):
    """Gated-Dconv FFN."""

    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden * 2, hidden * 2, kernel_size=3, stride=1, padding=1,
            groups=hidden * 2, bias=bias,
        )
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class _Attention(nn.Module):
    """Multi-DConv Head Transposed Self-Attention."""

    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1,
            groups=dim * 3, bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        head_c = c // self.num_heads
        # 'b (head c) h w -> b head c (h w)'
        q = q.reshape(b, self.num_heads, head_c, h * w)
        k = k.reshape(b, self.num_heads, head_c, h * w)
        v = v.reshape(b, self.num_heads, head_c, h * w)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        # 'b head c (h w) -> b (head c) h w'
        out = out.reshape(b, self.num_heads * head_c, h, w)
        return self.project_out(out)


class _TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, layer_norm_type):
        super().__init__()
        self.norm1 = _LayerNorm(dim, layer_norm_type)
        self.attn = _Attention(dim, num_heads, bias)
        self.norm2 = _LayerNorm(dim, layer_norm_type)
        self.ffn = _FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class _OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class _Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class _Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Restormer(nn.Module):
    """Vendored Restormer (state_dict-compatible with the official release)."""

    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=_DIM,
        num_blocks=None,
        num_refinement_blocks=_REFINEMENT_BLOCKS,
        heads=None,
        ffn_expansion_factor=_FFN_EXPANSION,
        bias=False,
        LayerNorm_type="WithBias",
        dual_pixel_task=False,
    ):
        super().__init__()
        num_blocks = num_blocks or _BLOCKS
        heads = heads or _HEADS
        self.patch_embed = _OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = nn.Sequential(*[
            _TransformerBlock(dim=dim, num_heads=heads[0],
                              ffn_expansion_factor=ffn_expansion_factor,
                              bias=bias, layer_norm_type=LayerNorm_type)
            for _ in range(num_blocks[0])
        ])
        self.down1_2 = _Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            _TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1],
                              ffn_expansion_factor=ffn_expansion_factor,
                              bias=bias, layer_norm_type=LayerNorm_type)
            for _ in range(num_blocks[1])
        ])
        self.down2_3 = _Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential(*[
            _TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2],
                              ffn_expansion_factor=ffn_expansion_factor,
                              bias=bias, layer_norm_type=LayerNorm_type)
            for _ in range(num_blocks[2])
        ])
        self.down3_4 = _Downsample(int(dim * 2 ** 2))
        self.latent = nn.Sequential(*[
            _TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3],
                              ffn_expansion_factor=ffn_expansion_factor,
                              bias=bias, layer_norm_type=LayerNorm_type)
            for _ in range(num_blocks[3])
        ])
        self.up4_3 = _Upsample(int(dim * 2 ** 3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2),
                                            kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            _TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2],
                              ffn_expansion_factor=ffn_expansion_factor,
                              bias=bias, layer_norm_type=LayerNorm_type)
            for _ in range(num_blocks[2])
        ])
        self.up3_2 = _Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1),
                                            kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            _TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1],
                              ffn_expansion_factor=ffn_expansion_factor,
                              bias=bias, layer_norm_type=LayerNorm_type)
            for _ in range(num_blocks[1])
        ])
        self.up2_1 = _Upsample(int(dim * 2 ** 1))
        self.decoder_level1 = nn.Sequential(*[
            _TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0],
                              ffn_expansion_factor=ffn_expansion_factor,
                              bias=bias, layer_norm_type=LayerNorm_type)
            for _ in range(num_blocks[0])
        ])
        self.refinement = nn.Sequential(*[
            _TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0],
                              ffn_expansion_factor=ffn_expansion_factor,
                              bias=bias, layer_norm_type=LayerNorm_type)
            for _ in range(num_refinement_blocks)
        ])
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels,
                                kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img
        return out_dec_level1


# --------------------------------------------------------------------------
# Blur measurement — the STRICT gate
# --------------------------------------------------------------------------
# Gate design was probe-driven (.freebuff probe, deleted post-phase; the
# numbers live in the test fixtures). Two independent fire paths:
#
# PATH A — defocus/out-of-focus: EDGE-SPREAD acutance. Strong-edge cores are
#   the 99th-percentile gradient pixels; spread = gradient energy in the two
#   pixel rings around those cores relative to the core energy. A lens PSF
#   widens EVERY edge, so spread rises with blur radius (probe: sigma 1.5
#   -> 0.48, 2.5 -> 0.64, 4 -> 0.76) while the real corpus tops out at 0.456
#   (soft foliage / flake-textured edges — natural material, NOT lens blur;
#   deblurring those would hallucinate detail). Threshold 0.47 sits in the
#   measured gap. Laplacian variance alone was REJECTED by the probe: it
#   false-fired on sharp-but-low-detail photos (case11/4 measured 112) and
#   missed motion blur entirely (perpendicular edge energy dominates).
#
# PATH B — motion blur: H/V gradient-variance anisotropy. Directional blur
#   kills gradient energy along the blur axis; the corpus max is 1.69 while
#   9-15px motion fixtures measure 2.4-5.6. Threshold 2.2.

def _edge_spread(gray: np.ndarray) -> tuple[Optional[float], int]:
    """Return (spread, strong_edge_count); spread None when too few edges."""
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    G = np.sqrt(gx ** 2 + gy ** 2)
    thr = np.percentile(G, 99.0)
    if thr <= 8.0:
        return None, 0
    core = (G >= thr).astype(np.uint8)
    k3 = np.ones((3, 3), np.uint8)
    k5 = np.ones((5, 5), np.uint8)
    ring1 = cv2.dilate(core, k3).astype(bool) & ~core.astype(bool)
    ring2 = cv2.dilate(core, k5).astype(bool) & ~cv2.dilate(core, k3).astype(bool)
    e_core = float(G[core.astype(bool)].mean())
    e_r1 = float(G[ring1].mean()) if ring1.any() else 0.0
    e_r2 = float(G[ring2].mean()) if ring2.any() else 0.0
    if e_core <= 1e-9:
        return None, 0
    return (e_r1 + e_r2) / (2.0 * e_core), int(core.sum())


def _directional_ratio(gray: np.ndarray) -> float:
    lx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    ly = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    var_x = float(np.var(lx)) + 1e-9
    var_y = float(np.var(ly)) + 1e-9
    return max(var_x / var_y, var_y / var_x)


def estimate_optical_blur(img: np.ndarray) -> dict:
    """Measure optical blur with the two-path probe-calibrated gate.

    Returns {edge_spread, strong_edges, directional_ratio, is_optical}.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    spread, n_edges = _edge_spread(gray)
    ratio = _directional_ratio(gray)

    path_a = spread is not None and spread >= BLUR_SPREAD_GATE and n_edges >= MIN_STRONG_EDGES
    # Mean ABSOLUTE gradient — signed Sobel means cancel to ~0 on any image.
    mean_grad = float(np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)).mean())
    path_b = ratio >= MOTION_RATIO_GATE and mean_grad >= MIN_MEAN_GRADIENT
    return {
        "edge_spread": None if spread is None else round(float(spread), 3),
        "strong_edges": n_edges,
        "directional_ratio": round(ratio, 3),
        "is_optical": bool(path_a or path_b),
        "fire_path": ("A" if path_a else "") + ("B" if path_b else "") or None,
    }


# Gate constants (probe-calibrated on the corpus; see tests).
BLUR_SPREAD_GATE = 0.47    # corpus max 0.456; defocus sigma>=1.5 measures 0.48+
MIN_STRONG_EDGES = 150     # below this there is not enough edge evidence
MOTION_RATIO_GATE = 2.2    # corpus max 1.69; motion fixtures 2.4+
MIN_MEAN_GRADIENT = 3.0    # path B needs actual edges to be directional


# --------------------------------------------------------------------------
# Tiled inference
# --------------------------------------------------------------------------
class ImageDeblurrer:
    """Gated Restormer deblur stage resource (loaded lazily, like LaMa)."""

    def __init__(self):
        self.model: Optional[Restormer] = None
        self._device: Optional[str] = None

    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self, device: str = "cuda") -> None:
        if self.model is not None and self._device == device:
            return
        if not os.path.exists(WEIGHTS_PATH):
            raise FileNotFoundError(
                f"Deblur weights missing: {WEIGHTS_PATH} — run "
                f"scripts/fetch_deblur_weights.py (or admin upload) first."
            )
        model = Restormer()
        state = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
        if "params" in state and isinstance(state["params"], dict):
            state = state["params"]  # official release wraps the state_dict
        model.load_state_dict(state, strict=True)
        model.eval()
        self.model = model
        self._device = device
        self.model.to(device)

    def unload(self) -> None:
        self.model = None
        self._device = None

    @torch.no_grad()
    def _run_tile(self, tile_f32: np.ndarray, device: str) -> np.ndarray:
        t = torch.from_numpy(tile_f32).permute(2, 0, 1).unsqueeze(0).to(device)
        out = self.model(t)
        return out.squeeze(0).permute(1, 2, 0).cpu().numpy()

    def deblur(self, img: np.ndarray, device: str = "cpu") -> tuple[np.ndarray, dict]:
        """Deblur with tiled inference. Returns (image, info)."""
        info: dict = {"applied": False}
        if self.model is None:
            info["reason"] = "weights not loaded"
            return img, info
        try:
            measurement = estimate_optical_blur(img)
            info.update(measurement)
            if not measurement["is_optical"]:
                info["reason"] = "gate: not optical blur"
                return img, info

            h, w = img.shape[:2]
            img_f = img.astype(np.float32)
            if max(h, w) <= _TILE:
                out = self._run_tile(img_f, device)
            else:
                out = np.zeros_like(img_f)
                weight = np.zeros((h, w, 1), dtype=np.float32)
                step = _TILE - 2 * _TILE_OVERLAP
                ys = list(range(0, max(h - _TILE, 1) + 1, step))
                xs = list(range(0, max(w - _TILE, 1) + 1, step))
                if ys[-1] + _TILE < h:
                    ys.append(h - _TILE)
                if xs[-1] + _TILE < w:
                    xs.append(w - _TILE)
                for y in ys:
                    for x in xs:
                        y0, x0 = max(0, y), max(0, x)
                        y1, x1 = min(h, y + _TILE), min(w, x + _TILE)
                        pad_top, pad_left = y0 - y, x0 - x
                        tile = np.pad(
                            img_f[y0:y1, x0:x1],
                            ((pad_top, _TILE - (y1 - y0) - pad_top),
                             (pad_left, _TILE - (x1 - x0) - pad_left),
                             (0, 0)),
                            mode="reflect",
                        )
                        processed = self._run_tile(tile, device)
                        core = processed[pad_top:pad_top + (y1 - y0),
                                         pad_left:pad_left + (x1 - x0)]
                        out[y0:y1, x0:x1] += core
                        weight[y0:y1, x0:x1] += 1.0
                out = out / np.maximum(weight, 1.0)

            result = np.clip(out, 0, 255).astype(np.uint8)
            info["applied"] = True
            return result, info
        except Exception as exc:  # defensive: deblur must never kill a restore
            info["error"] = str(exc)
            return img, info
