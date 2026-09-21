"""
defect_segmenter.py — Pretrained Deep Semantic Defect Segmentation (U-Net).

Architecture based on Microsoft Research "Bringing Old Photos Back to Life" (CVPR 2020).
Trained on tens of thousands of real and synthetic historical photo damages
(scratches, emulsion flakes, fold tears, dust specks).

Provides semantic defect probability masks without brittle mathematical thresholds.
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class Downsample(nn.Module):
    """Anti-aliased downsampling layer."""

    def __init__(self, pad_type: str = "reflect", filt_size: int = 3, stride: int = 2, channels: Optional[int] = None, pad_off: int = 0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
        ]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.0)
        self.channels = channels

        if filt_size == 1:
            a = np.array([1.0])
        elif filt_size == 2:
            a = np.array([1.0, 1.0])
        elif filt_size == 3:
            a = np.array([1.0, 2.0, 1.0])
        elif filt_size == 4:
            a = np.array([1.0, 3.0, 3.0, 1.0])
        elif filt_size == 5:
            a = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
        elif filt_size == 6:
            a = np.array([1.0, 5.0, 10.0, 10.0, 5.0, 1.0])
        elif filt_size == 7:
            a = np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])
        else:
            raise ValueError(f"Unsupported filter size: {filt_size}")

        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)
        self.register_buffer("filt", filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))
        self.pad = nn.ReflectionPad2d(self.pad_sizes)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class UNetConvBlock(nn.Module):
    def __init__(self, conv_num: int, in_size: int, out_size: int, padding: bool, batch_norm: bool):
        super(UNetConvBlock, self).__init__()
        block = []
        for _ in range(conv_num):
            block.append(nn.ReflectionPad2d(padding=int(padding)))
            block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=0))
            if batch_norm:
                block.append(nn.BatchNorm2d(out_size))
            block.append(nn.LeakyReLU(0.2, True))
            in_size = out_size
        self.block = nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetUpBlock(nn.Module):
    def __init__(self, conv_num: int, in_size: int, out_size: int, up_mode: str, padding: bool, batch_norm: bool):
        super(UNetUpBlock, self).__init__()
        if up_mode == "upconv":
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == "upsample":
            self.up = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2, align_corners=False),
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_size, out_size, kernel_size=3, padding=0),
            )
        else:
            raise ValueError(f"Unknown up_mode: {up_mode}")

        self.conv_block = UNetConvBlock(conv_num, in_size, out_size, padding, batch_norm)

    def center_crop(self, layer: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])]

    def forward(self, x: torch.Tensor, bridge: torch.Tensor) -> torch.Tensor:
        up = self.up(x)
        crop1 = self.center_crop(bridge, (up.shape[2], up.shape[3]))
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)
        return out


class ScratchUNet(nn.Module):
    """5-layer / 4-depth U-Net for Scratch and Defect Segmentation."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        depth: int = 4,
        conv_num: int = 2,
        wf: int = 6,
        padding: bool = True,
        batch_norm: bool = True,
        up_mode: str = "upsample",
        antialiasing: bool = True,
    ):
        super(ScratchUNet, self).__init__()
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels

        self.first = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 2 ** wf, kernel_size=7),
            nn.LeakyReLU(0.2, True),
        )
        prev_channels = 2 ** wf

        self.down_path = nn.ModuleList()
        self.down_sample = nn.ModuleList()
        for i in range(depth):
            if antialiasing and depth > 0:
                self.down_sample.append(
                    nn.Sequential(
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(prev_channels, prev_channels, kernel_size=3, stride=1, padding=0),
                        nn.BatchNorm2d(prev_channels),
                        nn.LeakyReLU(0.2, True),
                        Downsample(channels=prev_channels, stride=2),
                    )
                )
            else:
                self.down_sample.append(
                    nn.Sequential(
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(prev_channels, prev_channels, kernel_size=4, stride=2, padding=0),
                        nn.BatchNorm2d(prev_channels),
                        nn.LeakyReLU(0.2, True),
                    )
                )
            self.down_path.append(
                UNetConvBlock(conv_num, prev_channels, 2 ** (wf + i + 1), padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i + 1)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth)):
            self.up_path.append(
                UNetUpBlock(conv_num, 2 ** (wf + i + 1), 2 ** (wf + i), up_mode, padding, batch_norm)
            )

        self.last = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(2 ** wf, out_channels, kernel_size=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first(x)
        blocks = []
        for i, down_block in enumerate(self.down_path):
            blocks.append(x)
            x = self.down_sample[i](x)
            x = down_block(x)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)


class DefectSegmenter:
    """High-level semantic defect segmentation engine for PixRevive."""

    def __init__(self, weights_path: Optional[str] = None, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if weights_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.weights_path = os.path.join(base_dir, "weights", "defect_detect", "FT_Epoch_latest.pt")
        else:
            self.weights_path = weights_path

        self.model: Optional[ScratchUNet] = None
        self.is_loaded = False
        self.load()

    def load(self) -> bool:
        if not os.path.exists(self.weights_path):
            logger.warning(f"DefectSegmenter: weights not found at {self.weights_path}")
            return False

        try:
            net = ScratchUNet(in_channels=1, out_channels=1, depth=4, conv_num=2, wf=6, antialiasing=True)
            checkpoint = torch.load(self.weights_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint.get("model_state", checkpoint)

            # Strip module. prefix if multi-gpu saved
            clean_state = {}
            for k, v in state_dict.items():
                clean_k = k.replace("module.", "")
                clean_state[clean_k] = v

            net.load_state_dict(clean_state, strict=True)
            net.to(self.device)
            net.eval()
            self.model = net
            self.is_loaded = True
            logger.info(f"[OK] DefectSegmenter (U-Net) loaded on {self.device}.")
            return True
        except Exception as exc:
            logger.error(f"Failed to load DefectSegmenter: {exc}")
            self.model = None
            self.is_loaded = False
            return False

    @torch.no_grad()
    def segment(self, img_rgb: np.ndarray, prob_threshold: float = 0.45, min_area: int = 15) -> np.ndarray:
        """
        Segment real scratches, flakes, and cracks from an RGB/grayscale image.

        Args:
            img_rgb: Input uint8 RGB image (H x W x 3) or grayscale (H x W)
            prob_threshold: Sigmoid probability threshold (default 0.45)
            min_area: Minimum connected component pixel area to retain as defect

        Returns:
            Binary mask (H x W) uint8 where 255 = damage, 0 = clean
        """
        if not self.is_loaded or self.model is None:
            logger.warning("DefectSegmenter not loaded, returning empty mask.")
            return np.zeros(img_rgb.shape[:2], dtype=np.uint8)

        h_orig, w_orig = img_rgb.shape[:2]

        # Convert to single-channel grayscale
        if img_rgb.ndim == 3:
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_rgb.copy()

        # Scale down for UNet if larger than 512 (BOPBTL trained at 256-512px; saves 90% VRAM)
        max_dim = 512
        scale = 1.0
        if max(h_orig, w_orig) > max_dim:
            scale = max_dim / max(h_orig, w_orig)
            h_scaled = max(16, round(h_orig * scale))
            w_scaled = max(16, round(w_orig * scale))
        else:
            h_scaled = h_orig
            w_scaled = w_orig

        # Scale to multiple of 16 for UNet feature pyramid alignment
        h_pad = ((h_scaled + 15) // 16) * 16
        w_pad = ((w_scaled + 15) // 16) * 16

        gray_input = cv2.resize(gray, (w_pad, h_pad), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)

        # Normalize to [-1, 1] range expected by BOPBTL ScratchUNet
        tensor_in = (gray_input.astype(np.float32) / 127.5) - 1.0
        tensor_in = torch.from_numpy(tensor_in).unsqueeze(0).unsqueeze(0).to(self.device)

        # Forward pass with no_grad
        try:
            with torch.inference_mode():
                logits = self.model(tensor_in)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Resize probability map back to original size
        probs = cv2.resize(probs, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

        # Apply probability threshold
        raw_mask = (probs >= prob_threshold).astype(np.uint8) * 255

        # Area filtering for noise suppression
        if min_area > 0 and np.any(raw_mask > 0):
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)
            clean_mask = np.zeros_like(raw_mask)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_area:
                    clean_mask[labels == i] = 255
            return clean_mask

        return raw_mask
