"""
line_structure.py — Shared Structure-Tensor and Line Discriminator.

Differentiates natural coherent linear textures (hair strands, fur, fabric weave,
whiskers) from structural damage (creases, scratches, tears) and halftone grain.

Physics and Mathematical Basis:
1. Local Structure Tensor:
     J = [ [G * (Ix^2), G * (Ix*Iy)],
           [G * (Ix*Iy), G * (Iy^2)] ]
   Eigenvalues lambda1 >= lambda2 >= 0.
   Coherence C = ((lambda1 - lambda2) / (lambda1 + lambda2 + eps))^2.
   Flow orientation theta = 0.5 * atan2(2*Jxy, Jxx - Jyy) + pi/2.

2. Discriminator Rule:
   - Hair / fur / fabric: high coherence (C >= 0.30), dense parallel lines,
     smoothly varying flow direction, component aligned with flow.
   - Crease / scratch / crack: isolated lines or cutting ACROSS local flow,
     low local line density, breaking local structure.
   - Grain / halftone: isotropic (C ~ 0), low elongation (aspect < 2.5),
     small compact components.
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


class LineStructureAnalyzer:
    """Computes structure-tensor coherence, local orientation flow, and component shapes."""

    def __init__(self, ksize: int = 9, sigma: float = 2.0):
        self.ksize = ksize | 1
        self.sigma = sigma

    def compute_tensor_field(self, gray_u8: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute structure tensor eigenvalues, coherence, and flow angle theta.
        
        Returns:
            dict containing:
              - 'coherence': float32 map in [0.0, 1.0]
              - 'theta': float32 flow angle in [-pi/2, pi/2] (along edge direction)
              - 'grad_mag': float32 gradient magnitude
              - 'l1', 'l2': float32 eigenvalues
        """
        if gray_u8.ndim == 3:
            gray_u8 = cv2.cvtColor(gray_u8, cv2.COLOR_RGB2GRAY)
        gray_f = gray_u8.astype(np.float32)

        gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx * gx + gy * gy)

        jxx = cv2.GaussianBlur(gx * gx, (self.ksize, self.ksize), self.sigma)
        jyy = cv2.GaussianBlur(gy * gy, (self.ksize, self.ksize), self.sigma)
        jxy = cv2.GaussianBlur(gx * gy, (self.ksize, self.ksize), self.sigma)

        diff = jxx - jyy
        root = np.sqrt(diff * diff + 4.0 * jxy * jxy)
        l1 = 0.5 * (jxx + jyy + root)
        l2 = 0.5 * (jxx + jyy - root)

        denom = l1 + l2 + 1e-5
        coherence = np.clip(((l1 - l2) / denom) ** 2, 0.0, 1.0)
        # Flow direction is perpendicular to gradient direction
        theta = 0.5 * np.arctan2(2.0 * jxy, diff) + (np.pi / 2.0)
        # Wrap theta into [-pi/2, pi/2]
        theta = (theta + np.pi / 2.0) % np.pi - np.pi / 2.0

        return {
            "coherence": coherence,
            "theta": theta,
            "grad_mag": grad_mag,
            "l1": l1,
            "l2": l2,
        }

    def analyze_components(
        self,
        binary_mask: np.ndarray,
        tensor_field: Optional[Dict[str, np.ndarray]] = None,
        gray_u8: Optional[np.ndarray] = None,
    ) -> List[Dict[str, float]]:
        """
        Analyze connected components of a candidate binary damage or speckle mask.

        Returns list of dicts with:
          - 'label': int CC label
          - 'area': int pixel area
          - 'aspect_ratio': elongation ratio (major / minor axis)
          - 'coherence': mean structure coherence in neighborhood
          - 'local_density': number of neighboring elongated components
          - 'cuts_across_flow': float score in [0.0, 1.0]
          - 'is_elongated': bool (aspect_ratio >= 2.5)
          - 'is_hair_like': bool (elongated, high coherence, flow-aligned, dense)
          - 'is_crease_like': bool (elongated, but sparse, low coherence, or cuts flow)
        """
        if binary_mask.dtype != np.uint8:
            binary_mask = (binary_mask > 0).astype(np.uint8)

        if tensor_field is None and gray_u8 is not None:
            tensor_field = self.compute_tensor_field(gray_u8)

        coherence = tensor_field["coherence"] if tensor_field is not None else None
        flow_theta = tensor_field["theta"] if tensor_field is not None else None

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )

        h, w = binary_mask.shape[:2]
        comps_raw = []
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            cx, cy = centroids[i]
            comp_mask = labels == i
            # Central moments mu20, mu02, mu11 for rotation-invariant elongation and angle
            m = cv2.moments(comp_mask.astype(np.uint8))
            mu20 = m["mu20"]
            mu02 = m["mu02"]
            mu11 = m["mu11"]
            diff = mu20 - mu02
            root = np.sqrt(diff * diff + 4.0 * mu11 * mu11)
            l1 = 0.5 * (mu20 + mu02 + root)
            l2 = 0.5 * (mu20 + mu02 - root)
            moment_elongation = float(np.sqrt(l1 / max(l2, 0.05)))
            bbox_aspect = float(max(bw, bh) / max(1, min(bw, bh)))
            aspect = max(moment_elongation, bbox_aspect)

            comp_theta = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)
            comp_theta = (comp_theta + np.pi / 2.0) % np.pi - np.pi / 2.0

            # Local background flow theta
            if flow_theta is not None:
                # Average flow angle around component
                local_flow = float(np.mean(flow_theta[comp_mask]))
                d_theta = abs(comp_theta - local_flow)
                d_theta = min(d_theta, np.pi - d_theta)
                cuts_score = float(d_theta / (np.pi / 2.0))
            else:
                cuts_score = 0.0

            mean_coh = float(np.mean(coherence[comp_mask])) if coherence is not None else 0.0
            is_elongated = aspect >= 2.5 and area >= 6
            comps_raw.append({
                "label": i,
                "area": area,
                "cx": cx,
                "cy": cy,
                "aspect_ratio": round(aspect, 2),
                "coherence": round(mean_coh, 3),
                "cuts_across_flow": round(cuts_score, 3),
                "is_elongated": is_elongated,
            })

        # Compute local line density (count of elongated components within radius 35)
        elongated_centers = [
            (c["cx"], c["cy"]) for c in comps_raw if c["is_elongated"]
        ]
        results = []
        for c in comps_raw:
            if not c["is_elongated"]:
                density = 0
            else:
                cx, cy = c["cx"], c["cy"]
                density = sum(
                    1 for ex, ey in elongated_centers
                    if (ex - cx) ** 2 + (ey - cy) ** 2 <= 35.0 ** 2
                )
            c["local_density"] = density

            # Hair/fur strands: elongated, high coherence, flow-aligned (cuts < 0.4), high density
            is_hair = (
                c["is_elongated"]
                and c["coherence"] >= 0.30
                and c["cuts_across_flow"] <= 0.40
                and density >= 2
            )
            # Creases / scratches / tears: elongated, but cutting across flow, isolated, or breaking field
            is_crease = (
                c["is_elongated"]
                and (c["cuts_across_flow"] > 0.40 or density < 2 or c["coherence"] < 0.25)
            )

            c["is_hair_like"] = is_hair
            c["is_crease_like"] = is_crease
            results.append(c)

        return results

    def filter_hair_from_damage_mask(
        self,
        damage_mask: np.ndarray,
        gray_u8: np.ndarray,
    ) -> np.ndarray:
        """
        Prunes false-positive damage candidates that are actually hair strands or fur.
        Keeps genuine creases, cracks, and flakes.
        """
        if damage_mask is None or np.sum(damage_mask > 0) == 0:
            return damage_mask

        tensor_field = self.compute_tensor_field(gray_u8)
        comps = self.analyze_components(damage_mask, tensor_field=tensor_field)

        num_labels, labels, _, _ = cv2.connectedComponentsWithStats(damage_mask, connectivity=8)
        filtered = damage_mask.copy()

        for c in comps:
            if c["is_hair_like"]:
                filtered[labels == c["label"]] = 0

        return filtered

    def filter_hair_from_inpaint_mask(
        self,
        inpaint_mask: np.ndarray,
        gray_u8: np.ndarray,
    ) -> np.ndarray:
        """
        Protects texture and hair zones in the inpainting mask unless passing a
        damage-specific test (stroke width wider than a strand, or strong brightness popout).
        """
        if inpaint_mask is None or np.sum(inpaint_mask > 0) == 0:
            return inpaint_mask

        tensor_field = self.compute_tensor_field(gray_u8)
        comps = self.analyze_components(inpaint_mask, tensor_field=tensor_field, gray_u8=gray_u8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inpaint_mask, connectivity=8)
        filtered = inpaint_mask.copy()

        local_median = cv2.medianBlur(gray_u8, 15).astype(np.float32)

        for c in comps:
            if c["is_hair_like"]:
                lbl = c["label"]
                comp_m = labels == lbl
                area = c["area"]
                bw = int(stats[lbl, cv2.CC_STAT_WIDTH])
                bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
                # Estimated stroke width: area / max(span, 1)
                stroke_width = area / max(1.0, float(max(bw, bh)))

                # Brightness popout vs local median
                comp_gray = gray_u8[comp_m].astype(np.float32)
                comp_med = local_median[comp_m]
                popout = float(np.mean(np.abs(comp_gray - comp_med)))

                # Damage specific tests:
                # 1. Wider than a strand (stroke width >= 4.0 px)
                # 2. Stark emulsion loss or tear shadow (|popout| >= 28.0)
                if stroke_width >= 4.0 or popout >= 28.0:
                    continue  # Keep damage
                
                # Prune natural hair strand
                filtered[comp_m] = 0

        return filtered

    @staticmethod
    def classify_speckle_vs_crease(hostile_u8: np.ndarray) -> bool:
        """
        Classifies whether a population of hostile residual pixels is dominated by
        crease/crack structure (elongated lines) rather than isotropic halftone/grain specks.
        Returns True if significant hostile pixels belong to crease structure.
        """
        if hostile_u8 is None or np.sum(hostile_u8 > 0) == 0:
            return False

        n_h, _, stats_h, _ = cv2.connectedComponentsWithStats(hostile_u8, connectivity=8)
        crease_px = 0
        total_px = float(np.sum(hostile_u8 > 0))

        for idx in range(1, n_h):
            a = int(stats_h[idx, cv2.CC_STAT_AREA])
            if a >= 6:
                w = int(stats_h[idx, cv2.CC_STAT_WIDTH])
                h = int(stats_h[idx, cv2.CC_STAT_HEIGHT])
                span_sq = w * w + h * h
                thinness = span_sq / max(1, a)
                if thinness >= 3.5 or max(w, h) >= 15:
                    crease_px += a

        return (crease_px / max(total_px, 1.0)) > 0.40
