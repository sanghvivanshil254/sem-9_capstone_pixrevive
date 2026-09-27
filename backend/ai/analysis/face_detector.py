import os
import cv2
import numpy as np
import sys
import torch
import ai.compat  # noqa: F401 - compatibility shim for basicsr / facexlib

from facexlib.detection import init_detection_model
from ai.pipeline.pipeline_context import PipelineContext

class FaceDetector:
    def __init__(self, device='cpu'):
        self.device = device
        facelib_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "weights", "facelib"
        )
        self._facelib_dir = facelib_dir
        # Using retinaface for primary deep detection
        self.det_net = init_detection_model(
            'retinaface_resnet50', half=False, device=self.device, model_rootpath=facelib_dir
        )

    def unload(self) -> None:
        """Free the RetinaFace ResNet50 (~1.9 GB fp32 VRAM; Stage 12 bridge).

        Lazily reloaded on the next detect() call — the detector is used
        early in each request, before Stage 12 releases it again.
        """
        self.det_net = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _ensure_det_net(self) -> bool:
        if self.det_net is not None:
            return True
        try:
            self.det_net = init_detection_model(
                'retinaface_resnet50', half=False, device=self.device,
                model_rootpath=self._facelib_dir,
            )
            return self.det_net is not None
        except Exception as exc:  # noqa: BLE001 - degrade, never crash
            import logging
            logging.getLogger(__name__).warning(
                "FaceDetector reload failed: %s", exc)
            return False
        
    def detect(self, context: PipelineContext):
        if not self._ensure_det_net():
            context.add_log("Face detect: RetinaFace unavailable — skipped.")
            return
        img_np = context.working_image
        img_h, img_w = img_np.shape[:2]
        
        context.add_log("Detecting faces (multi-pass adaptive)...")
        bboxes = []
        
        def _extract_bboxes(res):
            if res is None:
                return None
            if isinstance(res, np.ndarray):
                return res if len(res) > 0 else None
            if isinstance(res, (list, tuple)):
                if len(res) >= 1 and res[0] is not None:
                    first = res[0]
                    if isinstance(first, np.ndarray):
                        return first if len(first) > 0 else None
                    return res
            return None

        # Ensure PyTorch grad is disabled to prevent Tensor.numpy() errors on PyTorch 2.x
        with torch.no_grad():
            # ── Pass 1: RetinaFace primary pass at robust 0.40 threshold ──────
            try:
                result = self.det_net.detect_faces(img_np, 0.40)
                bboxes = _extract_bboxes(result)
            except Exception as e:
                context.add_log(f"RetinaFace pass 1 failed: {e}")

            # ── Pass 2: Denoised preview pass for noisy/degraded photos ───────
            if bboxes is None or len(bboxes) == 0:
                try:
                    med = cv2.medianBlur(img_np, 3)
                    result = self.det_net.detect_faces(med, 0.35)
                    bboxes = _extract_bboxes(result)
                    if bboxes is not None and len(bboxes) > 0:
                        context.add_log("RetinaFace detected face via noise-filtered pass.")
                except Exception as e:
                    context.add_log(f"RetinaFace pass 2 failed: {e}")

            # ── Pass 3: Contrast-enhanced pass for faded/vintage portraits ────
            if bboxes is None or len(bboxes) == 0:
                try:
                    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
                    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
                    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                    enhanced_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                    
                    result = self.det_net.detect_faces(enhanced_rgb, 0.30)
                    bboxes = _extract_bboxes(result)
                    if bboxes is not None and len(bboxes) > 0:
                        context.add_log("RetinaFace detected face via contrast-enhanced pass.")
                except Exception as e:
                    context.add_log(f"RetinaFace pass 3 failed: {e}")

        # ── Pass 4: Haar Frontal & Profile Cascade fallback ───────────────────
        if bboxes is None or len(bboxes) == 0:
            try:
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                min_face_dim = max(40, min(img_w, img_h) // 12)
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                face_cascade = cv2.CascadeClassifier(cascade_path)
                haar_faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.08, minNeighbors=5, minSize=(min_face_dim, min_face_dim)
                )
                if len(haar_faces) == 0:
                    profile_path = cv2.data.haarcascades + "haarcascade_profileface.xml"
                    prof_cascade = cv2.CascadeClassifier(profile_path)
                    haar_faces = prof_cascade.detectMultiScale(
                        gray, scaleFactor=1.08, minNeighbors=5, minSize=(min_face_dim, min_face_dim)
                    )

                if len(haar_faces) > 0:
                    bboxes = []
                    for (hx, hy, hw, hh) in haar_faces:
                        bboxes.append([hx, hy, hx + hw, hy + hh, 0.70])
                    context.add_log(f"HaarCascade fallback detected {len(bboxes)} face(s).")
            except Exception as e:
                context.add_log(f"HaarCascade fallback failed: {e}")

        faces = []
        if bboxes is not None and len(bboxes) > 0:
            if isinstance(bboxes, np.ndarray) and bboxes.ndim == 1:
                bboxes = [bboxes]

            for bbox in bboxes:
                if len(bbox) < 4:
                    continue
                x0, y0, x1, y1 = bbox[:4]
                conf = float(bbox[4]) if len(bbox) > 4 else 0.75

                # Expand box slightly (6%) to ensure full anatomical coverage
                w_b = x1 - x0
                h_b = y1 - y0
                pad_x = int(w_b * 0.06)
                pad_y = int(h_b * 0.06)

                x0 = max(0, min(img_w, int(x0 - pad_x)))
                y0 = max(0, min(img_h, int(y0 - pad_y)))
                x1 = max(0, min(img_w, int(x1 + pad_x)))
                y1 = max(0, min(img_h, int(y1 + pad_y)))

                # Require minimum viable face size
                if (x1 - x0) < 32 or (y1 - y0) < 32:
                    continue

                # Optional 5-point landmarks from RetinaFace with geometric sanity validation
                landmarks = None
                if len(bbox) >= 14:
                    pts = np.array(bbox[5:15], dtype=np.float32).reshape((5, 2))
                    lx_min, ly_min = np.min(pts, axis=0)
                    lx_max, ly_max = np.max(pts, axis=0)
                    l_w, l_h = lx_max - lx_min, ly_max - ly_min
                    aspect = l_w / max(1.0, float(l_h))

                    # Geometric sanity check: aspect ratio and bounds validation
                    if (
                        0.5 <= aspect <= 1.8
                        and 0 <= lx_min < img_w
                        and 0 <= ly_min < img_h
                        and 0 < lx_max <= img_w
                        and 0 < ly_max <= img_h
                    ):
                        landmarks = pts.tolist()

                # Check intersection with damage mask
                damage_ratio = 0.0
                mask = context.damage_masks.get('final')
                if mask is not None and mask.shape[:2] == (img_h, img_w) and x1 > x0 and y1 > y0:
                    face_mask = mask[y0:y1, x0:x1]
                    damage_ratio = float(np.sum(face_mask > 0) / face_mask.size)

                faces.append({
                    "bbox": [x0, y0, x1, y1],
                    "confidence": conf,
                    "damage_ratio": damage_ratio,
                    "landmarks": landmarks,
                })

            # Sort faces by area descending (largest/primary face first)
            faces.sort(key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]), reverse=True)

        context.image_metadata['faces'] = faces
        context.image_metadata['faces_frame_size'] = (img_h, img_w)
        context.add_log(f"Detected {len(faces)} face(s).")
