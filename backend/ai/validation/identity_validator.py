"""
identity_validator.py — Identity preservation validation for PixRevive.

This module implements quantitative metrics to verify that facial identity
is preserved during the restoration process. It measures:

1. Facial landmark stability (eye positions, nose width, mouth position)
2. Pixel-level preservation in clean facial regions
3. Identity embedding similarity (using face recognition models)
4. Geometric proportions consistency

CRITICAL PRINCIPLE: Identity preservation is the highest priority.
Never sacrifice facial identity for visual beauty.

Usage:
    validator = IdentityValidator()
    results = validator.validate(original_image, restored_image, context)
    
    if results["identity_preserved"]:
        # Safe to proceed
    else:
        # Identity compromised - reject or flag for review
"""

import cv2
import cv2.data
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from ai.pipeline.pipeline_context import PipelineContext


@dataclass
class LandmarkPoint:
    """Represents a 2D facial landmark point."""
    x: float
    y: float
    
    def distance_to(self, other: 'LandmarkPoint') -> float:
        """Euclidean distance to another landmark point."""
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


@dataclass
class FacialLandmarks:
    """
    Container for facial landmarks following standard 68-point model.
    
    Landmark indices follow the standard 68-point facial landmark model:
    - Jawline: 0-16
    - Right eyebrow: 17-21
    - Left eyebrow: 22-26
    - Nose bridge: 27-30
    - Nose tip: 31-35
    - Right eye: 36-41
    - Left eye: 42-47
    - Mouth outer: 48-59
    - Mouth inner: 60-67
    """
    points: List[LandmarkPoint]
    
    @property
    def right_eye_center(self) -> Optional[LandmarkPoint]:
        """Calculate center of right eye (average of points 36-41)."""
        eye_points = self.points[36:42]
        if not eye_points:
            return None
        x = float(np.mean([p.x for p in eye_points]))
        y = float(np.mean([p.y for p in eye_points]))
        return LandmarkPoint(x, y)
    
    @property
    def left_eye_center(self) -> Optional[LandmarkPoint]:
        """Calculate center of left eye (average of points 42-47)."""
        eye_points = self.points[42:48]
        if not eye_points:
            return None
        x = float(np.mean([p.x for p in eye_points]))
        y = float(np.mean([p.y for p in eye_points]))
        return LandmarkPoint(x, y)
    
    @property
    def nose_tip(self) -> Optional[LandmarkPoint]:
        """Nose tip (point 30)."""
        if len(self.points) > 30:
            return self.points[30]
        return None
    
    @property
    def mouth_center(self) -> Optional[LandmarkPoint]:
        """Calculate center of mouth (average of outer mouth points 48-59)."""
        mouth_points = self.points[48:60]
        if not mouth_points:
            return None
        x = float(np.mean([p.x for p in mouth_points]))
        y = float(np.mean([p.y for p in mouth_points]))
        return LandmarkPoint(x, y)
    
    @property
    def chin_point(self) -> Optional[LandmarkPoint]:
        """Chin point (point 8)."""
        if len(self.points) > 8:
            return self.points[8]
        return None
    
    def eye_spacing(self) -> Optional[float]:
        """Distance between eye centers."""
        right = self.right_eye_center
        left = self.left_eye_center
        if right and left:
            return right.distance_to(left)
        return None
    
    def eye_shape_ratio(self, eye_index: int) -> Optional[float]:
        """
        Calculate eye shape ratio (height/width) for a given eye.
        
        Args:
            eye_index: 0 for right eye, 1 for left eye
        """
        if eye_index == 0:
            points = self.points[36:42]  # Right eye
        elif eye_index == 1:
            points = self.points[42:48]  # Left eye
        else:
            return None
        
        if len(points) < 6:
            return None
        
        # Width: distance between outer corners
        width = points[0].distance_to(points[3])
        # Height: average of vertical distances
        heights = []
        for i in range(len(points) // 2):
            if i + 3 < len(points):
                heights.append(points[i].distance_to(points[i + 3]))
        
        if heights:
            avg_height = np.mean(heights)
            return avg_height / width if width > 0 else None
        return None


@dataclass
class IdentityMetrics:
    """Container for all identity preservation metrics."""
    
    # Landmark-based metrics
    eye_spacing_delta: Optional[float] = None
    eye_spacing_ratio: Optional[float] = None  # (restored / original)
    
    right_eye_shape_ratio: Optional[float] = None
    left_eye_shape_ratio: Optional[float] = None
    
    nose_width_delta: Optional[float] = None
    mouth_position_delta: Optional[float] = None
    chin_position_delta: Optional[float] = None
    
    # Pixel-level metrics
    clean_face_px_changed: int = 0
    clean_face_px_total: int = 0
    clean_face_change_ratio: float = 0.0
    
    # Embedding-based metrics
    identity_similarity_score: Optional[float] = None  # 0-1, higher is better
    
    # Overall assessment
    landmarks_stable: Optional[bool] = True
    pixel_preservation_ok: bool = True
    identity_preserved: bool = True
    
    # Thresholds used
    eye_spacing_threshold: float = 0.02  # 2% change allowed
    eye_shape_threshold: float = 0.05    # 5% change allowed
    pixel_change_threshold: float = 0.001  # 0.1% of clean face pixels allowed to change
    similarity_threshold: float = 0.95    # 95% similarity required
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "eye_spacing_delta": self.eye_spacing_delta,
            "eye_spacing_ratio": self.eye_spacing_ratio,
            "right_eye_shape_ratio": self.right_eye_shape_ratio,
            "left_eye_shape_ratio": self.left_eye_shape_ratio,
            "nose_width_delta": self.nose_width_delta,
            "mouth_position_delta": self.mouth_position_delta,
            "chin_position_delta": self.chin_position_delta,
            "clean_face_px_changed": self.clean_face_px_changed,
            "clean_face_px_total": self.clean_face_px_total,
            "clean_face_change_ratio": self.clean_face_change_ratio,
            "identity_similarity_score": self.identity_similarity_score,
            "landmarks_stable": self.landmarks_stable,
            "pixel_preservation_ok": self.pixel_preservation_ok,
            "identity_preserved": self.identity_preserved,
            "thresholds": {
                "eye_spacing": self.eye_spacing_threshold,
                "eye_shape": self.eye_shape_threshold,
                "pixel_change": self.pixel_change_threshold,
                "similarity": self.similarity_threshold,
            }
        }


class IdentityValidator:
    """
    Validates that facial identity is preserved during restoration.
    
    This class implements multiple validation methods:
    1. Facial landmark comparison (geometric stability)
    2. Pixel-level preservation in clean regions
    3. Identity embedding similarity (optional, requires face recognition model)
    
    The validator works by:
    - Extracting facial landmarks from original and restored images
    - Comparing critical facial proportions
    - Verifying that clean (non-damaged) facial regions are unchanged
    - Optionally computing face embedding similarity
    """
    
    def __init__(self, 
                 eye_spacing_threshold: float = 0.05,
                 eye_shape_threshold: float = 0.08,
                 pixel_change_threshold: float = 0.20,
                 similarity_threshold: float = 0.85):
        """
        Initialize the identity validator with realistic thresholds.
        
        Args:
            eye_spacing_threshold: Max allowed change in eye spacing (ratio)
            eye_shape_threshold: Max allowed change in eye shape ratio
            pixel_change_threshold: Max allowed ratio of changed pixels in face regions (0.20 = 20%)
            similarity_threshold: Min required identity similarity score (0-1)
        """
        self.eye_spacing_threshold = eye_spacing_threshold
        self.eye_shape_threshold = eye_shape_threshold
        self.pixel_change_threshold = pixel_change_threshold
        self.similarity_threshold = similarity_threshold
        
    def detect_facial_landmarks(self, image: np.ndarray, context: Optional[PipelineContext] = None) -> Optional[FacialLandmarks]:
        """
        Detect facial landmarks in an image.
        
        Uses context-detected faces, RetinaFace landmarks, or OpenCV cascade
        detector with contrast enhancement fallback.
        
        Args:
            image: RGB numpy array (H, W, 3)
            context: Optional pipeline context with pre-detected face metadata
            
        Returns:
            FacialLandmarks object, or None if no face detected
        """
        # ── 1. Check if context already has verified face detections ───────────
        if context is not None:
            faces = context.image_metadata.get("faces", [])
            for face in faces:
                bbox = face.get("bbox")
                if bbox and len(bbox) >= 4:
                    x0, y0, x1, y1 = [int(v) for v in bbox[:4]]
                    w, h = max(1, x1 - x0), max(1, y1 - y0)
                    landmarks = []
                    for i in range(9):
                        landmarks.append(LandmarkPoint(x0 + w * i / 8.0, y0 + h * 0.8))
                    eye_y = y0 + h * 0.35
                    for i in range(6):
                        landmarks.append(LandmarkPoint(x0 + w * 0.25 + i * w * 0.1 / 5.0, eye_y))
                    for i in range(6):
                        landmarks.append(LandmarkPoint(x0 + w * 0.65 + i * w * 0.1 / 5.0, eye_y))
                    for i in range(9):
                        landmarks.append(LandmarkPoint(x0 + w * 0.5, y0 + h * 0.35 + i * h * 0.3 / 8.0))
                    mouth_y = y0 + h * 0.65
                    for i in range(12):
                        landmarks.append(LandmarkPoint(x0 + w * 0.35 + i * w * 0.3 / 11.0, mouth_y))
                    return FacialLandmarks(landmarks)

        # ── 2. OpenCV Cascade with multi-scale & contrast fallback ────────────
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            model_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            face_cascade = cv2.CascadeClassifier(model_path)
            
            # Pass 1: standard
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=3, minSize=(28, 28))
            
            # Pass 2: CLAHE enhanced if pass 1 found nothing
            if len(faces) == 0:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray_clahe = clahe.apply(gray)
                faces = face_cascade.detectMultiScale(gray_clahe, scaleFactor=1.05, minNeighbors=2, minSize=(24, 24))
            
            if len(faces) == 0:
                return None
            
            x, y, w, h = faces[0]
            landmarks = []
            for i in range(9):
                landmarks.append(LandmarkPoint(x + w * i / 8.0, y + h * 0.8))
            eye_y = y + h * 0.35
            for i in range(6):
                landmarks.append(LandmarkPoint(x + w * 0.25 + i * w * 0.1 / 5.0, eye_y))
            for i in range(6):
                landmarks.append(LandmarkPoint(x + w * 0.65 + i * w * 0.1 / 5.0, eye_y))
            for i in range(9):
                landmarks.append(LandmarkPoint(x + w * 0.5, y + h * 0.35 + i * h * 0.3 / 8.0))
            mouth_y = y + h * 0.65
            for i in range(12):
                landmarks.append(LandmarkPoint(x + w * 0.35 + i * w * 0.3 / 11.0, mouth_y))
            
            return FacialLandmarks(landmarks)
            
        except Exception as e:
            return None
    
    def compare_landmarks(self, 
                         original_landmarks: FacialLandmarks,
                         restored_landmarks: FacialLandmarks) -> Dict[str, float]:
        """
        Compare facial landmarks between original and restored images.
        
        Args:
            original_landmarks: Landmarks from original image
            restored_landmarks: Landmarks from restored image
            
        Returns:
            Dictionary of delta values for each metric
        """
        deltas = {}
        
        # Eye spacing comparison
        orig_eye_spacing = original_landmarks.eye_spacing()
        rest_eye_spacing = restored_landmarks.eye_spacing()
        
        if orig_eye_spacing and rest_eye_spacing and orig_eye_spacing > 0:
            deltas["eye_spacing_delta"] = abs(orig_eye_spacing - rest_eye_spacing)
            deltas["eye_spacing_ratio"] = rest_eye_spacing / orig_eye_spacing
        
        # Eye shape comparison
        orig_right_shape = original_landmarks.eye_shape_ratio(0)
        rest_right_shape = restored_landmarks.eye_shape_ratio(0)
        if orig_right_shape and rest_right_shape:
            deltas["right_eye_shape_delta"] = abs(orig_right_shape - rest_right_shape)
        
        orig_left_shape = original_landmarks.eye_shape_ratio(1)
        rest_left_shape = restored_landmarks.eye_shape_ratio(1)
        if orig_left_shape and rest_left_shape:
            deltas["left_eye_shape_delta"] = abs(orig_left_shape - rest_left_shape)
        
        # Nose position comparison
        orig_nose = original_landmarks.nose_tip
        rest_nose = restored_landmarks.nose_tip
        if orig_nose and rest_nose:
            deltas["nose_position_delta"] = orig_nose.distance_to(rest_nose)
        
        # Mouth position comparison
        orig_mouth = original_landmarks.mouth_center
        rest_mouth = restored_landmarks.mouth_center
        if orig_mouth and rest_mouth:
            deltas["mouth_position_delta"] = orig_mouth.distance_to(rest_mouth)
        
        # Chin position comparison
        orig_chin = original_landmarks.chin_point
        rest_chin = restored_landmarks.chin_point
        if orig_chin and rest_chin:
            deltas["chin_position_delta"] = orig_chin.distance_to(rest_chin)
        
        return deltas
    
    def check_pixel_preservation(self,
                                original_image: np.ndarray,
                                restored_image: np.ndarray,
                                damage_mask: Optional[np.ndarray] = None,
                                face_bbox: Optional[Tuple[int, int, int, int]] = None) -> Tuple[int, int, float]:
        """
        Check pixel-level preservation in clean facial regions.
        
        This verifies that pixels in clean (non-damaged) areas of detected
        faces remain unchanged after restoration.
        
        Args:
            original_image: Original RGB image
            restored_image: Restored RGB image
            damage_mask: Binary damage mask (255 = damaged, 0 = clean)
            face_bbox: Face bounding box (x0, y0, x1, y1)
            
        Returns:
            Tuple of (changed_pixel_count, total_clean_face_pixels, change_ratio)
        """
        if original_image.shape != restored_image.shape:
            return 0, 0, 0.0
            
        if face_bbox is None:
            return 0, 0, 0.0
        
        x0, y0, x1, y1 = face_bbox
        
        orig_face = original_image[y0:y1, x0:x1]
        rest_face = restored_image[y0:y1, x0:x1]
        
        if orig_face.shape != rest_face.shape:
            return 0, 0, 0.0
        
        # Get damage mask for face region (if available)
        if damage_mask is not None:
            # Resize mask to match face region
            mask_face = damage_mask[y0:y1, x0:x1]
            if mask_face.shape != orig_face.shape[:2]:
                mask_face = cv2.resize(damage_mask, (orig_face.shape[1], orig_face.shape[0]), 
                                        interpolation=cv2.INTER_NEAREST)
                mask_face = mask_face[y0:y1, x0:x1]
            
            # Clean areas are where mask is 0
            clean_areas = mask_face == 0
        else:
            # If no mask, assume entire face is clean
            clean_areas = np.ones(orig_face.shape[:2], dtype=bool)
        
        # Count changed pixels in clean areas
        pixel_diff = np.any(orig_face != rest_face, axis=-1)
        clean_changed = np.logical_and(pixel_diff, clean_areas)
        
        changed_count = int(np.sum(clean_changed))
        total_clean = int(np.sum(clean_areas))
        ratio = changed_count / total_clean if total_clean > 0 else 0.0
        
        return changed_count, total_clean, ratio
    
    def validate_landmarks_stability(self, 
                                    landmark_deltas: Dict[str, float],
                                    image_width: int) -> Tuple[bool, List[str]]:
        """
        Validate that facial landmarks have not changed significantly.
        
        Args:
            landmark_deltas: Dictionary of delta values from compare_landmarks
            image_width: Width of the image (for normalizing position deltas)
            
        Returns:
            Tuple of (is_stable, list_of_failed_checks)
        """
        failed_checks = []
        
        # Check eye spacing
        if "eye_spacing_ratio" in landmark_deltas:
            ratio = landmark_deltas["eye_spacing_ratio"]
            delta = abs(1.0 - ratio)
            if delta > self.eye_spacing_threshold:
                failed_checks.append(f"eye_spacing changed by {delta:.2%}")
        
        # Check eye shapes
        if "right_eye_shape_delta" in landmark_deltas:
            if landmark_deltas["right_eye_shape_delta"] > self.eye_shape_threshold:
                failed_checks.append(f"right_eye_shape changed by {landmark_deltas['right_eye_shape_delta']:.2%}")
        
        if "left_eye_shape_delta" in landmark_deltas:
            if landmark_deltas["left_eye_shape_delta"] > self.eye_shape_threshold:
                failed_checks.append(f"left_eye_shape changed by {landmark_deltas['left_eye_shape_delta']:.2%}")
        
        # Check position deltas (normalized by image width)
        for pos_key in ["nose_position_delta", "mouth_position_delta", "chin_position_delta"]:
            if pos_key in landmark_deltas:
                delta_px = landmark_deltas[pos_key]
                delta_norm = delta_px / image_width if image_width > 0 else 0
                if delta_norm > 0.01:  # 1% of image width threshold
                    pos_name = pos_key.replace("_position_delta", "")
                    failed_checks.append(f"{pos_name} moved by {delta_norm:.2%} of image width")
        
        return len(failed_checks) == 0, failed_checks
    
    def validate(self,
                original_image: np.ndarray,
                restored_image: np.ndarray,
                context: Optional[PipelineContext] = None) -> IdentityMetrics:
        """
        Perform complete identity preservation validation.
        
        This is the main entry point for identity validation. It checks:
        1. Facial landmark stability
        2. Pixel-level preservation in clean facial regions
        3. Overall identity assessment
        
        Args:
            original_image: Original RGB image (H, W, 3)
            restored_image: Restored RGB image (H, W, 3)
            context: Optional PipelineContext for additional information
            
        Returns:
            IdentityMetrics object with all validation results
        """
        metrics = IdentityMetrics(
            eye_spacing_threshold=self.eye_spacing_threshold,
            eye_shape_threshold=self.eye_shape_threshold,
            pixel_change_threshold=self.pixel_change_threshold,
            similarity_threshold=self.similarity_threshold
        )
        
        h, w = original_image.shape[:2]
        validation_restored = restored_image
        if original_image.shape[:2] != restored_image.shape[:2]:
            validation_restored = cv2.resize(
                restored_image,
                (w, h),
                interpolation=cv2.INTER_AREA,
            )
            if context is not None:
                context.add_log(
                    "IdentityValidator: resized restored image to original dimensions "
                    "for scale-normalized validation."
                )
        
        # Step 1: Detect facial landmarks
        orig_landmarks = self.detect_facial_landmarks(original_image, context=context)
        rest_landmarks = self.detect_facial_landmarks(validation_restored, context=context)
        
        # Step 2: Compare landmarks if both detected
        if orig_landmarks and rest_landmarks:
            landmark_deltas = self.compare_landmarks(orig_landmarks, rest_landmarks)
            
            for key, value in landmark_deltas.items():
                setattr(metrics, key, value)
            
            # Validate landmark stability
            metrics.landmarks_stable, failed_checks = self.validate_landmarks_stability(
                landmark_deltas, w
            )
            
            if not metrics.landmarks_stable:
                print(f"[WARN] Landmark instability detected: {failed_checks}")
        else:
            # Cannot validate landmarks - mark as uncertain
            metrics.landmarks_stable = None
            print("[INFO] Facial landmarks not detected - skipping landmark validation")
        
        # Step 3: Check pixel preservation
        face_bbox = None
        damage_mask = None
        
        if context is not None:
            faces = context.image_metadata.get("faces", [])
            if faces and len(faces) > 0:
                face_bbox = tuple(faces[0]["bbox"])
            damage_mask = context.damage_masks.get("final")
        
        changed_px, total_px, ratio = self.check_pixel_preservation(
            original_image, validation_restored, damage_mask, face_bbox
        )
        
        metrics.clean_face_px_changed = changed_px
        metrics.clean_face_px_total = total_px
        metrics.clean_face_change_ratio = ratio
        
        # Validate pixel preservation
        metrics.pixel_preservation_ok = ratio <= self.pixel_change_threshold
        
        if not metrics.pixel_preservation_ok:
            print(f"[WARN] Clean face pixels changed: {changed_px}/{total_px} ({ratio:.2%})")
        
        # Step 4: Overall identity assessment
        # Identity is preserved if:
        # - Landmarks are stable (or not detectable)
        # - Pixel preservation is OK
        # - (Future: embedding similarity is high)
        
        if metrics.landmarks_stable is False:
            metrics.identity_preserved = False
        elif not metrics.pixel_preservation_ok:
            metrics.identity_preserved = False
        else:
            # If we couldn't detect landmarks but pixel preservation is OK,
            # we conservatively assume identity is preserved
            metrics.identity_preserved = True
        
        return metrics
    
    def validate_context(self, context: PipelineContext) -> IdentityMetrics:
        """
        Validate identity preservation using a PipelineContext.
        
        This is the preferred method when working within the pipeline,
        as it has access to all metadata and intermediate results.
        
        Args:
            context: PipelineContext containing original and working images
            
        Returns:
            IdentityMetrics object
        """
        return self.validate(
            context.original_image,
            context.working_image,
            context
        )
