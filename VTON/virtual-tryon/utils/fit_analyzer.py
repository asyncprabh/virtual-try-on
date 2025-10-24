import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

@dataclass
class FitMetrics:
    shoulder_alignment: float  # 0-1 score for shoulder alignment
    chest_fit: float  # 0-1 score for chest area fit
    waist_fit: float  # 0-1 score for waist area fit
    arm_holes: float  # 0-1 score for arm hole positioning
    overall_score: float  # Weighted average of all metrics
    problem_areas: List[str]  # List of areas needing adjustment

class FitAnalyzer:
    def __init__(self):
        # MediaPipe pose landmark indices
        self.SHOULDER_LEFT = 11
        self.SHOULDER_RIGHT = 12
        self.WAIST_LEFT = 23
        self.WAIST_RIGHT = 24
        self.ELBOW_LEFT = 13
        self.ELBOW_RIGHT = 14
        self.HIP_LEFT = 23
        self.HIP_RIGHT = 24
        self.NECK = 0
        
        # Fit threshold values
        self.GOOD_FIT_THRESHOLD = 0.85
        self.MODERATE_FIT_THRESHOLD = 0.7
        self.POOR_FIT_THRESHOLD = 0.5
        
        # Color maps for visualization
        self.colors = {
            'good': (0, 255, 0),     # Green
            'moderate': (0, 255, 255),  # Yellow
            'poor': (0, 0, 255)      # Red
        }
        
    def analyze_fit(self, frame: np.ndarray, clothing_mask: np.ndarray, 
                   landmarks: List, clothing_points: np.ndarray) -> FitMetrics:
        """Analyze the fit of the clothing based on body landmarks"""
        h, w = frame.shape[:2]
        
        # Convert landmarks to pixel coordinates
        keypoints = self._get_keypoints(landmarks, w, h)
        
        # Calculate fit metrics
        metrics = self._calculate_fit_metrics(keypoints, clothing_mask, clothing_points)
        
        return metrics
    
    def _get_keypoints(self, landmarks, width: int, height: int) -> Dict[str, np.ndarray]:
        """Extract keypoints from landmarks in pixel coordinates"""
        keypoints = {}
        
        # Extract key body points
        keypoints['left_shoulder'] = np.array([
            landmarks[self.SHOULDER_LEFT].x * width,
            landmarks[self.SHOULDER_LEFT].y * height
        ])
        keypoints['right_shoulder'] = np.array([
            landmarks[self.SHOULDER_RIGHT].x * width,
            landmarks[self.SHOULDER_RIGHT].y * height
        ])
        keypoints['left_waist'] = np.array([
            landmarks[self.WAIST_LEFT].x * width,
            landmarks[self.WAIST_LEFT].y * height
        ])
        keypoints['right_waist'] = np.array([
            landmarks[self.WAIST_RIGHT].x * width,
            landmarks[self.WAIST_RIGHT].y * height
        ])
        keypoints['neck'] = np.array([
            landmarks[self.NECK].x * width,
            landmarks[self.NECK].y * height
        ])
        
        return keypoints
    
    def _calculate_fit_metrics(self, keypoints: Dict[str, np.ndarray],
                             clothing_mask: np.ndarray,
                             clothing_points: np.ndarray) -> FitMetrics:
        """Calculate various fit metrics"""
        # Calculate shoulder alignment
        shoulder_alignment = self._calculate_shoulder_alignment(
            keypoints['left_shoulder'],
            keypoints['right_shoulder'],
            clothing_points
        )
        
        # Calculate chest fit
        chest_fit = self._calculate_chest_fit(
            keypoints,
            clothing_mask
        )
        
        # Calculate waist fit
        waist_fit = self._calculate_waist_fit(
            keypoints['left_waist'],
            keypoints['right_waist'],
            clothing_mask
        )
        
        # Calculate arm hole positioning
        arm_holes = self._calculate_arm_holes_fit(
            keypoints,
            clothing_mask
        )
        
        # Calculate overall score
        weights = {
            'shoulder': 0.3,
            'chest': 0.3,
            'waist': 0.2,
            'arms': 0.2
        }
        
        overall_score = (
            weights['shoulder'] * shoulder_alignment +
            weights['chest'] * chest_fit +
            weights['waist'] * waist_fit +
            weights['arms'] * arm_holes
        )
        
        # Identify problem areas
        problem_areas = []
        if shoulder_alignment < self.MODERATE_FIT_THRESHOLD:
            problem_areas.append('shoulders')
        if chest_fit < self.MODERATE_FIT_THRESHOLD:
            problem_areas.append('chest')
        if waist_fit < self.MODERATE_FIT_THRESHOLD:
            problem_areas.append('waist')
        if arm_holes < self.MODERATE_FIT_THRESHOLD:
            problem_areas.append('arm_holes')
        
        return FitMetrics(
            shoulder_alignment=shoulder_alignment,
            chest_fit=chest_fit,
            waist_fit=waist_fit,
            arm_holes=arm_holes,
            overall_score=overall_score,
            problem_areas=problem_areas
        )
    
    def generate_heatmap(self, frame: np.ndarray, metrics: FitMetrics,
                        keypoints: Dict[str, np.ndarray]) -> np.ndarray:
        """Generate a heatmap overlay based on fit metrics"""
        h, w = frame.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        # Generate heatmap for each body region
        self._add_shoulder_heatmap(heatmap, keypoints, metrics.shoulder_alignment)
        self._add_chest_heatmap(heatmap, keypoints, metrics.chest_fit)
        self._add_waist_heatmap(heatmap, keypoints, metrics.waist_fit)
        self._add_arm_holes_heatmap(heatmap, keypoints, metrics.arm_holes)
        
        # Normalize heatmap
        heatmap = cv2.normalize(heatmap, None, 0, 1, cv2.NORM_MINMAX)
        
        # Convert heatmap to color
        heatmap_color = self._heatmap_to_color(heatmap)
        
        # Blend with original frame
        alpha = 0.5
        blended = cv2.addWeighted(frame, 1, heatmap_color, alpha, 0)
        
        return blended
    
    def _heatmap_to_color(self, heatmap: np.ndarray) -> np.ndarray:
        """Convert grayscale heatmap to color visualization"""
        # Create color map
        heatmap_color = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        
        # Adjust color map (red for poor fit, green for good fit)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        return heatmap_color
    
    def _add_shoulder_heatmap(self, heatmap: np.ndarray,
                            keypoints: Dict[str, np.ndarray],
                            score: float):
        """Add shoulder region to heatmap"""
        left_shoulder = keypoints['left_shoulder']
        right_shoulder = keypoints['right_shoulder']
        
        # Create shoulder region mask
        shoulder_mask = np.zeros_like(heatmap)
        cv2.line(shoulder_mask,
                tuple(map(int, left_shoulder)),
                tuple(map(int, right_shoulder)),
                1, thickness=20)
        
        # Add to heatmap with intensity based on score
        intensity = 1 - score  # Invert score (higher intensity for poor fit)
        heatmap += shoulder_mask * intensity
    
    def _add_chest_heatmap(self, heatmap: np.ndarray,
                          keypoints: Dict[str, np.ndarray],
                          score: float):
        """Add chest region to heatmap"""
        left_shoulder = keypoints['left_shoulder']
        right_shoulder = keypoints['right_shoulder']
        left_waist = keypoints['left_waist']
        right_waist = keypoints['right_waist']
        
        # Create chest region polygon
        pts = np.array([
            left_shoulder,
            right_shoulder,
            right_waist,
            left_waist
        ], dtype=np.int32)
        
        # Create chest region mask
        chest_mask = np.zeros_like(heatmap)
        cv2.fillPoly(chest_mask, [pts], 1)
        
        # Add to heatmap with intensity based on score
        intensity = 1 - score
        heatmap += chest_mask * intensity
    
    def _add_waist_heatmap(self, heatmap: np.ndarray,
                          keypoints: Dict[str, np.ndarray],
                          score: float):
        """Add waist region to heatmap"""
        left_waist = keypoints['left_waist']
        right_waist = keypoints['right_waist']
        
        # Create waist region mask
        waist_mask = np.zeros_like(heatmap)
        cv2.line(waist_mask,
                tuple(map(int, left_waist)),
                tuple(map(int, right_waist)),
                1, thickness=20)
        
        # Add to heatmap with intensity based on score
        intensity = 1 - score
        heatmap += waist_mask * intensity
    
    def _add_arm_holes_heatmap(self, heatmap: np.ndarray,
                              keypoints: Dict[str, np.ndarray],
                              score: float):
        """Add arm holes region to heatmap"""
        left_shoulder = keypoints['left_shoulder']
        right_shoulder = keypoints['right_shoulder']
        
        # Create arm holes masks
        arm_holes_mask = np.zeros_like(heatmap)
        
        # Left arm hole
        cv2.circle(arm_holes_mask,
                  tuple(map(int, left_shoulder)),
                  radius=15,
                  color=1,
                  thickness=-1)
        
        # Right arm hole
        cv2.circle(arm_holes_mask,
                  tuple(map(int, right_shoulder)),
                  radius=15,
                  color=1,
                  thickness=-1)
        
        # Add to heatmap with intensity based on score
        intensity = 1 - score
        heatmap += arm_holes_mask * intensity
    
    def _calculate_shoulder_alignment(self, left_shoulder: np.ndarray,
                                   right_shoulder: np.ndarray,
                                   clothing_points: np.ndarray) -> float:
        """Calculate shoulder alignment score"""
        # Find clothing shoulder points
        clothing_top = clothing_points[clothing_points[:, 1].argsort()][:2]
        
        # Calculate alignment error
        left_error = np.linalg.norm(clothing_top[0] - left_shoulder)
        right_error = np.linalg.norm(clothing_top[1] - right_shoulder)
        
        # Normalize error and convert to score
        max_error = np.linalg.norm(right_shoulder - left_shoulder) * 0.5
        score = 1 - (left_error + right_error) / (2 * max_error)
        
        return max(0, min(1, score))
    
    def _calculate_chest_fit(self, keypoints: Dict[str, np.ndarray],
                           clothing_mask: np.ndarray) -> float:
        """Calculate chest fit score"""
        # Calculate chest region coverage
        chest_region = self._get_chest_region_mask(keypoints)
        overlap = cv2.bitwise_and(chest_region, clothing_mask)
        
        coverage = np.sum(overlap) / np.sum(chest_region)
        return max(0, min(1, coverage))
    
    def _calculate_waist_fit(self, left_waist: np.ndarray,
                           right_waist: np.ndarray,
                           clothing_mask: np.ndarray) -> float:
        """Calculate waist fit score"""
        # Create waist region mask
        waist_mask = np.zeros_like(clothing_mask)
        cv2.line(waist_mask,
                tuple(map(int, left_waist)),
                tuple(map(int, right_waist)),
                1, thickness=20)
        
        # Calculate overlap
        overlap = cv2.bitwise_and(waist_mask, clothing_mask)
        coverage = np.sum(overlap) / np.sum(waist_mask)
        
        return max(0, min(1, coverage))
    
    def _calculate_arm_holes_fit(self, keypoints: Dict[str, np.ndarray],
                               clothing_mask: np.ndarray) -> float:
        """Calculate arm holes fit score"""
        # Create arm holes mask
        arm_holes_mask = np.zeros_like(clothing_mask)
        
        # Add circles for arm holes
        for shoulder in [keypoints['left_shoulder'], keypoints['right_shoulder']]:
            cv2.circle(arm_holes_mask,
                      tuple(map(int, shoulder)),
                      radius=15,
                      color=1,
                      thickness=-1)
        
        # Calculate overlap
        overlap = cv2.bitwise_and(arm_holes_mask, clothing_mask)
        coverage = np.sum(overlap) / np.sum(arm_holes_mask)
        
        return max(0, min(1, coverage))
    
    def _get_chest_region_mask(self, keypoints: Dict[str, np.ndarray]) -> np.ndarray:
        """Create mask for chest region"""
        pts = np.array([
            keypoints['left_shoulder'],
            keypoints['right_shoulder'],
            keypoints['right_waist'],
            keypoints['left_waist']
        ], dtype=np.int32)
        
        mask = np.zeros_like(clothing_mask)
        cv2.fillPoly(mask, [pts], 1)
        
        return mask
