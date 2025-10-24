import numpy as np
import cv2
from dataclasses import dataclass
from typing import Tuple, List, Dict

@dataclass
class BodyMeasurements:
    shoulder_width: float
    torso_height: float
    waist_width: float
    shoulder_slope: float
    shoulder_midpoint: Tuple[float, float]

class KeypointTracker:
    def __init__(self):
        # MediaPipe pose landmark indices
        self.SHOULDER_LEFT = 11
        self.SHOULDER_RIGHT = 12
        self.WAIST_LEFT = 23
        self.WAIST_RIGHT = 24
        self.HIP_LEFT = 23
        self.HIP_RIGHT = 24
        
        # Smoothing parameters
        self.smooth_factor = 0.7
        self.prev_measurements = None
        
    def get_body_measurements(self, landmarks) -> BodyMeasurements:
        """Extract and calculate body measurements from landmarks"""
        if not landmarks:
            return None
            
        # Extract key points
        left_shoulder = np.array([landmarks[self.SHOULDER_LEFT].x, 
                                landmarks[self.SHOULDER_LEFT].y])
        right_shoulder = np.array([landmarks[self.SHOULDER_RIGHT].x, 
                                 landmarks[self.SHOULDER_RIGHT].y])
        left_waist = np.array([landmarks[self.WAIST_LEFT].x, 
                             landmarks[self.WAIST_LEFT].y])
        right_waist = np.array([landmarks[self.WAIST_RIGHT].x, 
                              landmarks[self.WAIST_RIGHT].y])
        
        # Calculate measurements
        shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
        waist_width = np.linalg.norm(right_waist - left_waist)
        
        # Calculate shoulder midpoint
        shoulder_midpoint = (left_shoulder + right_shoulder) / 2
        
        # Calculate torso height (shoulder to waist)
        waist_midpoint = (left_waist + right_waist) / 2
        torso_height = np.linalg.norm(shoulder_midpoint - waist_midpoint)
        
        # Calculate shoulder slope
        shoulder_slope = np.arctan2(right_shoulder[1] - left_shoulder[1],
                                  right_shoulder[0] - left_shoulder[0])
        
        measurements = BodyMeasurements(
            shoulder_width=shoulder_width,
            torso_height=torso_height,
            waist_width=waist_width,
            shoulder_slope=shoulder_slope,
            shoulder_midpoint=tuple(shoulder_midpoint)
        )
        
        # Apply smoothing if we have previous measurements
        if self.prev_measurements:
            measurements = self._smooth_measurements(measurements)
        
        self.prev_measurements = measurements
        return measurements
    
    def _smooth_measurements(self, current: BodyMeasurements) -> BodyMeasurements:
        """Apply exponential smoothing to measurements"""
        if not self.prev_measurements:
            return current
            
        a = self.smooth_factor
        prev = self.prev_measurements
        
        return BodyMeasurements(
            shoulder_width=a * current.shoulder_width + (1 - a) * prev.shoulder_width,
            torso_height=a * current.torso_height + (1 - a) * prev.torso_height,
            waist_width=a * current.waist_width + (1 - a) * prev.waist_width,
            shoulder_slope=a * current.shoulder_slope + (1 - a) * prev.shoulder_slope,
            shoulder_midpoint=(
                a * current.shoulder_midpoint[0] + (1 - a) * prev.shoulder_midpoint[0],
                a * current.shoulder_midpoint[1] + (1 - a) * prev.shoulder_midpoint[1]
            )
        )
    
    def calculate_shirt_transform(self, measurements: BodyMeasurements, 
                                frame_shape: Tuple[int, int],
                                shirt_shape: Tuple[int, int]) -> Dict:
        """Calculate transformation parameters for the shirt"""
        if not measurements:
            return None
            
        frame_height, frame_width = frame_shape
        shirt_height, shirt_width = shirt_shape
        
        # Convert normalized coordinates to pixel coordinates
        shoulder_mid_x = int(measurements.shoulder_midpoint[0] * frame_width)
        shoulder_mid_y = int(measurements.shoulder_midpoint[1] * frame_height)
        
        # Calculate scaling factors
        shoulder_width_pixels = measurements.shoulder_width * frame_width
        torso_height_pixels = measurements.torso_height * frame_height
        
        scale_x = shoulder_width_pixels / shirt_width * 1.2  # Add 20% for comfort
        scale_y = torso_height_pixels / shirt_height
        
        # Calculate shirt position
        shirt_x = shoulder_mid_x - (shirt_width * scale_x) / 2
        shirt_y = shoulder_mid_y - (shirt_height * scale_y) * 0.2  # Place top 20% above shoulders
        
        return {
            'position': (int(shirt_x), int(shirt_y)),
            'scale': (scale_x, scale_y),
            'rotation': measurements.shoulder_slope,
            'shoulder_width': shoulder_width_pixels,
            'torso_height': torso_height_pixels
        }
    
    def get_warping_points(self, measurements: BodyMeasurements, 
                          frame_shape: Tuple[int, int],
                          transform: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate warping points for perspective transform"""
        frame_height, frame_width = frame_shape
        position = transform['position']
        scale = transform['scale']
        rotation = transform['rotation']
        
        # Source points (shirt corners)
        src_points = np.array([
            [0, 0],  # Top-left
            [shirt_width, 0],  # Top-right
            [shirt_width, shirt_height],  # Bottom-right
            [0, shirt_height]  # Bottom-left
        ], dtype=np.float32)
        
        # Calculate destination points with rotation and scaling
        cos_rot = np.cos(rotation)
        sin_rot = np.sin(rotation)
        
        def rotate_point(x, y):
            rx = x * cos_rot - y * sin_rot
            ry = x * sin_rot + y * cos_rot
            return rx, ry
        
        dst_points = []
        for x, y in src_points:
            # Scale
            x *= scale[0]
            y *= scale[1]
            
            # Rotate
            x, y = rotate_point(x, y)
            
            # Translate
            x += position[0]
            y += position[1]
            
            dst_points.append([x, y])
            
        return src_points, np.array(dst_points, dtype=np.float32)
    
    def adjust_shirt_points(self, points: np.ndarray, 
                          measurements: BodyMeasurements,
                          frame_shape: Tuple[int, int]) -> np.ndarray:
        """Adjust shirt control points based on body measurements"""
        frame_height, frame_width = frame_shape
        
        # Convert normalized measurements to pixel coordinates
        waist_width_px = measurements.waist_width * frame_width
        shoulder_width_px = measurements.shoulder_width * frame_width
        
        # Calculate waist reduction factor
        waist_factor = waist_width_px / shoulder_width_px
        
        # Adjust control points for waist tapering
        adjusted_points = points.copy()
        
        # Find waist level points and adjust their position
        for i, point in enumerate(adjusted_points):
            # Assuming points near vertical center are waist points
            relative_height = point[1] / frame_height
            if 0.4 <= relative_height <= 0.7:  # Waist region
                # Calculate horizontal adjustment
                distance_from_center = point[0] - frame_width / 2
                adjusted_x = (frame_width / 2 + 
                            distance_from_center * waist_factor)
                adjusted_points[i, 0] = adjusted_x
                
        return adjusted_points
