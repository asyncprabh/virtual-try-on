import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class Pose3D:
    landmarks_3d: np.ndarray  # (N, 3) array of 3D landmarks
    visibility: np.ndarray    # (N,) array of visibility scores
    rotation: np.ndarray      # (3,) array of rotation angles
    depth_scale: float       # Depth scaling factor

class PoseEstimator3D:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # Use the most accurate model
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # MediaPipe pose landmark indices
        self.SHOULDER_LEFT = 11
        self.SHOULDER_RIGHT = 12
        self.ELBOW_LEFT = 13
        self.ELBOW_RIGHT = 14
        self.WAIST_LEFT = 23
        self.WAIST_RIGHT = 24
        self.HIP_LEFT = 23
        self.HIP_RIGHT = 24
        self.NECK = 0
        
        # Initialize smoothing filters
        self.landmark_history = []
        self.max_history = 5
        
    def process_frame(self, frame: np.ndarray) -> Optional[Pose3D]:
        """Process a frame and return 3D pose information"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get pose landmarks
        results = self.pose.process(rgb_frame)
        if not results.pose_world_landmarks:
            return None
            
        # Extract 3D landmarks
        landmarks_3d = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks.landmark])
        visibility = np.array([lm.visibility for lm in results.pose_world_landmarks.landmark])
        
        # Apply temporal smoothing
        self.landmark_history.append(landmarks_3d)
        if len(self.landmark_history) > self.max_history:
            self.landmark_history.pop(0)
        
        smoothed_landmarks = np.mean(self.landmark_history, axis=0)
        
        # Calculate rotation angles
        rotation = self._calculate_rotation(smoothed_landmarks)
        
        # Calculate depth scale
        depth_scale = self._calculate_depth_scale(smoothed_landmarks)
        
        return Pose3D(
            landmarks_3d=smoothed_landmarks,
            visibility=visibility,
            rotation=rotation,
            depth_scale=depth_scale
        )
    
    def _calculate_rotation(self, landmarks_3d: np.ndarray) -> np.ndarray:
        """Calculate rotation angles from 3D landmarks"""
        # Get key points
        left_shoulder = landmarks_3d[self.SHOULDER_LEFT]
        right_shoulder = landmarks_3d[self.SHOULDER_RIGHT]
        left_hip = landmarks_3d[self.HIP_LEFT]
        right_hip = landmarks_3d[self.HIP_RIGHT]
        
        # Calculate rotation angles
        # Yaw (around vertical axis)
        shoulder_vector = right_shoulder - left_shoulder
        yaw = np.arctan2(shoulder_vector[2], shoulder_vector[0])
        
        # Pitch (forward/backward tilt)
        torso_vector = (left_shoulder + right_shoulder) / 2 - (left_hip + right_hip) / 2
        pitch = np.arctan2(torso_vector[2], torso_vector[1])
        
        # Roll (side-to-side tilt)
        roll = np.arctan2(shoulder_vector[1], shoulder_vector[0])
        
        return np.array([yaw, pitch, roll])
    
    def _calculate_depth_scale(self, landmarks_3d: np.ndarray) -> float:
        """Calculate depth scaling factor based on shoulder width"""
        left_shoulder = landmarks_3d[self.SHOULDER_LEFT]
        right_shoulder = landmarks_3d[self.SHOULDER_RIGHT]
        
        # Use shoulder width as reference for depth
        shoulder_width_3d = np.linalg.norm(right_shoulder - left_shoulder)
        
        # Normalize to a reference width (you may need to adjust this)
        reference_width = 0.5
        depth_scale = shoulder_width_3d / reference_width
        
        return depth_scale
    
    def get_transform_matrix(self, pose_3d: Pose3D, frame_size: Tuple[int, int]) -> np.ndarray:
        """Calculate 3D transformation matrix for clothing overlay"""
        height, width = frame_size
        
        # Extract rotation angles
        yaw, pitch, roll = pose_3d.rotation
        
        # Create rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # Combine rotations
        R = Rz @ Ry @ Rx
        
        # Create full transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = R
        
        # Add scaling
        scale_matrix = np.eye(4)
        scale = pose_3d.depth_scale
        scale_matrix[0, 0] = scale
        scale_matrix[1, 1] = scale
        scale_matrix[2, 2] = scale
        
        # Combine transformations
        transform = transform @ scale_matrix
        
        return transform
    
    def get_keypoints_2d(self, pose_3d: Pose3D, frame_size: Tuple[int, int]) -> dict:
        """Project 3D landmarks to 2D image coordinates"""
        height, width = frame_size
        
        # Define key points we're interested in
        key_points = {
            'left_shoulder': self.SHOULDER_LEFT,
            'right_shoulder': self.SHOULDER_RIGHT,
            'left_elbow': self.ELBOW_LEFT,
            'right_elbow': self.ELBOW_RIGHT,
            'left_waist': self.WAIST_LEFT,
            'right_waist': self.WAIST_RIGHT,
            'neck': self.NECK
        }
        
        # Project 3D points to 2D
        keypoints_2d = {}
        for name, idx in key_points.items():
            point_3d = pose_3d.landmarks_3d[idx]
            # Simple perspective projection (you might want to use a more sophisticated model)
            x = point_3d[0] * width + width / 2
            y = point_3d[1] * height + height / 2
            visibility = pose_3d.visibility[idx]
            
            keypoints_2d[name] = {
                'x': x,
                'y': y,
                'z': point_3d[2],
                'visibility': visibility
            }
        
        return keypoints_2d
