import cv2
import numpy as np
from .keypoint_tracker import KeypointTracker, BodyMeasurements
from .fit_analyzer import FitAnalyzer, FitMetrics
from .pose_estimator_3d import PoseEstimator3D, Pose3D

class ClothingAdjuster:
    def __init__(self):
        self.keypoint_tracker = KeypointTracker()
        self.fit_analyzer = FitAnalyzer()
        self.pose_estimator = PoseEstimator3D()
        self.last_transform = None
        # MediaPipe pose landmark indices
        self.SHOULDER_LEFT = 11
        self.SHOULDER_RIGHT = 12
        self.WAIST_LEFT = 23
        self.WAIST_RIGHT = 24
        self.HIP_LEFT = 23
        self.HIP_RIGHT = 24
        self.NECK = 0
        
    def adjust_clothing(self, frame, clothing_img, landmarks):
        """Adjust clothing based on 3D pose estimation"""
        if frame is None or clothing_img is None or landmarks is None:
            return frame, None, None
            
        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]
        clothing_height, clothing_width = clothing_img.shape[:2]
        
        # Get 3D pose information
        pose_3d = self.pose_estimator.process_frame(frame)
        if pose_3d is None:
            return frame, None, None
            
        # Get 2D keypoints from 3D pose
        keypoints_2d = self.pose_estimator.get_keypoints_2d(pose_3d, frame.shape[:2])
        
        # Calculate 3D transformation matrix
        transform_3d = self.pose_estimator.get_transform_matrix(pose_3d, frame.shape[:2])
        
        # Extract key measurements
        left_shoulder = np.array([keypoints_2d['left_shoulder']['x'],
                                keypoints_2d['left_shoulder']['y']])
        right_shoulder = np.array([keypoints_2d['right_shoulder']['x'],
                                 keypoints_2d['right_shoulder']['y']])
        left_waist = np.array([keypoints_2d['left_waist']['x'],
                             keypoints_2d['left_waist']['y']])
        right_waist = np.array([keypoints_2d['right_waist']['x'],
                              keypoints_2d['right_waist']['y']])
        neck = np.array([keypoints_2d['neck']['x'],
                        keypoints_2d['neck']['y']])
        
        # Calculate body measurements
        shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
        waist_width = np.linalg.norm(right_waist - left_waist)
        torso_height = np.mean([
            np.linalg.norm(left_shoulder - left_waist),
            np.linalg.norm(right_shoulder - right_waist)
        ])
        
        # Calculate scaling factors with depth consideration
        depth_scale = pose_3d.depth_scale
        width_scale = (shoulder_width / clothing_width * 1.1) * depth_scale
        height_scale = (torso_height / clothing_height * 1.2) * depth_scale
        
        # Calculate shirt position with depth consideration
        shoulder_midpoint = (left_shoulder + right_shoulder) / 2
        neck_to_shoulder = np.mean([
            np.linalg.norm(neck - left_shoulder),
            np.linalg.norm(neck - right_shoulder)
        ])
        
        # Adjust shirt placement considering 3D rotation
        rotation_matrix = transform_3d[:3, :3]
        shirt_top = shoulder_midpoint[1] - neck_to_shoulder * 0.5
        shirt_left = shoulder_midpoint[0] - (shoulder_width * 0.55)
        
        # Calculate corner points for perspective transform
        src_points = np.array([
            [0, 0],
            [clothing_width, 0],
            [clothing_width, clothing_height],
            [0, clothing_height]
        ], dtype=np.float32)
        
        # Apply 3D transformation to destination points
        dst_points = []
        for src_point in src_points:
            x, y = src_point
            
            # Normalize coordinates
            x_norm = x / clothing_width - 0.5
            y_norm = y / clothing_height - 0.5
            
            # Apply 3D transformation
            point_3d = np.array([x_norm, y_norm, 0, 1])
            transformed = transform_3d @ point_3d
            
            # Project back to 2D
            x_proj = transformed[0] / transformed[3] * width_scale + shirt_left
            y_proj = transformed[1] / transformed[3] * height_scale + shirt_top
            
            # Add curved transformation
            if y == 0:  # Top points
                x_proj = shirt_left + (x / clothing_width) * shoulder_width
                y_proj = shirt_top
            else:  # Bottom points
                x_proj = shirt_left + (x / clothing_width) * waist_width
                y_proj = shirt_top + (y / clothing_height) * torso_height
                
                # Add curve to sides
                if x == 0 or x == clothing_width:
                    curve_factor = np.sin(y / clothing_height * np.pi) * 0.1 * shoulder_width
                    x_proj += curve_factor if x == 0 else -curve_factor
            
            dst_points.append([x_proj, y_proj])
        
        dst_points = np.array(dst_points, dtype=np.float32)
        
        # Calculate and apply perspective transform
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        warped_clothing = cv2.warpPerspective(
            clothing_img,
            matrix,
            (frame_width, frame_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_TRANSPARENT
        )
        
        # Create mask for the warped clothing
        if warped_clothing.shape[2] == 4:  # Image has alpha channel
            mask = warped_clothing[:, :, 3]
        else:
            gray = cv2.cvtColor(warped_clothing, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # Apply Gaussian blur to the mask for smoother edges
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # Analyze fit and generate heatmap
        fit_metrics = self.fit_analyzer.analyze_fit(frame, mask, landmarks, dst_points)
        heatmap = self.fit_analyzer.generate_heatmap(frame, fit_metrics, keypoints_2d)
        
        # Blend the warped clothing with the frame
        result = self._blend_images(frame, warped_clothing, mask)
        
        return result, fit_metrics, heatmap
    
    def _blend_images(self, frame, clothing, mask):
        """Blend clothing onto frame using mask with smooth edges"""
        # Normalize mask to range [0, 1]
        mask = mask.astype(float) / 255
        
        # Expand mask to 3 channels
        mask = np.stack([mask] * 3, axis=2)
        
        # Extract BGR channels from clothing
        if clothing.shape[2] == 4:
            clothing = clothing[:, :, :3]
        
        # Blend images using mask
        blended = frame * (1 - mask) + clothing * mask
        
        return blended.astype(np.uint8)
    
    def get_debug_info(self, measurements: BodyMeasurements):
        """Get debug information for visualization"""
        if measurements is None:
            return None
            
        return {
            'shoulder_width': measurements.shoulder_width,
            'torso_height': measurements.torso_height,
            'waist_width': measurements.waist_width,
            'shoulder_slope': measurements.shoulder_slope,
            'shoulder_midpoint': measurements.shoulder_midpoint
        }
