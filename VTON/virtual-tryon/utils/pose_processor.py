import mediapipe as mp
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
import time

class PoseProcessor:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.frame_queue = Queue(maxsize=2)  # Buffer for async processing
        self.result_queue = Queue()
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        self.running = False
        self.last_process_time = 0
        self.fps = 0

    def start_processing(self):
        """Start async frame processing"""
        self.running = True
        self.thread_pool.submit(self._process_frames)

    def stop_processing(self):
        """Stop async frame processing"""
        self.running = False
        self.thread_pool.shutdown()

    def _process_frames(self):
        """Background thread for frame processing"""
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                start_time = time.time()
                
                # Process frame
                result = self.process_frame(frame)
                
                # Calculate FPS
                process_time = time.time() - start_time
                self.fps = 1 / process_time if process_time > 0 else 0
                self.last_process_time = process_time
                
                self.result_queue.put(result)

    def process_frame(self, frame):
        """Process a single frame to detect pose and prepare for virtual try-on"""
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if not results.pose_landmarks:
            return None

        # Extract key points
        landmarks = {}
        img_h, img_w, _ = frame.shape
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            landmarks[idx] = np.array([
                landmark.x * img_w,
                landmark.y * img_h,
                landmark.z * img_w  # Use width for z to maintain aspect ratio
            ])

        # Calculate body measurements
        measurements = self._calculate_body_measurements(landmarks)
        
        # Create body mask
        body_mask = self._create_body_mask(frame, landmarks)
        
        return {
            'landmarks': landmarks,
            'measurements': measurements,
            'body_mask': body_mask
        }

    def _calculate_body_measurements(self, landmarks):
        """Calculate body measurements from landmarks"""
        measurements = {}
        
        # Shoulder width
        left_shoulder = landmarks.get(11)  # Left shoulder
        right_shoulder = landmarks.get(12)  # Right shoulder
        if left_shoulder is not None and right_shoulder is not None:
            measurements['shoulder_width'] = np.linalg.norm(left_shoulder - right_shoulder)
        
        # Torso length
        neck = landmarks.get(0)  # Nose as reference for neck
        hip = np.mean([landmarks.get(23, 0), landmarks.get(24, 0)], axis=0)  # Mid-hip
        if neck is not None:
            measurements['torso_length'] = np.linalg.norm(neck - hip)
        
        # Waist width approximation
        left_hip = landmarks.get(23)  # Left hip
        right_hip = landmarks.get(24)  # Right hip
        if left_hip is not None and right_hip is not None:
            measurements['waist_width'] = np.linalg.norm(left_hip - right_hip)
        
        return measurements

    def _create_body_mask(self, frame, landmarks):
        """Create a binary mask of the body"""
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        # Connect landmarks to create body segments
        connections = [
            # Torso
            (11, 12), (11, 23), (12, 24), (23, 24),
            # Arms
            (11, 13), (13, 15), (12, 14), (14, 16),
            # Legs
            (23, 25), (25, 27), (24, 26), (26, 28)
        ]
        
        for start_idx, end_idx in connections:
            if start_idx in landmarks and end_idx in landmarks:
                start_point = landmarks[start_idx][:2].astype(int)
                end_point = landmarks[end_idx][:2].astype(int)
                cv2.line(mask, tuple(start_point), tuple(end_point), 255, thickness=20)
        
        # Fill body segments
        cv2.floodFill(mask, None, (int(landmarks[11][0]), int(landmarks[11][1])), 255)
        
        # Apply morphological operations to smooth the mask
        kernel = np.ones((15, 15), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask

    def estimate_depth(self, landmarks):
        """Estimate depth values for better clothing placement"""
        depth_map = {}
        
        # Use hip points as reference depth (0)
        mid_hip_z = (landmarks[23][2] + landmarks[24][2]) / 2
        
        for idx, point in landmarks.items():
            # Calculate relative depth from mid-hip
            depth_map[idx] = point[2] - mid_hip_z
        
        return depth_map

    def add_frame(self, frame):
        """Add a frame to the processing queue"""
        if not self.frame_queue.full():
            self.frame_queue.put(frame)
            return True
        return False

    def get_latest_result(self):
        """Get the latest processed result"""
        if not self.result_queue.empty():
            return self.result_queue.get()
        return None

    def get_performance_stats(self):
        """Get processing performance statistics"""
        return {
            'fps': self.fps,
            'latency': self.last_process_time * 1000  # Convert to ms
        }
