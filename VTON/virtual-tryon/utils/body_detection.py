import cv2
import mediapipe as mp
import numpy as np

class BodyDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # Increased model complexity for better accuracy
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def detect_body(self, image):
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        return results

    def overlay_clothing(self, image, clothing_image, pose_landmarks):
        if not pose_landmarks:
            return image

        ih, iw, _ = image.shape
        
        try:
            # Get key body landmarks
            l_shoulder = (int(pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x * iw),
                        int(pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y * ih))
            r_shoulder = (int(pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x * iw),
                        int(pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y * ih))
            l_hip = (int(pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].x * iw),
                    int(pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y * ih))
            r_hip = (int(pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].x * iw),
                    int(pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].y * ih))

            # Calculate clothing dimensions based on body
            shoulder_width = int(np.sqrt((r_shoulder[0] - l_shoulder[0])**2 + 
                                       (r_shoulder[1] - l_shoulder[1])**2))
            torso_height = int(np.sqrt((l_shoulder[0] - l_hip[0])**2 + 
                                     (l_shoulder[1] - l_hip[1])**2))

            # Add padding to dimensions
            shoulder_width = int(shoulder_width * 1.2)  # 20% wider
            torso_height = int(torso_height * 1.1)  # 10% taller

            # Read and preprocess clothing image
            if isinstance(clothing_image, str):
                clothing_image = cv2.imread(clothing_image, cv2.IMREAD_UNCHANGED)
                if clothing_image is None:
                    raise ValueError("Could not load clothing image")

            # Create alpha channel if not present
            if clothing_image.shape[2] == 3:
                clothing_image = cv2.cvtColor(clothing_image, cv2.COLOR_BGR2BGRA)

            # Resize clothing to match body dimensions
            clothing_resized = cv2.resize(clothing_image, (shoulder_width, torso_height))
            
            # Calculate position to place clothing
            center_x = (l_shoulder[0] + r_shoulder[0]) // 2
            center_y = (l_shoulder[1] + r_shoulder[1]) // 2
            
            # Calculate top-left corner for overlay
            x_offset = max(0, center_x - clothing_resized.shape[1] // 2)
            y_offset = max(0, center_y - clothing_resized.shape[0] // 4)  # Adjusted to align better with shoulders
            
            # Ensure we don't go out of bounds
            if x_offset + clothing_resized.shape[1] > iw:
                clothing_resized = clothing_resized[:, :iw - x_offset]
            if y_offset + clothing_resized.shape[0] > ih:
                clothing_resized = clothing_resized[:ih - y_offset, :]

            # Get the alpha channel
            alpha = clothing_resized[:, :, 3] / 255.0
            alpha = np.stack([alpha] * 3, axis=-1)

            # Get the region of interest (ROI)
            roi = image[y_offset:y_offset + clothing_resized.shape[0],
                       x_offset:x_offset + clothing_resized.shape[1]]

            # Blend the clothing with the image using alpha channel
            blended = (1 - alpha) * roi + alpha * clothing_resized[:, :, :3]
            
            # Put the blended region back into the image
            image[y_offset:y_offset + clothing_resized.shape[0],
                  x_offset:x_offset + clothing_resized.shape[1]] = blended

        except Exception as e:
            print(f"Error overlaying clothing: {str(e)}")
            return image
            
        return image

    def process_frame(self, frame, clothing_path):
        # Detect body in frame
        pose_results = self.detect_body(frame)
        
        if pose_results.pose_landmarks:
            # Overlay clothing on the detected body
            frame = self.overlay_clothing(frame.copy(), clothing_path, pose_results.pose_landmarks)
            
        return frame
