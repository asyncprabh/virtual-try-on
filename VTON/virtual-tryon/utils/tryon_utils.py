import cv2
import numpy as np
import mediapipe as mp
import os
import sys

# Windows console encoding fix
if sys.platform.startswith('win'):
    import locale
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')

class EnhancedVirtualTryOn:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5
        )

    def preprocess_image(self, image):
        """Preprocess image for better visibility"""
        if image.shape[2] == 3:  # RGB/BGR image
            # Convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply CLAHE to enhance contrast
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            lab = cv2.merge((l,a,b))
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # Enhance saturation and brightness
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            s = cv2.multiply(s, 1.4).astype(np.uint8)  # Increase saturation
            v = cv2.multiply(v, 1.2).astype(np.uint8)  # Increase brightness
            hsv = cv2.merge([h, s, v])
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
        return image

    def preprocess_clothing(self, clothing_img):
        """Preprocess clothing image to remove background and enhance visibility"""
        try:
            # Convert to RGBA if needed
            if clothing_img.shape[2] == 3:
                clothing_img = cv2.cvtColor(clothing_img, cv2.COLOR_BGR2BGRA)
            
            # Create binary mask for non-white pixels
            gray = cv2.cvtColor(clothing_img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
            
            # Remove noise
            kernel = np.ones((3,3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # Create alpha channel from binary mask
            clothing_img[:, :, 3] = binary
            
            # Enhance clothing visibility
            clothing_rgb = clothing_img[:, :, :3].copy()
            clothing_lab = cv2.cvtColor(clothing_rgb, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(clothing_lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # Merge back
            clothing_lab = cv2.merge((l,a,b))
            clothing_rgb = cv2.cvtColor(clothing_lab, cv2.COLOR_LAB2BGR)
            
            # Update RGB channels while preserving alpha
            clothing_img[:, :, :3] = clothing_rgb
            
            return clothing_img
            
        except Exception as e:
            print(f"[ERROR] In preprocess_clothing: {str(e)}")
            return None

    def detect_body_features(self, image):
        """Detect body landmarks and create body mask"""
        try:
            # Convert to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get pose landmarks
            with self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                min_detection_confidence=0.5,
                enable_segmentation=True  # Enable segmentation for better masking
            ) as pose:
                results = pose.process(image_rgb)
                
                if not results.pose_landmarks:
                    print("[ERROR] No pose landmarks detected")
                    return None, None
                
                # Get relevant landmark points for upper body
                height, width = image.shape[:2]
                points = []
                upper_body_landmarks = [
                    self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                    self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    self.mp_pose.PoseLandmark.LEFT_ELBOW,
                    self.mp_pose.PoseLandmark.RIGHT_ELBOW,
                    self.mp_pose.PoseLandmark.LEFT_HIP,
                    self.mp_pose.PoseLandmark.RIGHT_HIP,
                    self.mp_pose.PoseLandmark.LEFT_WRIST,
                    self.mp_pose.PoseLandmark.RIGHT_WRIST
                ]
                
                for landmark in upper_body_landmarks:
                    x = int(results.pose_landmarks.landmark[landmark.value].x * width)
                    y = int(results.pose_landmarks.landmark[landmark.value].y * height)
                    points.append([x, y])
                
                points = np.array(points, np.int32)
                
                # Create refined mask for upper body region
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                
                # Create a smoother hull by adding interpolated points
                hull = cv2.convexHull(points)
                smooth_hull = []
                for i in range(len(hull)):
                    pt1 = hull[i][0]
                    pt2 = hull[(i + 1) % len(hull)][0]
                    # Add intermediate points
                    for t in range(0, 100, 10):
                        x = int(pt1[0] + (pt2[0] - pt1[0]) * t / 100)
                        y = int(pt1[1] + (pt2[1] - pt1[1]) * t / 100)
                        smooth_hull.append([[x, y]])
                
                smooth_hull = np.array(smooth_hull, dtype=np.int32)
                cv2.fillConvexPoly(mask, smooth_hull, 255)
                
                # Apply Gaussian blur to smooth the mask edges
                mask = cv2.GaussianBlur(mask, (5, 5), 0)
                
                # Dilate mask to ensure coverage
                kernel = np.ones((15,15), np.uint8)  # Reduced kernel size
                mask = cv2.dilate(mask, kernel, iterations=1)
                
                return results.pose_landmarks.landmark, mask
                
        except Exception as e:
            print(f"[ERROR] In detect_body_features: {str(e)}")
            return None, None

    def get_body_measurements(self, landmarks, image_shape):
        """Get body measurements from landmarks"""
        try:
            height, width = image_shape[:2]
            
            # Get key points without visualization
            left_shoulder = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width),
                           int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height))
            right_shoulder = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * width),
                            int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * height))
            left_hip = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * width),
                       int(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * height))
            right_hip = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x * width),
                        int(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y * height))
            
            # Calculate measurements silently
            shoulder_width = np.sqrt((right_shoulder[0] - left_shoulder[0])**2 + 
                                   (right_shoulder[1] - left_shoulder[1])**2)
            torso_height = np.sqrt((left_hip[0] - left_shoulder[0])**2 + 
                                 (left_hip[1] - left_shoulder[1])**2)
            
            # Calculate center points
            shoulder_center = ((left_shoulder[0] + right_shoulder[0])//2,
                             (left_shoulder[1] + right_shoulder[1])//2)
            hip_center = ((left_hip[0] + right_hip[0])//2,
                         (left_hip[1] + right_hip[1])//2)
            
            # Calculate body angle
            body_angle = np.arctan2(hip_center[1] - shoulder_center[1],
                                  hip_center[0] - shoulder_center[0])
            
            return {
                'shoulder_width': shoulder_width,
                'torso_height': torso_height,
                'shoulder_center': shoulder_center,
                'hip_center': hip_center,
                'body_angle': body_angle,
                'left_shoulder': left_shoulder,
                'right_shoulder': right_shoulder,
                'left_hip': left_hip,
                'right_hip': right_hip
            }
            
        except Exception as e:
            print(f"[ERROR] In get_body_measurements: {str(e)}")
            return None

    def warp_clothing_to_body(self, clothing_img, measurements, target_shape):
        """Warp clothing image to match body measurements"""
        try:
            if measurements is None:
                return None
                
            height, width = target_shape[:2]
            
            # Calculate target width based on shoulder width with increased size
            target_width = int(measurements['shoulder_width'] * 1.8)  # Increased from 1.5
            
            # Calculate target height while maintaining aspect ratio
            aspect_ratio = clothing_img.shape[0] / clothing_img.shape[1]
            target_height = int(target_width * aspect_ratio * 1.2)  # Increased from 0.9
            
            # Ensure height doesn't exceed torso with more allowance
            max_height = int(measurements['torso_height'] * 1.4)  # Increased from 1.05
            if target_height > max_height:
                target_height = max_height
                target_width = int(target_height / (aspect_ratio * 1.2))
            
            # Create perspective transform points
            src_points = np.float32([
                [0, 0],
                [clothing_img.shape[1], 0],
                [0, clothing_img.shape[0]],
                [clothing_img.shape[1], clothing_img.shape[0]]
            ])
            
            # Calculate shoulder slope with dampening
            shoulder_slope = (measurements['right_shoulder'][1] - measurements['left_shoulder'][1]) / \
                           (measurements['right_shoulder'][0] - measurements['left_shoulder'][0] + 1e-6)
            shoulder_slope *= 0.5  # Further reduce slope effect
            
            # Adjust target points based on body measurements
            shoulder_center = measurements['shoulder_center']
            hip_center = measurements['hip_center']
            
            # Calculate vertical positioning with higher placement
            vertical_offset = int(target_height * 0.15)  # Increased from 0.05
            
            # Calculate horizontal positioning with reduced angle compensation
            body_angle = measurements['body_angle']
            angle_compensation = np.tan(body_angle) * target_height * 0.1  # Reduced from 0.2
            
            # Calculate base points with wider spread
            top_left = [
                shoulder_center[0] - target_width//2 + int(angle_compensation), 
                shoulder_center[1] - vertical_offset
            ]
            top_right = [
                shoulder_center[0] + target_width//2 + int(angle_compensation), 
                shoulder_center[1] - vertical_offset
            ]
            bottom_left = [
                hip_center[0] - target_width//2 - int(angle_compensation), 
                hip_center[1] + vertical_offset  # Added positive offset for length
            ]
            bottom_right = [
                hip_center[0] + target_width//2 - int(angle_compensation), 
                hip_center[1] + vertical_offset  # Added positive offset for length
            ]
            
            # Apply shoulder slope with reduced effect
            slope_offset = int(target_width * shoulder_slope * 0.1)  # Reduced from 0.2
            top_left[1] -= slope_offset
            top_right[1] += slope_offset
            
            # Add slight curve to bottom points for better fit
            bottom_curve = int(target_width * 0.03)  # Reduced from 0.05
            bottom_left[0] += bottom_curve
            bottom_right[0] -= bottom_curve
            
            dst_points = np.float32([top_left, top_right, bottom_left, bottom_right])
            
            # Get perspective transform
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # Apply transform with improved interpolation
            warped = cv2.warpPerspective(
                clothing_img, 
                matrix, 
                (width, height),
                flags=cv2.INTER_CUBIC,  # Changed to CUBIC for better quality
                borderMode=cv2.BORDER_REPLICATE
            )
            
            return warped
            
        except Exception as e:
            print(f"[ERROR] In warp_clothing_to_body: {str(e)}")
            return None

    def get_clothing_position(self, landmarks, clothing_shape, image_shape):
        """Calculate optimal position for clothing overlay"""
        try:
            height, width = image_shape[:2]
            clothing_height, clothing_width = clothing_shape[:2]
            
            # Get shoulder points
            left_shoulder_x = int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width)
            left_shoulder_y = int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height)
            right_shoulder_x = int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * width)
            right_shoulder_y = int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * height)
            
            # Calculate center position
            center_x = (left_shoulder_x + right_shoulder_x) // 2
            center_y = (left_shoulder_y + right_shoulder_y) // 2
            
            # Calculate position to place clothing
            x = center_x - (clothing_width // 2)
            y = center_y - int(clothing_height * 0.35)  # Move up to cover shoulders better
            
            # Ensure within image bounds
            x = max(0, min(x, width - clothing_width))
            y = max(0, min(y, height - clothing_height))
            
            return (x, y)
            
        except Exception as e:
            print(f"[ERROR] In get_clothing_position: {str(e)}")
            return (0, 0)

    def blend_images(self, clothing, person, position, body_mask):
        """Blend clothing onto person image with improved alpha blending"""
        try:
            if clothing is None or person is None:
                return person
                
            # Create output image
            result = person.copy()
            
            # Get clothing alpha channel with enhanced opacity
            alpha = clothing[:, :, 3] / 255.0 if clothing.shape[2] == 4 else np.ones(clothing.shape[:2])
            alpha = np.clip(alpha * 1.3, 0, 1)  # Increase opacity by 30%
            
            # Apply Gaussian blur for smoother edges
            alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
            
            # Get clothing RGB channels
            clothing_rgb = clothing[:, :, :3]
            
            # Enhance clothing visibility
            clothing_lab = cv2.cvtColor(clothing_rgb, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(clothing_lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # Merge back
            clothing_lab = cv2.merge((l,a,b))
            clothing_rgb = cv2.cvtColor(clothing_lab, cv2.COLOR_LAB2BGR)
            
            # Enhance contrast
            clothing_rgb = cv2.convertScaleAbs(clothing_rgb, alpha=1.1, beta=5)
            
            # Apply color correction to match lighting
            person_hsv = cv2.cvtColor(person, cv2.COLOR_BGR2HSV)
            clothing_hsv = cv2.cvtColor(clothing_rgb, cv2.COLOR_BGR2HSV)
            
            # Match brightness while maintaining contrast
            avg_v_person = np.mean(person_hsv[:, :, 2])
            avg_v_clothing = np.mean(clothing_hsv[:, :, 2])
            brightness_ratio = min(avg_v_person / (avg_v_clothing + 1e-6), 1.2)  # Cap at 120%
            
            clothing_hsv[:, :, 2] = np.clip(clothing_hsv[:, :, 2] * brightness_ratio, 0, 255)
            clothing_rgb = cv2.cvtColor(clothing_hsv, cv2.COLOR_HSV2BGR)
            
            # Create alpha channels for blending
            alpha_3d = np.stack([alpha] * 3, axis=2)
            
            # Apply body mask to clothing mask
            body_mask_roi = body_mask[position[1]:position[1]+clothing.shape[0], position[0]:position[0]+clothing.shape[1]] / 255.0
            alpha_3d = alpha_3d * body_mask_roi
            
            # Blend images with enhanced visibility
            blended = cv2.addWeighted(
                (clothing_rgb * alpha_3d).astype(np.uint8),
                1.0,  # Increased weight for clothing
                (result[position[1]:position[1]+clothing.shape[0], position[0]:position[0]+clothing.shape[1]] * (1 - alpha_3d)).astype(np.uint8),
                0.8,  # Reduced weight for original image
                0
            )
            
            # Copy only the valid region with feathered edges
            valid_region = (alpha_3d > 0.1).any(axis=2)  # Reduced threshold
            result[position[1]:position[1]+clothing.shape[0], position[0]:position[0]+clothing.shape[1]][valid_region] = blended[valid_region]
            
            return result
            
        except Exception as e:
            print(f"[ERROR] In blend_images: {str(e)}")
            return person

    def try_on(self, person_image_path, clothing_image_path, output_path):
        """Main virtual try-on function with improved body fitting"""
        try:
            # Load images
            person_img = cv2.imread(person_image_path)
            if person_img is None:
                raise ValueError("Could not load person image")
            
            clothing_img = cv2.imread(clothing_image_path, cv2.IMREAD_UNCHANGED)
            if clothing_img is None:
                raise ValueError("Could not load clothing image")
                
            # Convert clothing to RGBA if needed
            if clothing_img.shape[2] == 3:
                clothing_img = cv2.cvtColor(clothing_img, cv2.COLOR_BGR2BGRA)
            
            # Keep original person image
            original_person = person_img.copy()
            
            # Get body features and measurements
            landmarks, body_mask = self.detect_body_features(person_img)
            if landmarks is None:
                raise ValueError("No body landmarks detected - please ensure full upper body is visible")
            
            # Get image dimensions
            height, width = person_img.shape[:2]
            
            # Get key body points with proper scaling
            left_shoulder = (
                int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width),
                int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height)
            )
            right_shoulder = (
                int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * width),
                int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * height)
            )
            left_hip = (
                int(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * width),
                int(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * height)
            )
            right_hip = (
                int(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x * width),
                int(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y * height)
            )
            
            # Calculate measurements
            shoulder_width = int(np.sqrt((right_shoulder[0] - left_shoulder[0])**2 + 
                                      (right_shoulder[1] - left_shoulder[1])**2))
            
            shoulder_center = (
                (left_shoulder[0] + right_shoulder[0])//2,
                (left_shoulder[1] + right_shoulder[1])//2
            )
            
            hip_center = (
                (left_hip[0] + right_hip[0])//2,
                (left_hip[1] + right_hip[1])//2
            )
            
            torso_height = hip_center[1] - shoulder_center[1]
            
            # Calculate t-shirt dimensions
            target_width = int(shoulder_width * 1.8)
            
            # Use original aspect ratio
            original_aspect = clothing_img.shape[0] / clothing_img.shape[1]
            target_height = int(torso_height * 1.4)
            target_width = int(target_height / original_aspect)
            
            # Resize t-shirt
            resized_clothing = cv2.resize(clothing_img, (target_width, target_height),
                                        interpolation=cv2.INTER_LANCZOS4)
            
            # Position t-shirt
            x_offset = shoulder_center[0] - target_width//2
            y_offset = shoulder_center[1] - int(target_height * 0.1)
            
            # Apply minimal rotation
            body_angle = np.arctan2(right_shoulder[1] - left_shoulder[1],
                                  right_shoulder[0] - left_shoulder[0])
            x_offset += int(np.tan(body_angle) * target_height * 0.03)
            
            # Ensure within bounds
            x_offset = max(0, min(x_offset, person_img.shape[1] - target_width))
            y_offset = max(0, min(y_offset, person_img.shape[0] - target_height))
            
            # Create mask
            mask = resized_clothing[:, :, 3] if resized_clothing.shape[2] == 4 else np.ones(resized_clothing.shape[:2])
            mask = mask / 255.0
            mask = cv2.GaussianBlur(mask, (3, 3), 0)
            
            # Create output image
            result = original_person.copy()
            
            # Calculate valid region for placement
            y1, y2 = y_offset, y_offset + resized_clothing.shape[0]
            x1, x2 = x_offset, x_offset + resized_clothing.shape[1]
            
            # Ensure coordinates are within image bounds
            y1 = max(0, y1)
            y2 = min(result.shape[0], y2)
            x1 = max(0, x1)
            x2 = min(result.shape[1], x2)
            
            # Adjust clothing region to match valid coordinates
            clothing_y1 = 0 if y1 == y_offset else y_offset - y1
            clothing_y2 = resized_clothing.shape[0] if y2 == y_offset + resized_clothing.shape[0] else resized_clothing.shape[0] - (y_offset + resized_clothing.shape[0] - y2)
            clothing_x1 = 0 if x1 == x_offset else x_offset - x1
            clothing_x2 = resized_clothing.shape[1] if x2 == x_offset + resized_clothing.shape[1] else resized_clothing.shape[1] - (x_offset + resized_clothing.shape[1] - x2)
            
            # Get the region of interest
            roi = result[y1:y2, x1:x2]
            
            # Get the clothing region
            clothing_roi = resized_clothing[clothing_y1:clothing_y2, clothing_x1:clothing_x2]
            mask_roi = mask[clothing_y1:clothing_y2, clothing_x1:clothing_x2]
            
            # Apply body mask to clothing mask
            body_mask_roi = body_mask[y1:y2, x1:x2] / 255.0
            mask_roi = mask_roi * body_mask_roi
            
            # Blend the images
            for c in range(3):  # For each color channel
                roi[:, :, c] = roi[:, :, c] * (1 - mask_roi) + clothing_roi[:, :, c] * mask_roi
            
            # Place the blended region back into the result
            result[y1:y2, x1:x2] = roi
            
            # Save result
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            if not cv2.imwrite(output_path, result):
                raise ValueError("Failed to save result image")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] In try_on: {str(e)}")
            return False
