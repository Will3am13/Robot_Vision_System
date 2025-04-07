import cv2
import torch
import numpy as np
import math
import time
from ultralytics import YOLO


def load_model(model_path):
    """Load the YOLO segmentation model."""
    # Add safe globals to allow loading the Ultralytics model
    torch.serialization.add_safe_globals(['ultralytics.nn.tasks.SegmentationModel'])

    # Load using YOLO which handles the Ultralytics models properly
    try:
        model = YOLO(model_path)
        print(f"Model loaded successfully: {model}")
        return model
    except Exception as e:
        print(f"Error loading model with YOLO: {e}")

        # Fallback: try loading with torch.load and weights_only=False
        print("Trying fallback method...")
        try:
            model = torch.load(model_path, weights_only=False)
            print(f"Model loaded successfully with fallback method: {model}")
            return model
        except Exception as e2:
            print(f"Error with fallback method: {e2}")
            raise RuntimeError(f"Could not load model: {e2}")


def calculate_center_and_angle(contour):
    """Calculate center coordinates, distance from frame center, and angle of the object."""
    if contour is None or len(contour) < 5:  # Need at least 5 points for ellipse fitting
        return None, None, None

    # Get rotated rectangle (minimum area rectangle)
    rect = cv2.minAreaRect(contour)
    center, (width, height), angle = rect

    # Convert center to integers
    center_x, center_y = int(center[0]), int(center[1])

    # Calculate distance from frame center
    frame_width, frame_height = 640, 480  # Default, will be updated with actual frame size
    frame_center_x, frame_center_y = frame_width // 2, frame_height // 2
    distance_x = center_x - frame_center_x
    distance_y = center_y - frame_center_y

    # Adjust angle if width < height
    if width < height:
        angle += 90

    return (center_x, center_y), (distance_x, distance_y), angle


def get_closest_battery_to_center(results, frame):
    """Find the battery closest to the center of the frame."""
    orig_height, orig_width = frame.shape[:2]
    frame_center = (orig_width // 2, orig_height // 2)

    closest_battery = None
    min_distance = float('inf')
    battery_info = None

    # Process each detection with mask
    if hasattr(results[0], 'masks') and results[0].masks is not None:
        for i, mask in enumerate(results[0].masks):
            # Get the segmentation contours
            if hasattr(mask, 'xy'):
                contours = [np.array(mask.xy[0], dtype=np.int32)]
            else:
                # Convert mask tensor to numpy array
                mask_array = mask.data.cpu().numpy() if hasattr(mask, 'data') else mask.cpu().numpy()
                mask_array = (mask_array * 255).astype(np.uint8)

                # Find contours from mask
                contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Get the largest contour
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)

                # Get class information if available
                class_idx = None
                if hasattr(results[0], 'boxes') and i < len(results[0].boxes):
                    class_idx = int(results[0].boxes[i].cls.item())
                    confidence = results[0].boxes[i].conf.item()
                else:
                    class_idx = 0  # Default to 'battery'
                    confidence = 1.0

                # Calculate center, distance from frame center, and angle
                center, distance, angle = calculate_center_and_angle(largest_contour)

                if center is not None:
                    # Calculate Euclidean distance to frame center
                    euclidean_distance = math.sqrt(distance[0] ** 2 + distance[1] ** 2)

                    if euclidean_distance < min_distance:
                        min_distance = euclidean_distance
                        closest_battery = largest_contour
                        battery_info = {
                            'center': center,
                            'distance': distance,
                            'angle': angle,
                            'class_idx': class_idx,
                            'confidence': confidence
                        }

    # If no masks found, try processing boxes only
    elif hasattr(results[0], 'boxes') and results[0].boxes is not None and len(results[0].boxes) > 0:
        for i, box in enumerate(results[0].boxes):
            # Get bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Get class information
            class_idx = int(box.cls.item())
            confidence = box.conf.item()

            # Calculate center
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            center = (center_x, center_y)

            # Calculate distance from frame center
            distance_x = center_x - frame_center[0]
            distance_y = center_y - frame_center[1]
            distance = (distance_x, distance_y)

            # Calculate Euclidean distance to frame center
            euclidean_distance = math.sqrt(distance_x ** 2 + distance_y ** 2)

            # Calculate angle - for boxes, we'll use width/height ratio to approximate
            width = x2 - x1
            height = y2 - y1
            angle = 0 if width >= height else 90  # Rough approximation

            if euclidean_distance < min_distance:
                min_distance = euclidean_distance
                closest_battery = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                                           dtype=np.int32)  # Create contour from box
                battery_info = {
                    'center': center,
                    'distance': distance,
                    'angle': angle,
                    'class_idx': class_idx,
                    'confidence': confidence
                }

    return closest_battery, battery_info


def process_frame_visualization(frame, battery_contour, battery_info):
    """Process and visualize the detection results on the frame."""
    if battery_contour is None or battery_info is None:
        return frame

    # Make a copy of the frame to avoid modifying the original
    viz_frame = frame.copy()

    # Font and colors for visualization
    font = cv2.FONT_HERSHEY_SIMPLEX
    colors = {
        0: (0, 255, 0),  # Green for battery
        1: (0, 0, 255)  # Red for C_Battery
    }
    class_names = {
        0: "battery",
        1: "C_Battery"
    }

    # Frame center
    height, width = frame.shape[:2]
    frame_center = (width // 2, height // 2)

    # Draw the contour
    color = colors.get(battery_info['class_idx'], (255, 0, 0))
    cv2.drawContours(viz_frame, [battery_contour], -1, color, 2)

    # Draw center point
    cv2.circle(viz_frame, battery_info['center'], 5, color, -1)

    # Draw frame center
    cv2.circle(viz_frame, frame_center, 5, (255, 255, 255), -1)

    # Draw line from frame center to object center
    cv2.line(viz_frame, frame_center, battery_info['center'], color, 2)

    # Put text information
    text_lines = [
        f"Class: {class_names.get(battery_info['class_idx'], 'Unknown')} ({battery_info['confidence']:.2f})",
        f"Center: {battery_info['center']}",
        f"Offset from center: {battery_info['distance']}",
        f"Angle: {battery_info['angle']:.1f} deg"
    ]

    for i, text in enumerate(text_lines):
        y_pos = 30 + i * 30
        cv2.putText(viz_frame, text, (10, y_pos), font, 0.7, color, 2)

    return viz_frame


class MiniCamController:
    def __init__(self, model_path, camera_id=1, x_pixel_to_mm=0.1, y_pixel_to_mm=0.1, angle_to_robot=1.0):
        """
        Initialize the mini camera controller.

        Args:
            model_path (str): Path to the YOLO model for battery detection.
            camera_id (int): Camera ID for the mini camera (default: 1).
            x_pixel_to_mm (float): Conversion factor from x-axis pixels to mm for robot coordinates.
            y_pixel_to_mm (float): Conversion factor from y-axis pixels to mm for robot coordinates.
            angle_to_robot (float): Conversion factor from detected angle to robot angle.
        """
        self.model_path = model_path
        self.camera_id = camera_id
        self.x_pixel_to_mm = x_pixel_to_mm
        self.y_pixel_to_mm = y_pixel_to_mm
        self.angle_to_robot = angle_to_robot
        self.model = None
        self.cap = None

        # Initialize the model
        self.model = load_model(model_path)

    def start_camera(self):
        """Start the mini camera."""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)

            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            if not self.cap.isOpened():
                print("Error: Could not open mini camera.")
                return False

            print("Mini camera opened successfully.")
            return True
        except Exception as e:
            print(f"Error starting mini camera: {e}")
            return False

    def stop_camera(self):
        """Stop the mini camera."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("Mini camera stopped.")

    def calculate_adjustments(self, pixel_offset_x, pixel_offset_y, angle):
        """
        Calculate robot adjustments based on pixel offsets and angle.

        Args:
            pixel_offset_x (int): X-axis offset in pixels from frame center.
            pixel_offset_y (int): Y-axis offset in pixels from frame center.
            angle (float): Detected angle of the battery.

        Returns:
            tuple: (x_offset_mm, y_offset_mm, angle_adjustment)
        """
        # Convert pixel offsets to millimeters
        # Note: Depending on your setup, you might need to invert signs
        x_offset_mm = -pixel_offset_x * self.x_pixel_to_mm  # Invert X since camera and robot may have different coordinate systems
        y_offset_mm = -pixel_offset_y * self.y_pixel_to_mm  # Invert Y since camera and robot may have different coordinate systems

        # Calculate angle adjustment
        # Target angle for the robot is 90° (perpendicular)
        angle_difference = 90 - angle
        angle_adjustment = angle_difference * self.angle_to_robot

        return x_offset_mm, y_offset_mm, angle_adjustment

    def fine_tune_position(self, mc, current_xyz, threshold_distance=10, threshold_angle=5, max_attempts=3,
                           visualization=False):
        """
        Fine-tune the robot position using visual feedback from the mini camera.

        Args:
            mc: MyCobot280 instance
            current_xyz: Current XYZ coordinates of the robot
            threshold_distance (int): Maximum acceptable distance in pixels from center
            threshold_angle (float): Maximum acceptable angle difference in degrees
            max_attempts (int): Maximum number of adjustment attempts
            visualization (bool): Whether to show visualization window

        Returns:
            tuple: (success, adjusted_xyz, adjusted_angle)
        """
        print("Starting mini camera for position adjustment...")
        if not self.start_camera():
            return False, current_xyz, None

        attempt = 0
        success = False
        adjusted_xyz = list(current_xyz)
        adjusted_angle = None

        while attempt < max_attempts and not success:
            attempt += 1
            print(f"\nFine-tuning attempt {attempt}/{max_attempts}")

            # Capture frame from mini camera
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Couldn't read frame from mini camera.")
                break

            # Run inference with YOLO model
            results = self.model(frame)

            # Get the battery closest to the center
            battery_contour, battery_info = get_closest_battery_to_center(results, frame)

            if battery_contour is None or battery_info is None:
                print("No battery detected in mini camera view.")
                if visualization:
                    cv2.imshow('Mini Camera View', frame)
                    cv2.waitKey(1000)
                continue

            # Process frame for visualization
            if visualization:
                viz_frame = process_frame_visualization(frame, battery_contour, battery_info)
                cv2.imshow('Mini Camera View', viz_frame)
                cv2.waitKey(1)

            # Get pixel offsets and angle
            pixel_offset_x, pixel_offset_y = battery_info['distance']
            angle = battery_info['angle']

            # Calculate required adjustments
            x_offset_mm, y_offset_mm, angle_adjustment = self.calculate_adjustments(
                pixel_offset_x, pixel_offset_y, angle
            )

            print(f"Detected position: Pixel offset X: {pixel_offset_x}, Y: {pixel_offset_y}, Angle: {angle:.1f}°")
            print(
                f"Calculated adjustments: X: {x_offset_mm:.2f}mm, Y: {y_offset_mm:.2f}mm, Angle: {angle_adjustment:.2f}°")

            # Check if already within threshold
            distance_from_center = math.sqrt(pixel_offset_x ** 2 + pixel_offset_y ** 2)
            angle_difference = abs(90 - angle)

            if distance_from_center <= threshold_distance and angle_difference <= threshold_angle:
                print("Already within acceptable thresholds. No adjustment needed.")
                success = True
                adjusted_angle = angle
                break

            # Apply adjustments to current coordinates
            adjusted_xyz[0] += x_offset_mm
            adjusted_xyz[1] += y_offset_mm

            # Calculate final angle for the robot's end effector
            # Assuming the last angle in PICK_ORIENTATION is the one to adjust
            current_orientation = 45  # Default from the PICK_ORIENTATION
            adjusted_angle = current_orientation + angle_adjustment

            print(f"Moving robot to adjusted position: {adjusted_xyz}")

            # Move the robot to the adjusted position
            hover_coords = adjusted_xyz + [180, 0, adjusted_angle]  # Assuming orientation format

            # Check if coordinates are valid before sending to robot
            if not is_valid_coord(hover_coords):
                print("Calculated position is out of robot's working range. Trying again...")
                # Revert to original position
                adjusted_xyz = list(current_xyz)
                continue

            # Send the robot to the adjusted position
            mc.send_coords(hover_coords, 20, 1)
            time.sleep(2)

            # Capture another frame to verify position
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Couldn't read verification frame from mini camera.")
                break

            # Run inference again
            results = self.model(frame)

            # Get the battery position for verification
            battery_contour, battery_info = get_closest_battery_to_center(results, frame)

            if battery_contour is None or battery_info is None:
                print("No battery detected during verification.")
                continue

            # Check if now within threshold
            pixel_offset_x, pixel_offset_y = battery_info['distance']
            angle = battery_info['angle']

            distance_from_center = math.sqrt(pixel_offset_x ** 2 + pixel_offset_y ** 2)
            angle_difference = abs(90 - angle)

            if visualization:
                viz_frame = process_frame_visualization(frame, battery_contour, battery_info)
                cv2.putText(viz_frame,
                            f"Verification - Distance: {distance_from_center:.1f}px, Angle diff: {angle_difference:.1f}°",
                            (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow('Mini Camera View', viz_frame)
                cv2.waitKey(1000)

            if distance_from_center <= threshold_distance and angle_difference <= threshold_angle:
                print(f"Position successfully adjusted within thresholds. "
                      f"Final distance: {distance_from_center:.1f}px, Angle difference: {angle_difference:.1f}°")
                success = True
                break
            else:
                print(f"Position still needs adjustment. "
                      f"Distance: {distance_from_center:.1f}px (threshold: {threshold_distance}px), "
                      f"Angle diff: {angle_difference:.1f}° (threshold: {threshold_angle}°)")

        # Clean up
        if visualization:
            cv2.destroyAllWindows()

        self.stop_camera()

        if success:
            return True, adjusted_xyz, adjusted_angle
        else:
            print("Failed to adjust position within the maximum number of attempts.")
            return False, current_xyz, None


def is_valid_coord(coord):
    """Check if coordinate values are within the safe working range."""
    x, y, z, rx, ry, rz = coord
    if not (-281.45 <= x <= 281.45):  # x range
        print(f"Out of range in x: {x}")
        return False
    if not (-281.45 <= y <= 281.45):  # y range
        print(f"Out of range in y: {y}")
        return False
    if not (-70 <= z <= 412.67):  # z range
        print(f"Out of range in z: {z}")
        return False
    if not (-180 <= rx <= 180):  # rx range
        print(f"Out of range in roll: {rx}")
        return False
    if not (-180 <= ry <= 180):  # ry range
        print(f"Out of range in pitch: {ry}")
        return False
    if not (-180 <= rz <= 180):  # rz range
        print(f"Out of range in yaw: {rz}")
        return False
    return True