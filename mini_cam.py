import cv2
import torch
import numpy as np
import math
import os
from ultralytics import YOLO
from ultralytics.engine.results import Results


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


def process_results(results, frame):
    """Process detection results and draw information on the frame."""
    orig_height, orig_width = frame.shape[:2]
    frame_center = (orig_width // 2, orig_height // 2)

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

    # Process each detection
    if hasattr(results[0], 'masks') and results[0].masks is not None:
        # Process segmentation masks
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
                    # Draw the contour
                    cv2.drawContours(frame, [largest_contour], -1, colors.get(class_idx, (255, 0, 0)), 2)

                    # Draw center point
                    cv2.circle(frame, center, 5, colors.get(class_idx, (255, 0, 0)), -1)

                    # Draw line from frame center to object center
                    cv2.line(frame, frame_center, center, colors.get(class_idx, (255, 0, 0)), 2)

                    # Put text information
                    text_lines = [
                        f"Class: {class_names.get(class_idx, 'Unknown')} ({confidence:.2f})",
                        f"Center: {center}",
                        f"Dist from center: {distance}",
                        f"Angle: {angle:.1f} deg"
                    ]

                    for j, text in enumerate(text_lines):
                        y_pos = 30 + j * 30
                        cv2.putText(frame, text, (10, y_pos), font, 0.7, colors.get(class_idx, (255, 0, 0)), 2)

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

            # Calculate angle - for boxes, we'll use width/height ratio to approximate
            width = x2 - x1
            height = y2 - y1
            angle = 0 if width >= height else 90  # Rough approximation

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), colors.get(class_idx, (255, 0, 0)), 2)

            # Draw center point
            cv2.circle(frame, center, 5, colors.get(class_idx, (255, 0, 0)), -1)

            # Draw line from frame center to object center
            cv2.line(frame, frame_center, center, colors.get(class_idx, (255, 0, 0)), 2)

            # Put text information
            text_lines = [
                f"Class: {class_names.get(class_idx, 'Unknown')} ({confidence:.2f})",
                f"Center: {center}",
                f"Dist from center: {distance}",
                f"Angle: {angle} deg"
            ]

            for j, text in enumerate(text_lines):
                y_pos = 30 + j * 30
                cv2.putText(frame, text, (10, y_pos), font, 0.7, colors.get(class_idx, (255, 0, 0)), 2)

    return frame


def main():
    # Load model
    model_path = "C:\\Users\\willd\\Downloads\\top_battery.pt"
    model = load_model(model_path)

    # Initialize camera
    cap = cv2.VideoCapture(0)  # Use 0 for default camera, change if needed

    # Set resolution (adjust as needed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Camera opened successfully. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't read frame.")
            break

        # Run inference with YOLO model
        results = model(frame)

        # Process results and draw information on the frame
        processed_frame = process_results(results, frame.copy())

        # Display the resulting frame
        cv2.imshow('Battery Detection', processed_frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()