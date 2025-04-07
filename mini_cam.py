import cv2
import numpy as np
import requests
import json
import time
import io

def call_ultralytics_api(image_bytes, api_key="YOUR_API_KEY_HERE"):
    """Call the Ultralytics API for inference."""
    url = "https://predict.ultralytics.com"
    headers = {"x-api-key": api_key}
    
    # Adjust model and parameters as needed
    data = {
        "model": "https://hub.ultralytics.com/models/J5K7gvCLfrGZKOQj64aQ",  # Replace with your model URL
        "imgsz": 640,
        "conf": 0.25,
        "iou": 0.45
    }
    
    # Send request
    response = requests.post(
        url, 
        headers=headers, 
        data=data, 
        files={"file": image_bytes}
    )
    
    # Check for successful response
    response.raise_for_status()
    
    # Return the JSON results
    return response.json()

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
    
    # Calculate Euclidean distance
    distance_euclidean = np.sqrt(distance_x**2 + distance_y**2)

    # Adjust angle if width < height
    if width < height:
        angle += 90

    return (center_x, center_y), (distance_x, distance_y, distance_euclidean), angle

def process_api_results(api_results, frame):
    """Process API results and draw information on the frame."""
    orig_height, orig_width = frame.shape[:2]
    frame_center = (orig_width // 2, orig_height // 2)

    # Font and colors for visualization
    font = cv2.FONT_HERSHEY_SIMPLEX
    colors = {
        0: (0, 255, 0),  # Green for battery
        1: (0, 0, 255)   # Red for C_Battery
    }
    class_names = {
        0: "battery",
        1: "C_Battery"
    }
    
    # Check if results are available
    if 'images' not in api_results or len(api_results['images']) == 0:
        return frame
    
    # Get the first image results
    image_results = api_results['images'][0]
    
    # Process each detection
    if 'results' in image_results:
        for result in image_results['results']:
            # Get class and confidence
            class_idx = result['class'] if 'class' in result else 0
            confidence = result['confidence'] if 'confidence' in result else 1.0
            
            # Draw frame center
            cv2.circle(frame, frame_center, 7, (255, 255, 255), -1)
            cv2.circle(frame, frame_center, 5, (0, 0, 0), -1)
            
            # Process segmentation if available
            if 'segments' in result:
                segments = result['segments']
                
                # Check if segments contains x and y arrays
                if 'x' in segments and 'y' in segments:
                    x_coords = segments['x']
                    y_coords = segments['y']
                    
                    if len(x_coords) == len(y_coords) and len(x_coords) > 0:
                        # Create contour from x, y coordinates
                        pts = np.array(list(zip(x_coords, y_coords)), dtype=np.int32)
                        contour = pts.reshape((-1, 1, 2))
                        
                        # Draw the contour with a thicker line
                        cv2.drawContours(frame, [contour], -1, colors.get(class_idx, (255, 0, 0)), 3)
                        
                        # Fill contour with semi-transparent color
                        overlay = frame.copy()
                        cv2.drawContours(overlay, [contour], -1, colors.get(class_idx, (255, 0, 0)), -1)
                        alpha = 0.3  # Transparency factor
                        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                        
                        # Calculate center, distance from frame center, and angle
                        center, distance, angle = calculate_center_and_angle(contour)
                        
                        if center is not None:
                            # Draw center point
                            cv2.circle(frame, center, 7, (255, 255, 255), -1)
                            cv2.circle(frame, center, 5, colors.get(class_idx, (255, 0, 0)), -1)
                            
                            # Draw line from frame center to object center
                            cv2.line(frame, frame_center, center, colors.get(class_idx, (255, 0, 0)), 2)
                            
                            # Draw distance on line
                            mid_x = (frame_center[0] + center[0]) // 2
                            mid_y = (frame_center[1] + center[1]) // 2
                            
                            # Unpack distance variables
                            distance_x, distance_y, distance_euclidean = distance
                            
                            # Add distance text near the midpoint of the line
                            distance_text = f"{int(distance_euclidean)} px"
                            cv2.putText(frame, distance_text, (mid_x + 5, mid_y - 5), 
                                        font, 0.6, (255, 255, 255), 4)  # White outline
                            cv2.putText(frame, distance_text, (mid_x + 5, mid_y - 5), 
                                        font, 0.6, colors.get(class_idx, (255, 0, 0)), 2)
                            
                            # Draw angle arc
                            radius = 50
                            start_angle = 0  # horizontal
                            end_angle = angle
                            
                            # Draw arc to visualize angle
                            # Convert angles to radians for cv2.ellipse
                            start_angle_rad = 0
                            end_angle_rad = angle * np.pi / 180
                            
                            # Draw angle arc
                            cv2.ellipse(frame, center, (radius, radius), 0, 0, angle, 
                                       colors.get(class_idx, (255, 0, 0)), 2)
                            
                            # Add angle text
                            angle_x = center[0] + int(radius * 0.8 * np.cos(end_angle_rad / 2))
                            angle_y = center[1] + int(radius * 0.8 * np.sin(end_angle_rad / 2))
                            angle_text = f"{angle:.1f}°"
                            
                            cv2.putText(frame, angle_text, (angle_x, angle_y), 
                                       font, 0.6, (255, 255, 255), 4)  # White outline
                            cv2.putText(frame, angle_text, (angle_x, angle_y), 
                                       font, 0.6, colors.get(class_idx, (255, 0, 0)), 2)
                            
                            # Information panel
                            text_lines = [
                                f"Class: {class_names.get(class_idx, 'Unknown')} ({confidence:.2f})",
                                f"Center: ({center[0]}, {center[1]})",
                                f"Distance: {distance_euclidean:.1f} px",
                                f"Offset: dx={distance_x}, dy={distance_y}",
                                f"Angle: {angle:.1f} deg"
                            ]
                            
                            # Calculate panel position (top-right corner)
                            panel_x = orig_width - 300
                            panel_y = 30
                            
                            # Draw semi-transparent background for text
                            text_bg_height = len(text_lines) * 30 + 10
                            overlay = frame.copy()
                            cv2.rectangle(overlay, (panel_x - 10, panel_y - 20), 
                                         (panel_x + 290, panel_y + text_bg_height), 
                                         (0, 0, 0), -1)
                            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                            
                            # Draw text
                            for j, text in enumerate(text_lines):
                                y_pos = panel_y + j * 30
                                cv2.putText(frame, text, (panel_x, y_pos), 
                                           font, 0.7, (255, 255, 255), 2)
                            
                            # Print information to terminal
                            print(f"\n----- Detection Information -----")
                            print(f"Class: {class_names.get(class_idx, 'Unknown')} (Confidence: {confidence:.2f})")
                            print(f"Center: ({center[0]}, {center[1]})")
                            print(f"Distance from center: {distance_euclidean:.1f} pixels")
                            print(f"Offset: dx={distance_x}, dy={distance_y}")
                            print(f"Angle: {angle:.1f} degrees")
                            print(f"--------------------------------\n")
            
            # Process bounding box (only if no segments were found)
            elif 'box' in result:
                box = result['box']
                x1 = box['x1'] if 'x1' in box else 0
                y1 = box['y1'] if 'y1' in box else 0
                x2 = box['x2'] if 'x2' in box else 0
                y2 = box['y2'] if 'y2' in box else 0
                
                # Calculate center
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                center = (center_x, center_y)
                
                # Calculate distance from frame center
                distance_x = center_x - frame_center[0]
                distance_y = center_y - frame_center[1]
                distance_euclidean = np.sqrt(distance_x**2 + distance_y**2)
                distance = (distance_x, distance_y, distance_euclidean)
                
                # Calculate angle - for boxes, we'll use width/height ratio to approximate
                width = x2 - x1
                height = y2 - y1
                angle = 0 if width >= height else 90  # Rough approximation
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), colors.get(class_idx, (255, 0, 0)), 2)
                
                # Draw center point
                cv2.circle(frame, center, 7, (255, 255, 255), -1)
                cv2.circle(frame, center, 5, colors.get(class_idx, (255, 0, 0)), -1)
                
                # Draw line from frame center to object center
                cv2.line(frame, frame_center, center, colors.get(class_idx, (255, 0, 0)), 2)
                
                # Draw distance on line
                mid_x = (frame_center[0] + center[0]) // 2
                mid_y = (frame_center[1] + center[1]) // 2
                
                # Add distance text near the midpoint of the line
                distance_text = f"{int(distance_euclidean)} px"
                cv2.putText(frame, distance_text, (mid_x + 5, mid_y - 5), 
                            font, 0.6, (255, 255, 255), 4)  # White outline
                cv2.putText(frame, distance_text, (mid_x + 5, mid_y - 5), 
                            font, 0.6, colors.get(class_idx, (255, 0, 0)), 2)
                
                # Information panel
                text_lines = [
                    f"Class: {class_names.get(class_idx, 'Unknown')} ({confidence:.2f})",
                    f"Center: ({center[0]}, {center[1]})",
                    f"Distance: {distance_euclidean:.1f} px",
                    f"Offset: dx={distance_x}, dy={distance_y}",
                    f"Angle: {angle} deg (approx)"
                ]
                
                # Calculate panel position (top-right corner)
                panel_x = orig_width - 300
                panel_y = 30
                
                # Draw semi-transparent background for text
                text_bg_height = len(text_lines) * 30 + 10
                overlay = frame.copy()
                cv2.rectangle(overlay, (panel_x - 10, panel_y - 20), 
                             (panel_x + 290, panel_y + text_bg_height), 
                             (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                
                # Draw text
                for j, text in enumerate(text_lines):
                    y_pos = panel_y + j * 30
                    cv2.putText(frame, text, (panel_x, y_pos), 
                               font, 0.7, (255, 255, 255), 2)
                
                # Print information to terminal
                print(f"\n----- Detection Information -----")
                print(f"Class: {class_names.get(class_idx, 'Unknown')} (Confidence: {confidence:.2f})")
                print(f"Center: ({center[0]}, {center[1]})")
                print(f"Distance from center: {distance_euclidean:.1f} pixels")
                print(f"Offset: dx={distance_x}, dy={distance_y}")
                print(f"Angle: {angle} degrees (approximated from box)")
                print(f"--------------------------------\n")
    
    return frame

def main():
    # API key - replace with your actual key
    api_key = "18a92920bf0028361be6437baa030fde66e2b6d873"
    
    # Initialize camera
    cap = cv2.VideoCapture(0)  # Use 0 for default camera, change if needed
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print("Camera opened successfully. Press 'q' to quit.")
    
    # For performance reasons, we'll only call the API every few frames
    frame_count = 0
    api_call_frequency = 5  # Call API every 5 frames
    last_api_results = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't read frame.")
            break
        
        # Call API on selected frames
        if frame_count % api_call_frequency == 0:
            # Convert frame to bytes for API request
            _, img_encoded = cv2.imencode('.jpg', frame)
            img_bytes = io.BytesIO(img_encoded.tobytes())
            
            try:
                # Call API
                api_results = call_ultralytics_api(img_bytes, api_key)
                last_api_results = api_results
                print(f"API call successful. Processing results...")
            except Exception as e:
                print(f"API call failed: {e}")
        
        # Process results and draw on frame
        if last_api_results:
            processed_frame = process_api_results(last_api_results, frame.copy())
        else:
            processed_frame = frame
        
        # Display the resulting frame
        cv2.imshow('Battery Detection (API Version)', processed_frame)
        
        # Increment frame counter
        frame_count += 1
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
