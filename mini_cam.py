import cv2
import numpy as np
import requests
import json
import time
import base64
import io
from PIL import Image

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

    # Adjust angle if width < height
    if width < height:
        angle += 90

    return (center_x, center_y), (distance_x, distance_y), angle

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
    if not api_results.get('images') or len(api_results['images']) == 0:
        return frame
    
    # Get the first image results
    image_results = api_results['images'][0]
    
    # Process each detection
    for result in image_results.get('results', []):
        class_idx = result.get('class', 0)
        confidence = result.get('confidence', 1.0)
        
        # Process bounding box
        box = result.get('box')
        if box:
            x1, y1, x2, y2 = box.get('x1', 0), box.get('y1', 0), box.get('x2', 0), box.get('y2', 0)
            
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
        
        # Process segmentation if available
        segments = result.get('segments')
        if segments and segments.get('x') and segments.get('y'):
            # Create contour from x, y coordinates
            pts = np.array(list(zip(segments['x'], segments['y'])), dtype=np.int32)
            contour = pts.reshape((-1, 1, 2))
            
            # Draw the contour
            cv2.drawContours(frame, [contour], -1, colors.get(class_idx, (255, 0, 0)), 2)
            
            # Calculate center, distance from frame center, and angle
            center, distance, angle = calculate_center_and_angle(contour)
            
            if center is not None:
                # Draw center point
                cv2.circle(frame, center, 5, colors.get(class_idx, (255, 0, 0)), -1)
                
                # Draw line from frame center to object center
                cv2.line(frame, frame_center, center, colors.get(class_idx, (255, 0, 0)), 2)
                
                # We've already drawn text if we had a bounding box, so only draw if we didn't
                if not box:
                    text_lines = [
                        f"Class: {class_names.get(class_idx, 'Unknown')} ({confidence:.2f})",
                        f"Center: {center}",
                        f"Dist from center: {distance}",
                        f"Angle: {angle:.1f} deg"
                    ]
                    
                    for j, text in enumerate(text_lines):
                        y_pos = 30 + j * 30
                        cv2.putText(frame, text, (10, y_pos), font, 0.7, colors.get(class_idx, (255, 0, 0)), 2)
    
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
