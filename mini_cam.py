import cv2
import numpy as np
import math
import subprocess
import time
import os
import threading


def find_centroid(mask):
    """Find the centroid of a binary mask"""
    # Calculate moments of the binary image
    M = cv2.moments(mask)

    # Check to prevent division by zero
    if M["m00"] != 0:
        # Calculate centroid
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)
    else:
        # Return None if no contour found
        return None


def find_corners(mask, max_corners=4):
    """Find corners in the mask using Shi-Tomasi corner detection"""
    # Find contours first
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return []

    # Get the largest contour (assuming it's our object of interest)
    largest_contour = max(contours, key=cv2.contourArea)

    # Try to approximate the contour with fewer points
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # If we get a good polygon approximation with 4 corners, use it
    if len(approx) >= 3 and len(approx) <= 6:
        # Return the corner points (reshape to get rid of the extra dimension)
        return [tuple(pt[0]) for pt in approx]

    # Otherwise, use Shi-Tomasi to find corners
    # Create a mask with only the largest contour
    contour_mask = np.zeros_like(mask)
    cv2.drawContours(contour_mask, [largest_contour], 0, 255, -1)

    # Apply corner detection
    corners = cv2.goodFeaturesToTrack(contour_mask, maxCorners=max_corners,
                                      qualityLevel=0.01, minDistance=10)

    if corners is not None:
        return [tuple(map(int, corner[0])) for corner in corners]
    else:
        return []


def calculate_object_angle(corners, centroid):
    """Calculate the orientation angle of an object based on its corners and centroid"""
    if not corners or len(corners) < 2 or not centroid:
        return None

    # Find the principal axis using PCA-like approach
    # First, center the points by subtracting the centroid
    centered_corners = [(x - centroid[0], y - centroid[1]) for x, y in corners]

    # Calculate the covariance matrix
    n = len(centered_corners)
    sum_xx = sum(x * x for x, y in centered_corners)
    sum_yy = sum(y * y for x, y in centered_corners)
    sum_xy = sum(x * y for x, y in centered_corners)

    # Avoid division by zero
    if n == 0:
        return None

    cov_xx = sum_xx / n
    cov_yy = sum_yy / n
    cov_xy = sum_xy / n

    # Calculate the angle using the covariance matrix
    if abs(cov_xx - cov_yy) < 1e-10 and abs(cov_xy) < 1e-10:
        # Object is approximately circular
        return 0

    # Calculate the angle of the principal axis
    theta = 0.5 * math.atan2(2 * cov_xy, cov_xx - cov_yy)

    # Convert to degrees
    angle_degrees = math.degrees(theta)

    return angle_degrees


def calculate_distance_from_center(centroid, frame_center):
    """Calculate the pixel distance from the frame center to the object centroid"""
    if not centroid:
        return None, None, None

    # Calculate the x and y differences
    x_diff = centroid[0] - frame_center[0]
    y_diff = centroid[1] - frame_center[1]

    # Calculate the Euclidean distance
    distance = math.sqrt(x_diff ** 2 + y_diff ** 2)

    return x_diff, y_diff, distance


def toggle_usb_power(turn_on=True):
    """Simple function to toggle USB power using uhubctl or basic commands"""
    try:
        # First, try uhubctl if it's available (most reliable option)
        try:
            action = "on" if turn_on else "off"
            subprocess.run(['uhubctl', '-a', action], check=True)
            print(f"USB power turned {action.upper()} using uhubctl")
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            # If uhubctl is not available, try alternative methods
            pass
        
    except Exception as e:
        print(f"Error toggling USB power: {e}")
        return False


def power_cycle_camera():
    """Power cycle the camera - turn off, wait, turn back on"""
    print("Starting camera power cycle...")
    
    # Turn power OFF
    toggle_usb_power(False)
    print("USB power OFF - waiting 3 seconds...")
    
    # Wait for 3 seconds
    time.sleep(3)
    
    # Turn power back ON
    toggle_usb_power(True)
    print("USB power ON - waiting for camera initialization...")
    
    # Wait for camera to initialize
    time.sleep(5)
    
    print("Power cycle complete")


def initialize_camera():
    """Initialize the camera with a reliable approach"""
    camera = None
    attempt = 0
    max_attempts = 3
    
    while attempt < max_attempts:
        print(f"Camera initialization attempt {attempt+1}/{max_attempts}")
        
        # Try to open the camera
        camera = cv2.VideoCapture(0)
        
        # Check if camera is working
        if camera.isOpened():
            # Try to read a frame to confirm it's actually working
            ret, _ = camera.read()
            
            if ret:
                print("Camera initialized successfully!")
                return camera
            else:
                print("Camera opened but couldn't read frame")
                camera.release()
        
        # If we get here, the camera failed to initialize
        print("Camera initialization failed")
        
        # If this isn't the last attempt, try power cycling
        if attempt < max_attempts - 1:
            print("Trying power cycle...")
            power_cycle_camera()
        
        attempt += 1
    
    # If we've exhausted all attempts, return None or a dummy camera
    print("Camera initialization failed after multiple attempts")
    
    # Return a non-functional but initialized camera object
    return cv2.VideoCapture(0)


def main():
    # Initialize the camera with our robust approach
    print("Starting camera initialization...")
    cap = initialize_camera()

    # Initial threshold value (0-255)
    threshold = 120
    
    # Power cycling state
    power_cycling = False

    print("Controls:")
    print("  + : Increase threshold")
    print("  - : Decrease threshold")
    print("  r : Reset threshold to 120")
    print("  p : Power cycle camera (OFF for 3 seconds, then back ON)")
    print("  q : Quit the program")

    # Create a window with a trackbar for the power button
    cv2.namedWindow('Controls')
    # Create a button-like trackbar (it's a bit of a hack since OpenCV doesn't have buttons)
    cv2.createTrackbar('Power Cycle Camera', 'Controls', 0, 1, lambda x: None)

    # Last power cycle time for rate limiting
    last_cycle_time = 0
    min_cycle_interval = 8  # Minimum seconds between cycles

    while True:
        # Create a blank control panel
        control_panel = np.ones((150, 400, 3), dtype=np.uint8) * 240  # Light gray background
        
        # Check power cycle button
        button_state = cv2.getTrackbarPos('Power Cycle Camera', 'Controls')
        current_time = time.time()
        
        # Handle power cycle button press (only if not already cycling)
        if button_state == 1 and not power_cycling and (current_time - last_cycle_time) > min_cycle_interval:
            # Reset the button
            cv2.setTrackbarPos('Power Cycle Camera', 'Controls', 0)
            
            # Start power cycling
            power_cycling = True
            last_cycle_time = current_time
            
            # Run power cycle in a separate thread
            def power_cycle_thread():
                nonlocal power_cycling, cap
                
                # Do the power cycle
                power_cycle_camera()
                
                # Release old camera
                cap.release()
                
                # Create a new camera instance
                cap = cv2.VideoCapture(0)
                
                # Done cycling
                power_cycling = False
            
            # Start the thread
            cycle_thread = threading.Thread(target=power_cycle_thread)
            cycle_thread.daemon = True
            cycle_thread.start()
        
        # Update the control panel based on state
        if power_cycling:
            cv2.putText(control_panel, "Status: POWER CYCLING", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(control_panel, "Please wait...", 
                        (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        else:
            status = "READY" if cap.isOpened() else "NO CAMERA"
            status_color = (0, 128, 0) if cap.isOpened() else (0, 0, 255)
            
            cv2.putText(control_panel, f"Status: {status}", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(control_panel, f"Threshold: {threshold}", 
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(control_panel, "Press 'p' to power cycle, 'q' to quit", 
                        (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Show the control panel
        cv2.imshow('Controls', control_panel)
        
        # Only try to capture a frame if not power cycling
        if not power_cycling and cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to capture image from camera.")
                # Wait a bit before next attempt
                time.sleep(0.5)
                continue

            # Make a copy of the original frame for display
            display_frame = frame.copy()

            # Get frame dimensions
            height, width = frame.shape[:2]
            frame_center = (width // 2, height // 2)

            # Draw a crosshair at the center of the frame
            cv2.drawMarker(display_frame, frame_center, (255, 0, 255), markerType=cv2.MARKER_CROSS,
                        markerSize=20, thickness=2)

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Threshold the grayscale image (white background becomes black)
            # Values brighter than threshold become black (0), darker become white (255)
            _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw all contours on the display frame
            cv2.drawContours(display_frame, contours, -1, (0, 255, 0), 2)

            # Only process if contours are found
            if contours:
                # Find centroid
                centroid = find_centroid(mask)

                # Find corners
                corners = find_corners(mask)

                if centroid:
                    # Calculate distance from center
                    x_diff, y_diff, distance = calculate_distance_from_center(centroid, frame_center)

                    # Calculate object angle
                    angle = calculate_object_angle(corners, centroid)

                    # Draw centroid as a red circle
                    cv2.circle(display_frame, centroid, 5, (0, 0, 255), -1)

                    # Draw a line from center to centroid
                    cv2.line(display_frame, frame_center, centroid, (255, 0, 255), 2)

                    # Display centroid coordinates
                    cv2.putText(display_frame, f"Centroid: {centroid}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    # Display distance information
                    cv2.putText(display_frame, f"Center to object: {distance:.1f} pixels", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                    cv2.putText(display_frame, f"X-diff: {x_diff} px, Y-diff: {y_diff} px", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                    # Display angle if available
                    if angle is not None:
                        cv2.putText(display_frame, f"Angle: {angle:.1f} degrees", (10, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

                        # Draw the principal axis line
                        axis_length = 50  # Length of the axis line to draw
                        radian_angle = math.radians(angle)
                        end_x = int(centroid[0] + axis_length * math.cos(radian_angle))
                        end_y = int(centroid[1] + axis_length * math.sin(radian_angle))
                        cv2.line(display_frame, centroid, (end_x, end_y), (255, 165, 0), 2)

                # Draw corners as blue circles
                for i, corner in enumerate(corners):
                    cv2.circle(display_frame, corner, 5, (255, 0, 0), -1)
                    cv2.putText(display_frame, str(i), (corner[0] + 10, corner[1] + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Display the threshold value on the frame
            cv2.putText(display_frame, f"Threshold: {threshold}", (10, display_frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Display the resulting frames
            cv2.imshow('Original', frame)
            cv2.imshow('Processed', display_frame)
            cv2.imshow('Mask', mask)
        else:
            # Create a blank frame when camera is not available
            no_signal = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Add message
            if power_cycling:
                cv2.putText(no_signal, "POWER CYCLING...", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(no_signal, "Please wait for camera to initialize", (120, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
            else:
                cv2.putText(no_signal, "CAMERA UNAVAILABLE", (160, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(no_signal, "Press 'p' to power cycle the camera", (120, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
            
            # Display the blank frames
            cv2.imshow('Original', no_signal)
            cv2.imshow('Processed', no_signal)
            cv2.imshow('Mask', no_signal)

        # Wait for a key press and check which key was pressed
        key = cv2.waitKey(1) & 0xFF

        # Adjust threshold based on key press
        if key == ord('+') or key == ord('='):
            threshold = min(threshold + 5, 255)
        elif key == ord('-') or key == ord('_'):
            threshold = max(threshold - 5, 0)
        elif key == ord('r'):
            threshold = 120
        elif key == ord('p') and not power_cycling and (current_time - last_cycle_time) > min_cycle_interval:
            # Initiate power cycle (same logic as button press)
            power_cycling = True
            last_cycle_time = current_time
            
            # Run power cycle in a separate thread
            def power_cycle_thread():
                nonlocal power_cycling, cap
                
                # Do the power cycle
                power_cycle_camera()
                
                # Release old camera
                cap.release()
                
                # Create a new camera instance
                cap = cv2.VideoCapture(0)
                
                # Done cycling
                power_cycling = False
            
            # Start the thread
            cycle_thread = threading.Thread(target=power_cycle_thread)
            cycle_thread.daemon = True
            cycle_thread.start()
        elif key == ord('q'):
            break

    # When everything is done, release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
