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


def get_camera_usb_info():
    """Get information about USB cameras connected to the system"""
    try:
        # Run lsusb command to list USB devices
        result = subprocess.run(['lsusb'], stdout=subprocess.PIPE, text=True)
        usb_devices = result.stdout.strip().split('\n')
        
        # Filter for devices that might be cameras
        camera_devices = []
        for device in usb_devices:
            # Common camera-related keywords
            if any(keyword in device.lower() for keyword in ['cam', 'webcam', 'video', 'logitech', 'microsoft', 'uvc']):
                camera_devices.append(device)
        
        # If we found any potential cameras, return the first one's bus and device IDs
        if camera_devices:
            # Extract bus and device IDs from the first result (e.g., "Bus 001 Device 005: ID 046d:0825 Logitech")
            parts = camera_devices[0].split()
            if len(parts) >= 4:
                bus = parts[1]
                device = parts[3].rstrip(':')
                return {
                    'bus': bus,
                    'device': device,
                    'description': camera_devices[0]
                }
        
        # If we couldn't identify a camera device specifically
        return None
    except Exception as e:
        print(f"Error getting camera USB info: {e}")
        return None


def find_usb_hub_path():
    """Find the USB hub path for the camera"""
    try:
        # Get camera USB info
        camera_info = get_camera_usb_info()
        if not camera_info:
            print("Could not identify camera USB device")
            return None
            
        # Look for the camera in the sys filesystem
        result = subprocess.run(['find', '/sys/bus/usb/devices/', '-name', 'video*'], 
                               stdout=subprocess.PIPE, text=True)
        video_paths = result.stdout.strip().split('\n')
        
        # Try to find a path containing both "usb" and "hub"
        for path in video_paths:
            if not path:  # Skip empty lines
                continue
                
            # Get the device path by navigating up the directory tree
            device_path = os.path.dirname(path)
            
            # Check if this is the path we're looking for
            if os.path.exists(f"{device_path}/busnum") and os.path.exists(f"{device_path}/devnum"):
                try:
                    with open(f"{device_path}/busnum", 'r') as f:
                        busnum = f.read().strip()
                    with open(f"{device_path}/devnum", 'r') as f:
                        devnum = f.read().strip()
                        
                    # If this matches our camera bus and device, return the parent hub path
                    if busnum == camera_info['bus'] and devnum == camera_info['device']:
                        # Navigate upwards to find a hub
                        current_path = device_path
                        for _ in range(3):  # Don't go too many levels up
                            current_path = os.path.dirname(current_path)
                            if "hub" in current_path:
                                return current_path
                except Exception as e:
                    print(f"Error checking device path {device_path}: {e}")
        
        # If we couldn't find a specific hub path, try to use direct USB control
        return f"/sys/bus/usb/devices/{camera_info['bus']}-{camera_info['device']}"
    except Exception as e:
        print(f"Error finding USB hub path: {e}")
        return None


def toggle_usb_power_method1(hub_path, turn_on=True):
    """Toggle power to a USB port using sysfs (Method 1)"""
    if not hub_path:
        print("No USB hub path specified")
        return False
        
    try:
        # Some hubs support power control via authorized attribute
        authorized_path = f"{hub_path}/authorized"
        if os.path.exists(authorized_path):
            with open(authorized_path, 'w') as f:
                f.write('1' if turn_on else '0')
            print(f"USB power {'ON' if turn_on else 'OFF'} using authorized attribute")
            return True
            
        # If authorized doesn't exist, try using the remove attribute
        if not turn_on:  # We can only turn off with this method
            remove_path = f"{hub_path}/remove"
            if os.path.exists(remove_path):
                with open(remove_path, 'w') as f:
                    f.write('1')
                print("USB device removed")
                return True
                
        # If we're turning on and got here, we can't use this method directly
        if turn_on:
            # Try to rescan the USB bus
            usb_devices_path = '/sys/bus/usb/devices'
            for root, dirs, _ in os.walk(usb_devices_path):
                for directory in dirs:
                    if directory.startswith('usb'):
                        scan_path = os.path.join(root, directory, 'scan')
                        if os.path.exists(scan_path):
                            with open(scan_path, 'w') as f:
                                f.write('1')
                            print("Rescanned USB bus")
                            return True
                            
        return False
    except Exception as e:
        print(f"Error toggling USB power (Method 1): {e}")
        return False


def toggle_usb_power_method2(turn_on=True):
    """Toggle power to USB devices using uhubctl or system commands (Method 2)"""
    try:
        if turn_on:
            # Try to use uhubctl if available
            try:
                subprocess.run(['uhubctl', '-a', 'on'], check=True)
                print("Turned ON USB power using uhubctl")
                return True
            except (subprocess.SubprocessError, FileNotFoundError):
                # If uhubctl fails or is not installed, try system commands
                subprocess.run(['sudo', 'sh', '-c', 'echo "1-1" > /sys/bus/usb/drivers/usb/bind'], check=False)
                print("Attempted to bind USB device")
                
                # Force a rescan of USB devices
                try:
                    subprocess.run(['sudo', 'sh', '-c', 'echo "1" > /sys/bus/usb/drivers/usb/usb1/authorized'], check=False)
                    print("Forced USB device reauthorization")
                except:
                    pass
                return True
        else:
            # Try to use uhubctl if available
            try:
                subprocess.run(['uhubctl', '-a', 'off'], check=True)
                print("Turned OFF USB power using uhubctl")
                return True
            except (subprocess.SubprocessError, FileNotFoundError):
                # If uhubctl fails or is not installed, try system commands
                subprocess.run(['sudo', 'sh', '-c', 'echo "1-1" > /sys/bus/usb/drivers/usb/unbind'], check=False)
                print("Attempted to unbind USB device")
                return True
    except Exception as e:
        print(f"Error toggling USB power (Method 2): {e}")
    return False


def toggle_usb_power(turn_on=True):
    """Toggle power to the USB camera using available methods"""
    success = False
    
    # Try Method 1: Direct sysfs control
    hub_path = find_usb_hub_path()
    if hub_path and toggle_usb_power_method1(hub_path, turn_on):
        success = True
    
    # Try Method 2: System commands if Method 1 failed
    if not success and toggle_usb_power_method2(turn_on):
        success = True
    
    # If turning on, run additional steps to help with device detection
    if turn_on and success:
        # Force kernel to rescan USB devices
        try:
            # Try using usb-devices to trigger a rescan
            subprocess.run(['usb-devices'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        except:
            pass
            
        # Try to reset USB subsystem
        try:
            subprocess.run(['sudo', 'rmmod', 'uvcvideo'], check=False, stderr=subprocess.PIPE)
            time.sleep(0.5)
            subprocess.run(['sudo', 'modprobe', 'uvcvideo'], check=False, stderr=subprocess.PIPE)
            print("Reset UVC video driver")
        except:
            pass
    
    # If all methods failed
    if not success:
        print("Warning: Could not control USB power. USB power toggle functionality may not work.")
    
    return success


def ensure_camera_works(max_attempts=5, delay=1.0):
    """Try to make sure the camera is working by attempting to open it multiple times"""
    print("Testing camera connection...")
    
    for attempt in range(max_attempts):
        try:
            # Try to open the camera
            test_cap = cv2.VideoCapture(0)
            
            # Check if it opened successfully
            if test_cap.isOpened():
                # Try to read a frame to really confirm it's working
                ret, _ = test_cap.read()
                
                # Close the test capture
                test_cap.release()
                
                if ret:
                    print(f"Camera is working! (Attempt {attempt+1}/{max_attempts})")
                    return True
                else:
                    print(f"Camera opened but couldn't read frame (Attempt {attempt+1}/{max_attempts})")
            else:
                print(f"Failed to open camera (Attempt {attempt+1}/{max_attempts})")
                
            # Wait before trying again
            time.sleep(delay)
        except Exception as e:
            print(f"Error testing camera: {e}")
            time.sleep(delay)
    
    print(f"Camera still not working after {max_attempts} attempts")
    return False


def reset_camera_device(cam_device_num=0):
    """Attempt to reset the camera using various methods"""
    print("Attempting to reset camera...")
    success = False
    
    # Method 1: Close and reopen the camera
    try:
        temp_cap = cv2.VideoCapture(cam_device_num)
        temp_cap.release()
        time.sleep(0.5)
        print("Reset camera through close/reopen")
        success = True
    except:
        pass
    
    # Method 2: Try to reset UVC driver if available
    if not success:
        try:
            subprocess.run(['sudo', 'rmmod', 'uvcvideo'], check=False, stderr=subprocess.PIPE)
            time.sleep(1)
            subprocess.run(['sudo', 'modprobe', 'uvcvideo'], check=False, stderr=subprocess.PIPE)
            print("Reset UVC video driver")
            success = True
        except:
            pass
    
    # Method 3: Try to reset using v4l2-ctl if available
    if not success:
        try:
            subprocess.run(['v4l2-ctl', '--set-ctrl=power_line_frequency=0'], check=False)
            time.sleep(0.5)
            print("Reset camera using v4l2-ctl")
            success = True
        except:
            pass
    
    # Method 4: Try a more aggressive approach - USB power cycle
    if not success:
        print("Trying USB power cycle...")
        toggle_usb_power(False)
        time.sleep(1)  # Longer off time
        toggle_usb_power(True)
        time.sleep(2)  # Longer on time
        success = True
    
    return success


def auto_power_cycle_usb(power_cycle_callback=None):
    """Automatically power cycle the USB - turn it off, wait 3 seconds, turn it back on"""
    # Flag to track power cycle state
    is_cycling = True
    
    # Function to perform the power cycle in a separate thread
    def power_cycle_thread():
        # Turn power off
        toggle_usb_power(turn_on=False)
        print("USB power turned OFF, waiting 3 seconds...")
        
        # Wait 3 seconds
        time.sleep(3)
        
        # Turn power back on
        toggle_usb_power(turn_on=True)
        print("USB power turned back ON, waiting for device initialization...")
        
        # Wait for USB device to be fully recognized (longer wait)
        time.sleep(2)
        
        # Force USB device reset to ensure it's properly recognized
        reset_camera_device()
        
        # Wait a bit more
        time.sleep(1)
        
        # Check if camera is working
        camera_working = ensure_camera_works(max_attempts=3, delay=1.0)
        
        if power_cycle_callback:
            # Call the callback with the status
            power_cycle_callback(camera_working)
    
    # Start the power cycle in a background thread
    cycle_thread = threading.Thread(target=power_cycle_thread)
    cycle_thread.daemon = True  # Thread will exit when main program exits
    cycle_thread.start()
    
    return is_cycling


def main():
    # Check for camera USB information
    camera_info = get_camera_usb_info()
    if camera_info:
        print(f"Found camera: {camera_info['description']}")
    else:
        print("Warning: Could not identify camera USB device. Power toggle may not work.")
    
    # Open the default camera (usually webcam)
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam. Attempting reset...")
        # Try to reset the camera
        reset_camera_device()
        time.sleep(2)
        # Try opening again
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Still could not open webcam. Starting with no camera.")

    # Initial threshold value (0-255)
    threshold = 120
    
    # Flag to track if a power cycle is in progress
    power_cycling = False
    
    # Flag to track if camera is currently available
    camera_available = cap.isOpened()

    print("Controls:")
    print("  + : Increase threshold")
    print("  - : Decrease threshold")
    print("  r : Reset threshold to 120")
    print("  p : Power cycle USB (will auto-restore after 3 seconds)")
    print("  c : Try to reconnect camera without power cycle")
    print("  q : Quit the program")

    # Create a window with a trackbar for the power button
    cv2.namedWindow('Controls')
    # Create a button-like trackbar (it's a bit of a hack since OpenCV doesn't have buttons)
    cv2.createTrackbar('Power Cycle', 'Controls', 0, 1, lambda x: None)
    cv2.createTrackbar('Try Reconnect', 'Controls', 0, 1, lambda x: None)

    # Track last power cycle time to prevent rapid cycling
    last_cycle_time = 0
    min_cycle_interval = 5  # Minimum time between cycles in seconds
    
    # Initialize status message
    status_message = "READY" if camera_available else "NO CAMERA DETECTED"
    
    # Define callback for power cycle completion
    def power_cycle_complete_callback(success):
        nonlocal power_cycling, camera_available, cap, status_message
        
        # Update flags based on success
        power_cycling = False
        
        if success:
            print("Camera reconnected successfully after power cycle")
            status_message = "CAMERA RECONNECTED"
            
            # Make sure to properly release the old capture
            if cap.isOpened():
                cap.release()
            
            # Open a fresh capture
            cap = cv2.VideoCapture(0)
            camera_available = cap.isOpened()
            
            if not camera_available:
                status_message = "RECONNECT FAILED - TRY AGAIN"
                print("Warning: Camera still not available after power cycle")
        else:
            print("Failed to reconnect camera after power cycle")
            status_message = "RECONNECT FAILED - TRY AGAIN"
            camera_available = False

    # Function to attempt reconnection without power cycle
    def try_reconnect_camera():
        nonlocal cap, camera_available, status_message
        
        print("Attempting to reconnect camera without power cycle...")
        status_message = "RECONNECTING..."
        
        # Release current capture if any
        if cap.isOpened():
            cap.release()
        
        # Try to reset the camera
        reset_camera_device()
        
        # Wait a moment
        time.sleep(1)
        
        # Try to open the camera again
        cap = cv2.VideoCapture(0)
        camera_available = cap.isOpened()
        
        if camera_available:
            print("Camera reconnected successfully")
            status_message = "CAMERA RECONNECTED"
            return True
        else:
            print("Failed to reconnect camera")
            status_message = "RECONNECT FAILED - TRY AGAIN"
            return False

    while True:
        # Get current time for rate limiting
        current_time = time.time()
        
        # Create a blank control panel to show the current status
        control_panel = np.ones((150, 400, 3), dtype=np.uint8) * 240  # Light gray background
        
        # Add status text
        status_color = (0, 0, 255)  # Default is red
        if status_message == "READY" or status_message == "CAMERA RECONNECTED":
            status_color = (0, 128, 0)  # Green
        elif status_message == "POWER CYCLING..." or status_message == "RECONNECTING...":
            status_color = (0, 165, 255)  # Orange
            
        cv2.putText(control_panel, f"Status: {status_message}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(control_panel, f"Threshold: {threshold}", 
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        if power_cycling:
            cv2.putText(control_panel, "USB power will auto-restore in 3 seconds", 
                       (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        else:
            cv2.putText(control_panel, "Press 'p' to power cycle, 'c' to reconnect, 'q' to quit", 
                       (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Show the control panel
        cv2.imshow('Controls', control_panel)
        
        # Check the state of the power cycle "button"
        button_state = cv2.getTrackbarPos('Power Cycle', 'Controls')
        
        # If button is pressed (1) and not currently cycling and enough time has passed
        if button_state == 1 and not power_cycling and (current_time - last_cycle_time) > min_cycle_interval:
            # Reset the button state to 0
            cv2.setTrackbarPos('Power Cycle', 'Controls', 0)
            
            # Start the auto power cycle
            power_cycling = True
            last_cycle_time = current_time
            status_message = "POWER CYCLING..."
            print("Starting USB power cycle...")
            
            # Start the power cycle in the background with the callback
            auto_power_cycle_usb(power_cycle_complete_callback)
        
        # Check the state of the reconnect "button"
        reconnect_button_state = cv2.getTrackbarPos('Try Reconnect', 'Controls')
        
        # If reconnect button is pressed and not power cycling
        if reconnect_button_state == 1 and not power_cycling:
            # Reset the button state to 0
            cv2.setTrackbarPos('Try Reconnect', 'Controls', 0)
            
            # Try to reconnect
            try_reconnect_camera()
        
        # Only try to capture a frame if the camera is available and not power cycling
        if camera_available and not power_cycling:
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to capture image from camera.")
                # If we can't get a frame, the camera might be disconnected
                status_message = "CAMERA ERROR - TRY RECONNECTING"
                camera_available = False
                try_reconnect_camera()
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
            # Display a message when camera is unavailable
            no_signal = np.zeros((480, 640, 3), dtype=np.uint8)
            
            message = "POWER CYCLING..." if power_cycling else "CAMERA UNAVAILABLE"
            cv2.putText(no_signal, message, (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Add instructions
            if power_cycling:
                cv2.putText(no_signal, "USB power will auto-restore in 3 seconds", 
                           (90, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
            else:
                cv2.putText(no_signal, "Press 'p' to power cycle or 'c' to try reconnect", 
                           (90, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
            
            # Display the no signal frame
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
            # Start power cycle
            power_cycling = True
            last_cycle_time = current_time
            status_message = "POWER CYCLING..."
            print("Starting USB power cycle...")
            
            # Start the power cycle in the background with the callback
            auto_power_cycle_usb(power_cycle_complete_callback)
        elif key == ord('c') and not power_cycling:
            # Try to reconnect camera without power cycling
            try_reconnect_camera()
        elif key == ord('q'):
            break

    # When everything is done, release the capture and close windows
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
