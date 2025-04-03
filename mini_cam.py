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


def get_camera_device_path():
    """Find the specific USB device path for the camera"""
    try:
        # Run v4l2-ctl --list-devices to find video devices
        result = subprocess.run(['v4l2-ctl', '--list-devices'], stdout=subprocess.PIPE, text=True)
        output = result.stdout.strip()
        
        # Look for a camera device
        device_sections = output.split('\n\n')
        camera_path = None
        
        for section in device_sections:
            lines = section.strip().split('\n')
            if len(lines) >= 2:
                if any(cam_term in lines[0].lower() for cam_term in ['cam', 'webcam', 'video', 'uvc']):
                    # Found a likely camera, get its video device path
                    for line in lines[1:]:
                        if '/dev/video' in line:
                            camera_path = line.strip()
                            break
                    if camera_path:
                        break
        
        if camera_path:
            # Get the real path to the device
            result = subprocess.run(['readlink', '-f', camera_path], stdout=subprocess.PIPE, text=True)
            real_path = result.stdout.strip()
            
            # Extract the device name (e.g., "video0")
            device_name = os.path.basename(real_path)
            
            return {
                'device_path': camera_path,
                'real_path': real_path,
                'device_name': device_name
            }
        
        return None
    except Exception as e:
        print(f"Error finding camera device: {e}")
        return None


def find_specific_camera_usb_port():
    """Find the specific USB port where the camera is connected"""
    try:
        camera_info = get_camera_device_path()
        if not camera_info:
            print("Could not identify camera device")
            return None
            
        # Use sysfs to find the USB device for this video device
        # Example path: /sys/class/video4linux/video0/device/
        video_device_dir = f"/sys/class/video4linux/{camera_info['device_name']}"
        
        if not os.path.exists(video_device_dir):
            print(f"Video device directory not found: {video_device_dir}")
            return None
            
        # Navigate up the USB device tree to find the specific port
        # Try to find a directory structure that includes "usb"
        result = subprocess.run(['find', video_device_dir, '-name', '*usb*', '-type', 'd'], 
                              stdout=subprocess.PIPE, text=True)
        usb_paths = result.stdout.strip().split('\n')
        
        for path in usb_paths:
            if path and 'usb' in path:
                # Look for a directory pattern that includes a port number
                # USB paths often include a port identifier like "1-1.2" where:
                # - 1-1 indicates hub-port
                # - .2 indicates the specific port on that hub
                parts = path.split('/')
                for part in parts:
                    if '-' in part and any(c.isdigit() for c in part):
                        # This looks like a USB port identifier (e.g., "1-1.2")
                        port_id = part
                        # Get the full physical port path
                        port_path = f"/sys/bus/usb/devices/{port_id}"
                        if os.path.exists(port_path):
                            return {
                                'port_id': port_id,
                                'port_path': port_path,
                                'device_name': camera_info['device_name']
                            }
        
        # If we couldn't find a specific port, try a general approach
        # Find all USB devices
        result = subprocess.run(['lsusb'], stdout=subprocess.PIPE, text=True)
        usb_devices = result.stdout.strip().split('\n')
        
        # Look for a camera device in the USB list
        for device in usb_devices:
            if any(cam_term in device.lower() for cam_term in ['cam', 'webcam', 'camera', 'video', 'uvc']):
                parts = device.split()
                if len(parts) >= 6:
                    bus = parts[1].zfill(3)  # Zero-pad to 3 digits
                    device_num = parts[3].rstrip(':').zfill(3)  # Remove trailing colon and zero-pad
                    port_id = f"{bus}/{device_num}"
                    port_path = f"/dev/bus/usb/{bus}/{device_num}"
                    return {
                        'port_id': port_id,
                        'port_path': port_path,
                        'device_name': camera_info['device_name']
                    }
        
        return None
    except Exception as e:
        print(f"Error finding specific camera USB port: {e}")
        return None


def control_specific_usb_port(port_info, turn_on=True):
    """Control power to a specific USB port"""
    if not port_info:
        print("No USB port information available")
        return False
        
    try:
        # Method 1: Try using the authorized attribute if available
        if 'port_path' in port_info and os.path.exists(port_info['port_path']):
            authorized_path = f"{port_info['port_path']}/authorized"
            if os.path.exists(authorized_path):
                with open(authorized_path, 'w') as f:
                    f.write('1' if turn_on else '0')
                print(f"USB port power set to {'ON' if turn_on else 'OFF'} using authorized attribute")
                return True
        
        # Method 2: Try using uhubctl if installed (more precise control)
        try:
            # Check if uhubctl is installed
            subprocess.run(['which', 'uhubctl'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # If port_id has a format like "1-1.2", we can use it with uhubctl
            if 'port_id' in port_info and '-' in port_info['port_id']:
                # Extract hub and port information
                parts = port_info['port_id'].split('-')
                if len(parts) == 2:
                    hub = parts[0]
                    port = parts[1]
                    
                    # Use uhubctl to control just this specific port
                    action = 'on' if turn_on else 'off'
                    cmd = ['uhubctl', '-l', hub, '-p', port, '-a', action]
                    subprocess.run(cmd, check=True)
                    print(f"USB port {port_info['port_id']} power set to {'ON' if turn_on else 'OFF'} using uhubctl")
                    return True
            
            # If we don't have specific port info, try a more general approach with uhubctl
            action = 'on' if turn_on else 'off'
            cmd = ['uhubctl', '-a', action]
            subprocess.run(cmd, check=True)
            print(f"All USB ports power set to {'ON' if turn_on else 'OFF'} using uhubctl")
            return True
            
        except (subprocess.SubprocessError, FileNotFoundError):
            # uhubctl not installed or failed
            pass
            
        # Method 3: Try using usbreset for the specific device
        if turn_on and 'device_path' in port_info:
            try:
                # Try using usbreset tool if available
                subprocess.run(['usbreset', port_info['device_path']], check=True)
                print(f"Reset USB device at {port_info['device_path']}")
                return True
            except (subprocess.SubprocessError, FileNotFoundError):
                # usbreset not installed or failed
                pass
        
        # Method 4: For power off, try unbinding the driver
        if not turn_on and 'device_name' in port_info:
            try:
                # Find the USB driver for this device
                result = subprocess.run(['lsmod'], stdout=subprocess.PIPE, text=True)
                modules = result.stdout.strip().split('\n')
                
                # Look for common USB video drivers
                video_drivers = ['uvcvideo', 'videodev', 'usb_video']
                for driver in video_drivers:
                    if any(driver in line for line in modules):
                        # Try to unbind this device from the driver
                        unbind_path = f"/sys/bus/usb/drivers/{driver}/unbind"
                        if os.path.exists(unbind_path):
                            with open(unbind_path, 'w') as f:
                                f.write(port_info['port_id'])
                            print(f"Unbound device {port_info['device_name']} from driver {driver}")
                            return True
            except Exception as e:
                print(f"Error unbinding driver: {e}")
        
        # Method 5: As a last resort, try the more general approach
        # Warning: This might affect other USB devices
        if not turn_on:
            try:
                # Try to disable all USB devices (not recommended, but might work)
                for i in range(1, 8):  # Try common USB bus numbers
                    # Disable the USB bus
                    cmd = f"echo '1-{i}' > /sys/bus/usb/drivers/usb/unbind"
                    subprocess.run(['sudo', 'sh', '-c', cmd], check=False)
                print("Attempted to disable USB devices (general approach)")
                return True
            except Exception as e:
                print(f"Error disabling USB (general approach): {e}")
        else:
            try:
                # Try to enable all USB devices
                for i in range(1, 8):  # Try common USB bus numbers
                    # Enable the USB bus
                    cmd = f"echo '1-{i}' > /sys/bus/usb/drivers/usb/bind"
                    subprocess.run(['sudo', 'sh', '-c', cmd], check=False)
                print("Attempted to enable USB devices (general approach)")
                return True
            except Exception as e:
                print(f"Error enabling USB (general approach): {e}")
                
        return False
    except Exception as e:
        print(f"Error controlling USB port: {e}")
        return False


def power_cycle_camera_usb(port_info, callback=None):
    """Power cycle the camera USB port with a delay of 3 seconds"""
    # Define the power cycle sequence
    def power_cycle_sequence():
        # Disable power
        control_specific_usb_port(port_info, turn_on=False)
        print("USB power OFF - waiting 3 seconds before turning back ON...")
        
        # Wait 3 seconds
        time.sleep(3)
        
        # Enable power
        control_specific_usb_port(port_info, turn_on=True)
        print("USB power ON")
        
        # If a callback was provided, call it
        if callback:
            callback()
    
    # Start the power cycle in a separate thread to avoid blocking the UI
    power_thread = threading.Thread(target=power_cycle_sequence)
    power_thread.daemon = True  # Thread will exit when main program exits
    power_thread.start()
    
    return True


def main():
    # Find the camera's USB port
    camera_port_info = find_specific_camera_usb_port()
    if camera_port_info:
        print(f"Found camera on USB port: {camera_port_info['port_id']}")
    else:
        print("Warning: Could not identify specific camera USB port. Power cycling may affect all USB devices.")
    
    # Open the default camera (usually webcam)
    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Initial threshold value (0-255)
    threshold = 120
    
    # Power cycling state flag
    power_cycling = False

    print("Controls:")
    print("  + : Increase threshold")
    print("  - : Decrease threshold")
    print("  r : Reset threshold to 120")
    print("  p : Power cycle camera (OFF for 3 seconds, then back ON)")
    print("  q : Quit the program")

    # Create a window with a button for power cycling
    cv2.namedWindow('Controls')
    
    # Create a button-like trackbar for power cycling
    # (When pressed, it will be reset to 0 automatically)
    cv2.createTrackbar('Power Cycle Camera', 'Controls', 0, 1, lambda x: None)

    # Function to reconnect the camera after power cycling
    def reconnect_camera():
        nonlocal cap, power_cycling
        time.sleep(2)  # Wait a bit more for device to be ready
        print("Attempting to reconnect camera...")
        cap.release()
        cap = cv2.VideoCapture(0)
        power_cycling = False
        print("Camera reconnection complete")

    while True:
        # Check the state of the power cycle "button"
        button_state = cv2.getTrackbarPos('Power Cycle Camera', 'Controls')
        
        # If button is pressed (1) and we're not already power cycling
        if button_state == 1 and not power_cycling:
            # Reset the button to 0
            cv2.setTrackbarPos('Power Cycle Camera', 'Controls', 0)
            
            # Set the power cycling flag
            power_cycling = True
            
            # Start the power cycle
            print("Starting USB power cycle...")
            power_cycle_camera_usb(camera_port_info, reconnect_camera)
        
        # Create a blank control panel to show the current status
        control_panel = np.ones((150, 400, 3), dtype=np.uint8) * 240  # Light gray background
        
        # Add status text
        status_text = "POWER CYCLING..." if power_cycling else "READY"
        status_color = (0, 0, 255) if power_cycling else (0, 128, 0)
        
        cv2.putText(control_panel, f"Status: {status_text}", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(control_panel, f"Threshold: {threshold}", 
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(control_panel, "Press 'p' to power cycle, 'q' to quit", 
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Show the control panel
        cv2.imshow('Controls', control_panel)
        
        # Only try to capture a frame if we're not power cycling
        if not power_cycling and cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to capture image from camera.")
                # If we can't get a frame, assume camera might be disconnected
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
            # Display a message when camera is power cycling or unavailable
            message = "POWER CYCLING..." if power_cycling else "CAMERA UNAVAILABLE"
            no_signal = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(no_signal, message, (160, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
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
        elif key == ord('p') and not power_cycling:
            # Start power cycling
            power_cycling = True
            print("Starting USB power cycle...")
            power_cycle_camera_usb(camera_port_info, reconnect_camera)
        elif key == ord('q'):
            break

    # When everything is done, release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
