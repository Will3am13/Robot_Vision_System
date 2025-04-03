import time
import cv2
import numpy as np
import depthai as dai
import gc
import logging
import multiprocessing
import subprocess
import os
import sys
from typing import Dict, List, Tuple, Any, Optional

# Import settings from config
from config import (
    CLASS_SETTINGS, OBJECT_CLEANUP_TIME, Z_HISTORY_MAX_SIZE,
    OAK_BLOB_PATH
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("OakCamera")


def setup_vision_pipeline():
    """
    Setup and configure the DepthAI vision pipeline
    
    Returns:
        dai.Pipeline: Configured pipeline
    """
    logger.info("Setting up DepthAI vision pipeline")
    
    # Create pipeline
    pipeline = dai.Pipeline()

    # Define sources and outputs
    camRgb = pipeline.create(dai.node.ColorCamera)
    spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
    stereo = pipeline.create(dai.node.StereoDepth)
    monoRight = pipeline.create(dai.node.MonoCamera)

    # Create output nodes
    xoutVideo = pipeline.create(dai.node.XLinkOut)
    xoutNN = pipeline.create(dai.node.XLinkOut)

    # Set stream names
    xoutVideo.setStreamName("video")
    xoutNN.setStreamName("detections")

    # Properties for the color camera (left camera)
    camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
    camRgb.setVideoSize(1280, 800)
    camRgb.setPreviewSize(640, 352)  # Match YOLO input size
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    # Properties for the mono camera (right camera)
    monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)

    # Stereo depth settings
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
    stereo.setLeftRightCheck(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_B)
    stereo.setMedianFilter(dai.MedianFilter.KERNEL_3x3)

    # YOLO specific settings
    spatialDetectionNetwork.setBlobPath(OAK_BLOB_PATH)
    spatialDetectionNetwork.setConfidenceThreshold(0.2)
    spatialDetectionNetwork.input.setBlocking(False)
    spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
    spatialDetectionNetwork.setDepthLowerThreshold(100)
    spatialDetectionNetwork.setDepthUpperThreshold(5000)

    # YOLO input configuration
    spatialDetectionNetwork.setNumClasses(2)  # Battery and CBattery
    spatialDetectionNetwork.setCoordinateSize(4)
    spatialDetectionNetwork.setAnchors([])
    spatialDetectionNetwork.setAnchorMasks({})
    spatialDetectionNetwork.setIouThreshold(0.5)

    # Link nodes
    camRgb.preview.link(spatialDetectionNetwork.input)
    camRgb.video.link(xoutVideo.input)
    camRgb.video.link(stereo.left)
    monoRight.out.link(stereo.right)

    stereo.depth.link(spatialDetectionNetwork.inputDepth)
    spatialDetectionNetwork.out.link(xoutNN.input)

    logger.info("DepthAI vision pipeline configured successfully")
    return pipeline


def get_avg_z(object_data):
    """
    Calculate average Z value for an object from its Z history
    
    Args:
        object_data (dict): Object data containing z_history
        
    Returns:
        float: Average Z value or None if history is empty
    """
    z_history = object_data.get("z_history", [])
    if not z_history:
        return None
    return sum(z_history) / len(z_history)


def reset_usb_device(device_id=None):
    """
    Attempt to reset the USB device programmatically 
    
    Args:
        device_id (str): Optional device ID to match
    
    Returns:
        bool: True if reset was successful, False otherwise
    """
    try:
        # Find DepthAI devices
        logger.info("Looking for DepthAI devices to reset...")
        
        # First attempt: Using DepthAI API
        devices = dai.Device.getAllAvailableDevices()
        
        if devices:
            logger.info(f"Found {len(devices)} DepthAI devices")
            
            for i, device_info in enumerate(devices):
                mxid = device_info.getMxId()
                if device_id is None or device_id in mxid:
                    logger.info(f"Attempting to reset device {mxid}")
                    try:
                        # Try to open and immediately close with USB reset flag
                        with dai.Device(dai.OpenVINO.VERSION_2021_4, mxid, True) as d:
                            # Force device reset
                            d.close()
                            logger.info(f"Successfully reset device {mxid}")
                            time.sleep(2)  # Allow time for device to restart
                            return True
                    except Exception as e:
                        logger.error(f"Failed to reset device {mxid}: {str(e)}")
        
        # Second attempt: Use system-specific USB reset tools
        if sys.platform == "linux" or sys.platform == "linux2":
            # Try using Linux-specific usbreset utility or libusb
            try:
                # Find USB device with depthai in name or description
                logger.info("Attempting Linux USB reset...")
                
                # Try using lsusb to find the device
                output = subprocess.check_output("lsusb", shell=True).decode("utf-8")
                lines = output.strip().split("\n")
                
                # Look for DepthAI or LUXONIS devices
                for line in lines:
                    if "Luxonis" in line or "DepthAI" in line or "Movidius" in line:
                        parts = line.split()
                        if len(parts) >= 6:
                            bus = parts[1]
                            device = parts[3].rstrip(":")
                            
                            logger.info(f"Found DepthAI device at Bus {bus} Device {device}")
                            
                            # Attempt reset using unbind/bind technique (safer)
                            try:
                                # Get USB path
                                command = f"sudo sh -c 'echo {bus}-{device} > /sys/bus/usb/drivers/usb/unbind && sleep 2 && echo {bus}-{device} > /sys/bus/usb/drivers/usb/bind'"
                                subprocess.run(command, shell=True, timeout=5)
                                logger.info(f"Successfully reset USB device at {bus}:{device}")
                                time.sleep(3)  # Give device time to reset
                                return True
                            except Exception as e:
                                logger.error(f"Error unbinding/binding USB device: {str(e)}")
            except Exception as e:
                logger.error(f"Linux USB reset failed: {str(e)}")
                
        elif sys.platform == "darwin":
            # macOS doesn't have a simple command-line tool for resetting USB devices
            logger.info("USB reset on macOS is not supported programmatically")
            
        elif sys.platform == "win32":
            # On Windows, using devcon would be ideal, but it's not available by default
            logger.info("Attempting Windows USB reset...")
            try:
                # Using PowerShell to reset USB devices
                # This requires admin privileges
                command = 'powershell "Get-PnpDevice | Where-Object {$_.FriendlyName -like \'*DepthAI*\'} | Disable-PnpDevice -Confirm:$false; Start-Sleep -Seconds 2; Get-PnpDevice | Where-Object {$_.FriendlyName -like \'*DepthAI*\'} | Enable-PnpDevice -Confirm:$false"'
                subprocess.run(command, shell=True, timeout=10)
                logger.info("Successfully reset USB devices using PowerShell")
                time.sleep(3)  # Allow time for device to restart
                return True
            except Exception as e:
                logger.error(f"Windows USB reset failed: {str(e)}")
                
        logger.info("Could not reset USB device through available methods")
        return False
        
    except Exception as e:
        logger.error(f"Error in reset_usb_device: {str(e)}")
        return False


def recover_from_error(device, error_message, last_recovery_time, status_queue):
    """
    Attempt to recover from a device error
    
    Args:
        device: Current device instance
        error_message: Error message string
        last_recovery_time: Time of last recovery attempt
        status_queue: Queue to send status updates
        
    Returns:
        tuple: (success, device, last_recovery_time)
            - success: True if recovery was successful
            - device: New device instance or None
            - last_recovery_time: Updated recovery time
    """
    current_time = time.time()
    
    # Don't attempt recovery too frequently
    if current_time - last_recovery_time < 30:  # Minimum 30 seconds between recovery attempts
        logger.warning("Recovery attempt too soon after previous attempt. Skipping.")
        return False, device, last_recovery_time
        
    # Update status
    status_queue.put({"status": "recovering", "message": "Attempting to recover from communication error"})
    logger.warning(f"Attempting to recover from error: {error_message}")
    
    # Try to close device gracefully if it still exists
    if device is not None:
        try:
            logger.info("Closing existing device connection")
            device.close()
        except Exception as e:
            logger.error(f"Error closing device: {str(e)}")
    
    # Reset device through USB   
    logger.info("Attempting USB device reset")
    usb_reset_success = reset_usb_device()
    
    if not usb_reset_success:
        logger.warning("USB reset failed, will attempt to recreate device anyway")
    
    # Force garbage collection
    gc.collect()
    
    # Wait for device to stabilize
    time.sleep(3)
    
    # Create a new pipeline
    try:
        pipeline = setup_vision_pipeline()
        
        # Check if devices are available
        available_devices = dai.Device.getAllAvailableDevices()
        if not available_devices:
            logger.error("No DepthAI devices found after recovery attempt")
            return False, None, current_time
            
        # Try to create a new device
        logger.info("Creating new device instance")
        new_device = dai.Device(pipeline)
        logger.info("Successfully created new device instance")
        
        # Update status
        status_queue.put({"status": "recovered", "message": "Successfully recovered from error"})
        
        return True, new_device, current_time
        
    except Exception as e:
        logger.error(f"Failed to recover: {str(e)}")
        status_queue.put({"status": "error", "message": f"Recovery failed: {str(e)}"})
        return False, None, current_time


def run_oak_camera(detection_queue, command_queue, status_queue, frame_queue):
    """
    Run the Oak-D-SR camera in a separate process
    
    Args:
        detection_queue (multiprocessing.Queue): Queue to send detections to main process
        command_queue (multiprocessing.Queue): Queue to receive commands from main process
        status_queue (multiprocessing.Queue): Queue to send status updates to main process
        frame_queue (multiprocessing.Queue): Queue to send video frames to main process
    """
    logger.info("Starting Oak camera process")
    
    # Object tracking data
    class_objects = {
        "Battery": {},
        "CBattery": {}
    }
    
    # Error tracking
    last_recovery_time = 0
    error_count = 0
    max_consecutive_errors = 5
    consecutive_errors = 0
    
    try:
        # Set up the pipeline
        pipeline = setup_vision_pipeline()
        
        # Initialize status
        status_queue.put({"status": "initializing"})
        
        # Connect to device with improved error handling and retry
        max_retries = 5  # Increased from 3 to 5
        retry_count = 0
        device = None
        
        while retry_count < max_retries:
            try:
                logger.info(f"Connecting to Oak device (attempt {retry_count + 1}/{max_retries})")
                
                # Release any previously attempted device connection
                if device is not None:
                    try:
                        device.close()
                    except:
                        pass
                
                # Force garbage collection before attempting connection
                gc.collect()
                
                # Check for USB devices
                logger.info("Checking for available devices...")
                try:
                    devices = dai.Device.getAllAvailableDevices()
                    if len(devices) == 0:
                        logger.warning("No DepthAI devices found! Waiting for device to become available...")
                        status_queue.put({"status": "waiting_for_device"})
                    else:
                        logger.info(f"Found {len(devices)} DepthAI device(s)")
                        for i, deviceInfo in enumerate(devices):
                            logger.info(f"Device {i}: {deviceInfo.getMxId()} (state: {deviceInfo.state})")
                except Exception as dev_e:
                    logger.error(f"Error checking devices: {str(dev_e)}")
                
                # Try to create device
                device = dai.Device(pipeline)
                logger.info("Successfully connected to Oak device")
                break
                
            except Exception as e:
                retry_count += 1
                logger.error(f"Failed to connect to Oak device: {str(e)}")
                status_queue.put({"status": "retrying", "attempt": retry_count, "max_attempts": max_retries})
                
                if retry_count >= max_retries:
                    status_queue.put({"status": "error", "message": f"Failed to connect to Oak device after {max_retries} attempts: {str(e)}"})
                    return
                
                # Exponential backoff for retries (2^retry_count seconds)
                wait_time = min(2 ** retry_count, 10)  # Cap at 10 seconds
                logger.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
        
        # Output queues with limited size to prevent memory leaks
        qVideo = device.getOutputQueue(name="video", maxSize=4, blocking=False)
        qDet = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        
        # Update status
        status_queue.put({"status": "running"})
        logger.info("Oak camera is now running")
        
        running = True
        paused = False
        frame_count = 0
        last_gc_time = time.time()
        frozen_frame = None  # Store the last frame when paused
        
        while running:
            try:
                # Check for commands from main process (non-blocking)
                if not command_queue.empty():
                    cmd = command_queue.get()
                    if cmd.get("command") == "stop":
                        logger.info("Received stop command")
                        running = False
                        break
                    elif cmd.get("command") == "pause":
                        if not paused:
                            logger.info("Received pause command, pausing camera processing")
                            # Save current frame as frozen frame
                            if frozen_frame is None:
                                try:
                                    # Get current frame
                                    current_frame = qVideo.get().getCvFrame()
                                    # Create a copy to avoid reference issues
                                    frozen_frame = current_frame.copy()
                                    # Add paused indicator
                                    cv2.putText(frozen_frame, "CAMERA PAUSED", (50, 50), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                                    # Resize for display
                                    frozen_frame = cv2.resize(frozen_frame, (640, 400))
                                except Exception as e:
                                    logger.error(f"Error capturing frozen frame: {str(e)}")
                            
                            paused = True
                            status_queue.put({"status": "paused"})
                    elif cmd.get("command") == "resume":
                        if paused:
                            logger.info("Received resume command, resuming camera processing")
                            # Clear frozen frame
                            frozen_frame = None
                            paused = False
                            status_queue.put({"status": "running"})
                    elif cmd.get("command") == "reset":
                        logger.info("Received reset command, clearing object tracking data")
                        class_objects = {
                            "Battery": {},
                            "CBattery": {}
                        }
                    elif cmd.get("command") == "force_recovery":
                        logger.info("Received force recovery command")
                        # Force device recovery
                        success, device, last_recovery_time = recover_from_error(
                            device, "Manual recovery requested", 0, status_queue
                        )
                        
                        if success and device is not None:
                            # Recreate queues for the new device
                            qVideo = device.getOutputQueue(name="video", maxSize=4, blocking=False)
                            qDet = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
                            logger.info("Manual recovery successful")
                            # Reset error counters
                            consecutive_errors = 0
                        else:
                            logger.error("Manual recovery failed")
                            running = False
                            break
                
                # If paused, send the frozen frame but don't process new frames
                if paused:
                    try:
                        # Send the frozen frame to main process
                        if frozen_frame is not None and frame_queue.qsize() < 2:
                            frame_queue.put(frozen_frame, block=False)
                    except Exception as e:
                        logger.warning(f"Error sending frozen frame: {str(e)}")
                        
                    # Empty the input queues to prevent backlog
                    while qVideo.has():
                        qVideo.get()
                    while qDet.has():
                        qDet.get()
                        
                    # Sleep to reduce CPU usage while paused
                    time.sleep(0.1)
                    continue
                
                # Process frames with error handling
                try:
                    # Get frames and detections
                    inVideo = qVideo.get()
                    frame = inVideo.getCvFrame()
                    
                    # Only process detections if not paused
                    inDet = qDet.get()
                    detections = inDet.detections
                    
                    # Reset consecutive error counter on successful frame processing
                    consecutive_errors = 0
                    
                except Exception as e:
                    error_message = str(e)
                    error_count += 1
                    consecutive_errors += 1
                    
                    logger.error(f"Error processing frames: {error_message}")
                    
                    # Check for communication errors that might need device recovery
                    communication_error = False
                    if "Communication exception" in error_message or "X_LINK_ERROR" in error_message:
                        communication_error = True
                        logger.error("Detected communication error with Oak camera")
                    
                    # Try recovery if we have communication errors or too many consecutive errors
                    if communication_error or consecutive_errors >= max_consecutive_errors:
                        logger.warning(f"Attempting recovery after {consecutive_errors} consecutive errors")
                        
                        success, new_device, last_recovery_time = recover_from_error(
                            device, error_message, last_recovery_time, status_queue
                        )
                        
                        if success and new_device is not None:
                            # Update device and queues
                            device = new_device
                            qVideo = device.getOutputQueue(name="video", maxSize=4, blocking=False)
                            qDet = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
                            # Reset error counter
                            consecutive_errors = 0
                        else:
                            logger.error("Recovery failed, will retry later")
                            
                            # Notify main process
                            status_queue.put({
                                "status": "error", 
                                "message": "Device communication error, continuing to retry"
                            })
                            
                            # If we've failed recovery attempts, wait longer
                            time.sleep(5)
                    
                    # Skip this iteration and try again
                    continue
                
                frame_count += 1
                
                # Track best detection
                best_detection = None
                best_confidence = 0
                
                # Process each detection
                for detection in detections:
                    # Determine class name
                    class_name = "Battery" if detection.label == 0 else "CBattery"
                    
                    # Skip if below class-specific threshold
                    if class_name not in CLASS_SETTINGS or detection.confidence < CLASS_SETTINGS[class_name]["threshold"]:
                        continue
                    
                    # Get spatial coordinates (in millimeters)
                    x = detection.spatialCoordinates.x
                    y = detection.spatialCoordinates.y
                    z = detection.spatialCoordinates.z
                    
                    # Create a unique object identifier based on its approximate position in 3D space
                    # Round position to nearest 10mm to account for small movements
                    object_id = f"{round(x / 10) * 10}_{round(y / 10) * 10}"
                    
                    # Initialize object data if this is a new object
                    if object_id not in class_objects[class_name]:
                        class_objects[class_name][object_id] = {
                            "z_history": [],
                            "last_seen": time.time()
                        }
                    
                    # Update last seen time
                    class_objects[class_name][object_id]["last_seen"] = time.time()
                    
                    # Update Z history for this specific object (keep only last Z_HISTORY_MAX_SIZE values)
                    class_objects[class_name][object_id]["z_history"].append(z)
                    if len(class_objects[class_name][object_id]["z_history"]) > Z_HISTORY_MAX_SIZE:
                        class_objects[class_name][object_id]["z_history"].pop(0)
                    
                    # Calculate average Z for this specific object
                    avg_z = get_avg_z(class_objects[class_name][object_id])
                    
                    # Track best detection for picking
                    if detection.confidence > best_confidence:
                        best_confidence = detection.confidence
                        best_detection = {
                            "coordinates": [x, y, avg_z if avg_z is not None else z],  # Use averaged Z if available
                            "class_name": class_name,
                            "confidence": detection.confidence,
                            "object_id": object_id,
                            "raw_coordinates": [x, y, z],  # Also store raw coordinates
                            "frame": frame_count
                        }
                
                # Clean up old objects (not seen for more than OBJECT_CLEANUP_TIME seconds)
                current_time = time.time()
                for class_name in class_objects:
                    # Create a copy of the keys to avoid modifying during iteration
                    object_ids = list(class_objects[class_name].keys())
                    for object_id in object_ids:
                        if current_time - class_objects[class_name][object_id]["last_seen"] > OBJECT_CLEANUP_TIME:
                            del class_objects[class_name][object_id]
                
                # Send best detection to main process
                if best_detection:
                    # Don't want to send the entire frame to avoid large data transfer
                    # Just send the detection information
                    detection_queue.put(best_detection)
                
                # Send video frame to main process (only if queue not full to avoid blocking)
                try:
                    # Create a smaller version of the frame for display
                    display_frame = cv2.resize(frame, (640, 400))
                    
                    # Add detection visualization to the display frame
                    for detection in detections:
                        class_name = "Battery" if detection.label == 0 else "CBattery"
                        
                        # Skip if below threshold
                        if class_name not in CLASS_SETTINGS or detection.confidence < CLASS_SETTINGS[class_name]["threshold"]:
                            continue
                            
                        # Get bounding box coordinates
                        xmin = int(detection.xmin * display_frame.shape[1])
                        ymin = int(detection.ymin * display_frame.shape[0])
                        xmax = int(detection.xmax * display_frame.shape[1])
                        ymax = int(detection.ymax * display_frame.shape[0])
                        
                        # Get the color based on class
                        color = CLASS_SETTINGS[class_name]["color"]
                        
                        # Draw rectangle and text
                        cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), color, 2)
                        cv2.putText(display_frame, f"{class_name} {detection.confidence:.2f}",
                                    (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Put in queue with non-blocking (if queue is full, skip this frame)
                    if frame_queue.qsize() < 2:  # Only keep recent frames
                        frame_queue.put(display_frame, block=False)
                except Exception as e:
                    logger.warning(f"Error sending frame to main process: {str(e)}")
                
                # Periodic garbage collection to prevent memory leaks
                if current_time - last_gc_time > 10:  # Every 10 seconds
                    gc.collect()
                    last_gc_time = current_time
                
                # Sleep to reduce CPU usage
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in Oak camera process: {str(e)}")
                status_queue.put({"status": "error", "message": str(e)})
                time.sleep(1)  # Prevent tight error loop
        
        # Cleanup
        logger.info("Cleaning up Oak camera resources")
        if device is not None:
            device.close()
        
        # Final garbage collection
        gc.collect()
        
        # Update status
        status_queue.put({"status": "stopped"})
        logger.info("Oak camera process stopped")
        
    except Exception as e:
        logger.error(f"Fatal error in Oak camera process: {str(e)}")
        status_queue.put({"status": "error", "message": str(e)})


def start_oak_camera_process():
    """
    Start the Oak camera in a separate process
    
    Returns:
        tuple: (process, detection_queue, command_queue, status_queue, frame_queue)
    """
    # Create queues for communication between processes
    detection_queue = multiprocessing.Queue()
    command_queue = multiprocessing.Queue()
    status_queue = multiprocessing.Queue()
    frame_queue = multiprocessing.Queue(maxsize=2)  # Limit to 2 frames to prevent memory issues
    
    # Create and start the process
    process = multiprocessing.Process(
        target=run_oak_camera,
        args=(detection_queue, command_queue, status_queue, frame_queue)
    )
    process.daemon = True  # Process will terminate when main process exits
    process.start()
    
    logger.info(f"Started Oak camera process (PID: {process.pid})")
    
    return process, detection_queue, command_queue, status_queue, frame_queue


# For testing the Oak camera as a standalone module
if __name__ == "__main__":
    # Set up multiprocessing
    multiprocessing.set_start_method('spawn')
    
    # Start Oak camera process
    process, detection_queue, command_queue, status_queue, frame_queue = start_oak_camera_process()
    
    try:
        print("Oak camera started. Press Ctrl+C to exit.")
        
        # Wait for process to initialize
        status = status_queue.get(timeout=10)
        print(f"Oak camera status: {status}")
        
        # Create window for video feed
        cv2.namedWindow("Test Feed", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Test Feed", 640, 400)
        
        # Monitor detections for a while
        start_time = time.time()
        run_time = 30  # Run for 30 seconds
        
        print(f"Running test for {run_time} seconds")
        print("'p': pause, 'r': resume, 'f': force recovery, 'q': quit")
        
        while time.time() - start_time < run_time:
            if not detection_queue.empty():
                detection = detection_queue.get()
                print(f"Detection: {detection['class_name']} at {detection['coordinates']}")
            
            # Display video feed
            if not frame_queue.empty():
                frame = frame_queue.get()
                cv2.imshow("Test Feed", frame)
                
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('p'):
                print("Testing pause...")
                command_queue.put({"command": "pause"})
            elif key == ord('r'):
                print("Testing resume...")
                command_queue.put({"command": "resume"})
            elif key == ord('f'):
                print("Testing force recovery...")
                command_queue.put({"command": "force_recovery"})
            elif key == ord('q'):
                break
                
            time.sleep(0.01)
        
        # Send stop command
        print("Sending stop command...")
        command_queue.put({"command": "stop"})
        
        # Wait for process to stop
        process.join(timeout=5)
        
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up
        cv2.destroyAllWindows()
        if process.is_alive():
            print("Terminating Oak camera process...")
            process.terminate()
            process.join()
        
        print("Oak camera test complete")
