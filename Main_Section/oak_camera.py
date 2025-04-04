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
import subprocess
import re
import threading
from collections import deque
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

# Create a class to handle frame buffering
class FrameBuffer:
    def __init__(self, max_size=3):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
    
    def add_frame(self, frame):
        with self.lock:
            if frame is not None:
                # Make a copy of the frame to avoid reference issues
                self.buffer.append(frame.copy())
    
    def get_latest_frame(self):
        with self.lock:
            if not self.buffer:
                return None
            return self.buffer[-1].copy()  # Return a copy of the latest frame


class PositionTracker:
    """
    Class to track and smooth object positions over time
    """
    def __init__(self, smoothing_factor=0.7, history_size=10):
        # Higher smoothing_factor means more weight to historical positions
        self.smoothing_factor = smoothing_factor
        self.history_size = history_size
        # Dict to store objects by ID
        self.tracked_objects = {}
        
    def update_object(self, object_id, class_name, position, confidence, label, spatial_coords):
        """
        Update object position with smoothing
        
        Args:
            object_id (str): Unique identifier for the object
            class_name (str): Class name of the object
            position (tuple): Current position (x, y, z)
            confidence (float): Detection confidence
            label (int): Detection label
            spatial_coords: Original spatial coordinates object
            
        Returns:
            dict: Updated object data
        """
        # Get current time for tracking object age
        current_time = time.time()
        
        # Initialize object if it's new
        if object_id not in self.tracked_objects:
            self.tracked_objects[object_id] = {
                'class_name': class_name,
                'position_history': [],
                'smoothed_position': position,
                'confidence': confidence,
                'first_seen': current_time,
                'last_seen': current_time,
                'label': label,
                'detections_count': 0,
                'box_history': []
            }
        
        # Update existing object
        obj = self.tracked_objects[object_id]
        obj['last_seen'] = current_time
        obj['detections_count'] += 1
        
        # Add position to history
        obj['position_history'].append(position)
        
        # Keep history at specified size
        if len(obj['position_history']) > self.history_size:
            obj['position_history'].pop(0)
        
        # Update confidence (keep the highest confidence seen)
        if confidence > obj['confidence']:
            obj['confidence'] = confidence
            obj['label'] = label
        
        # Calculate smoothed position
        # If this is a new object or has few detections, use less smoothing
        if obj['detections_count'] < 5:
            # Use less smoothing for new objects
            temp_smoothing = self.smoothing_factor * 0.5
        else:
            # Use full smoothing for established objects
            temp_smoothing = self.smoothing_factor
        
        # Apply exponential smoothing to position
        for i in range(3):  # x, y, z
            obj['smoothed_position'][i] = (
                temp_smoothing * obj['smoothed_position'][i] +
                (1 - temp_smoothing) * position[i]
            )
            
        # Add the bounding box to history
        if hasattr(spatial_coords, 'xmin'):
            box = {
                'xmin': spatial_coords.xmin,
                'ymin': spatial_coords.ymin, 
                'xmax': spatial_coords.xmax,
                'ymax': spatial_coords.ymax
            }
            obj['box_history'].append(box)
            
            # Keep box history at specified size
            if len(obj['box_history']) > 5:
                obj['box_history'].pop(0)
        
        return obj
        
    def get_smoothed_box(self, object_id):
        """
        Get smoothed bounding box for an object
        
        Args:
            object_id (str): Unique identifier for the object
            
        Returns:
            dict: Smoothed bounding box coordinates or None
        """
        if object_id not in self.tracked_objects:
            return None
            
        obj = self.tracked_objects[object_id]
        if not obj.get('box_history'):
            return None
            
        # Average the box coordinates
        box_count = len(obj['box_history'])
        if box_count == 0:
            return None
            
        # Calculate weighted average with more recent boxes having higher weight
        smoothed_box = {'xmin': 0, 'ymin': 0, 'xmax': 0, 'ymax': 0}
        total_weight = 0
        
        for i, box in enumerate(obj['box_history']):
            # Weight increases for more recent boxes (i starts at 0)
            weight = i + 1
            total_weight += weight
            
            for key in smoothed_box:
                smoothed_box[key] += box[key] * weight
        
        # Normalize by total weight
        for key in smoothed_box:
            smoothed_box[key] /= total_weight
            
        return smoothed_box
        
    def clean_old_objects(self, max_age_seconds=10):
        """
        Remove objects that haven't been seen for a while
        
        Args:
            max_age_seconds (float): Maximum time since last detection
        """
        current_time = time.time()
        objects_to_remove = []
        
        for object_id, obj in self.tracked_objects.items():
            if current_time - obj['last_seen'] > max_age_seconds:
                objects_to_remove.append(object_id)
                
        for object_id in objects_to_remove:
            del self.tracked_objects[object_id]
            
    def get_all_tracked_objects(self):
        """
        Get all currently tracked objects
        
        Returns:
            dict: All tracked objects with their data
        """
        return self.tracked_objects


def apply_median_z_filter(position_tracker, object_id, current_z, window_size=5):
    """
    Apply a median filter to Z values to reduce noise/flickering.
    
    Args:
        position_tracker: The position tracker object
        object_id: The ID of the object to filter
        current_z: The current Z value from detection
        window_size: Size of the median filter window
        
    Returns:
        float: Filtered Z value
    """
    if object_id not in position_tracker.tracked_objects:
        return current_z
        
    obj = position_tracker.tracked_objects[object_id]
    
    # Get Z values from position history
    z_history = [pos[2] for pos in obj['position_history']]
    
    # If we don't have enough history, use the current Z
    if len(z_history) < 3:
        return current_z
        
    # Add current Z to the list
    z_values = z_history + [current_z]
    
    # Apply median filter
    sorted_z = sorted(z_values)
    median_z = sorted_z[len(sorted_z) // 2]
    
    # For objects with more detections, apply additional smoothing
    if obj['detections_count'] > 10:
        # Use a weighted average between median and current smoothed Z
        current_smoothed_z = obj['smoothed_position'][2]
        filtered_z = 0.7 * current_smoothed_z + 0.3 * median_z
    else:
        # For newer objects, use the median
        filtered_z = median_z
        
    return filtered_z

# Add this function to the position tracker class for convenience:
def PositionTracker_apply_z_filter(self, object_id, current_z):
    """Apply Z-value stabilization filter to an object"""
    return apply_median_z_filter(self, object_id, current_z)

# Monkey-patch this method into the PositionTracker class:
PositionTracker.apply_z_filter = PositionTracker_apply_z_filter


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
    Reset USB device by cutting power to the specific USB port used by the OAK camera
    on a Raspberry Pi, then turning it back on after 3 seconds.
    Performs the power cycle twice for more reliable reset.
    
    Args:
        device_id (str): Optional device ID to match
    
    Returns:
        bool: True if reset was successful, False otherwise
    """
    # Set up logging
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Looking for OAK camera to reset...")
        
        # Step 1: Find the OAK camera device and its USB port
        try:
            # Use lsusb to list all USB devices
            output = subprocess.check_output(["lsusb", "-v"], stderr=subprocess.PIPE).decode("utf-8")
            
            # Look for Luxonis/OAK/DepthAI devices
            matches = []
            
            # Parse the lsusb output to find the device
            bus_device = None
            for line in output.split('\n'):
                # Look for the bus and device IDs
                if "Bus " in line and "Device " in line:
                    bus_device = line.split()[1:4:2]  # Extract bus and device numbers
                
                # Look for Luxonis, DepthAI or OAK identifiers
                if any(x in line for x in ["Luxonis", "DepthAI", "OAK"]) and bus_device:
                    bus = bus_device[0]
                    device = bus_device[1].rstrip(':')
                    matches.append((bus, device))
                    logger.info(f"Found OAK camera at Bus {bus} Device {device}")
            
            # Filter by device_id if provided
            if device_id and matches:
                # Get more detailed information for each device to check MxID
                filtered_matches = []
                for bus, device in matches:
                    try:
                        # Get device details using udevadm
                        cmd = f"udevadm info -q property -n /dev/bus/usb/{bus}/{device}"
                        details = subprocess.check_output(cmd, shell=True).decode("utf-8")
                        if device_id in details:
                            filtered_matches.append((bus, device))
                            logger.info(f"Matched device ID {device_id} at Bus {bus} Device {device}")
                    except Exception as e:
                        logger.error(f"Error getting device details: {str(e)}")
                
                matches = filtered_matches if filtered_matches else matches
            
            if not matches:
                logger.error("No OAK camera found")
                return False
            
            # Use the first match if multiple found
            bus, device = matches[0]
            
            # Step 2: Find the USB port path using the bus and device numbers
            try:
                # Get the USB device path
                cmd = f"udevadm info -q path -n /dev/bus/usb/{bus}/{device}"
                usb_path = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
                
                # Extract the port number from the path
                # The path is like: /devices/platform/soc/3f980000.usb/usb1/1-1/1-1.2
                # We want to extract just the port part: 1-1.2
                port_match = re.search(r'usb\d+/(\d+-\d+(?:\.\d+)*)', usb_path)
                if not port_match:
                    logger.error(f"Could not extract port from path: {usb_path}")
                    return False
                
                port = port_match.group(1)
                logger.info(f"Found USB port: {port}")
                
                # Function to perform a single power cycle
                def perform_power_cycle():
                    # Step 3: Cut power to the USB port
                    # Find the control file for the port
                    hub_path = f"/sys/bus/usb/devices/{port.split('.')[0]}/port{port.split('.')[-1]}"
                    power_control = f"{hub_path}/power/control"
                    
                    # First check if the power control file exists
                    check_cmd = f"ls {power_control}"
                    try:
                        subprocess.check_output(check_cmd, shell=True, stderr=subprocess.PIPE)
                    except subprocess.CalledProcessError:
                        # If the direct method fails, we'll try the alternative method
                        hub_path = f"/sys/bus/usb/devices/{port}"
                        power_control = f"{hub_path}/power/control"
                    
                    # Step 3: Disable and re-enable the device by controlling power
                    logger.info(f"Cutting power to USB port {port}...")
                    
                    # First method: Try using power/control if available
                    try:
                        # Set power mode to auto (will allow power to be cut)
                        subprocess.run(f"echo auto > {power_control}", shell=True, check=True)
                        
                        # Remove power by removing the device
                        subprocess.run(f"echo 0 > {hub_path}/remove", shell=True, check=True)
                        logger.info("Power cut successfully")
                        
                        # Wait 3 seconds
                        time.sleep(3)
                        
                        # Scan for devices to restore power
                        logger.info("Restoring power...")
                        parent_hub = '/'.join(hub_path.split('/')[:-1])
                        subprocess.run(f"echo 1 > {parent_hub}/scan", shell=True, check=True)
                        logger.info("Power restored successfully")
                        
                        # Wait for device to initialize
                        time.sleep(2)
                        return True
                        
                    except Exception as e:
                        logger.error(f"Error using power control: {str(e)}")
                        
                        # Step 4: Alternative method using uhubctl if available
                        try:
                            logger.info("Trying alternative method with uhubctl...")
                            
                            # Check if uhubctl is installed
                            subprocess.check_output(["which", "uhubctl"])
                            
                            # Extract port number for uhubctl format
                            port_nums = port.split('.')
                            root_port = port_nums[0]
                            hub_port = port_nums[-1] if len(port_nums) > 1 else None
                            
                            if hub_port:
                                # Turn off power
                                cmd = f"uhubctl -l {root_port} -p {hub_port} -a 0"
                                subprocess.run(cmd, shell=True, check=True)
                                logger.info(f"Power cut successfully using uhubctl")
                                
                                # Wait 3 seconds
                                time.sleep(3)
                                
                                # Turn power back on
                                cmd = f"uhubctl -l {root_port} -p {hub_port} -a 1"
                                subprocess.run(cmd, shell=True, check=True)
                                logger.info(f"Power restored successfully using uhubctl")
                                
                                # Wait for device to initialize
                                time.sleep(2)
                                return True
                        except Exception as e:
                            logger.error(f"Error using uhubctl method: {str(e)}")
                            
                            # Step 5: Fallback to the bind/unbind method if other methods fail
                            try:
                                logger.info("Falling back to bind/unbind method...")
                                
                                # Get the driver
                                cmd = f"basename $(readlink -f /sys/bus/usb/devices/{port}/driver)"
                                driver = subprocess.check_output(cmd, shell=True).decode().strip()
                                
                                # Unbind
                                unbind_cmd = f"echo '{port}' > /sys/bus/usb/drivers/{driver}/unbind"
                                subprocess.run(unbind_cmd, shell=True, check=True)
                                logger.info("Device unbound successfully")
                                
                                # Wait 3 seconds
                                time.sleep(3)
                                
                                # Bind again
                                bind_cmd = f"echo '{port}' > /sys/bus/usb/drivers/{driver}/bind"
                                subprocess.run(bind_cmd, shell=True, check=True)
                                logger.info("Device bound successfully")
                                
                                # Wait for device to initialize
                                time.sleep(2)
                                return True
                                
                            except Exception as e:
                                logger.error(f"Error using bind/unbind method: {str(e)}")
                    
                    return False

                # Perform first power cycle
                logger.info("Performing first power cycle...")
                if not perform_power_cycle():
                    logger.error("First power cycle failed")
                    return False
                
                # Wait a bit longer between cycles
                logger.info("Waiting between power cycles...")
                time.sleep(2)
                
                # Perform second power cycle
                logger.info("Performing second power cycle...")
                if not perform_power_cycle():
                    logger.error("Second power cycle failed")
                    return False
                
                logger.info("Double power cycle completed successfully")
                return True
            
            except Exception as e:
                logger.error(f"Error finding USB port: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error finding OAK camera: {str(e)}")
        
        logger.error("Could not reset USB device using any available method")
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
    
    # Reset device through USB with double power cycle
    logger.info("Attempting USB device reset with double power cycle")
    usb_reset_success = reset_usb_device()
    
    if not usb_reset_success:
        logger.warning("USB reset failed, will attempt to recreate device anyway")
    
    # Force garbage collection
    gc.collect()
    
    # Wait for device to stabilize (longer wait after double power cycle)
    time.sleep(5)
    
    # Create a new pipeline
    try:
        pipeline = setup_vision_pipeline()
        
        # Check if devices are available
        available_devices = dai.Device.getAllAvailableDevices()
        if not available_devices:
            logger.error("No DepthAI devices found after recovery attempt")
            return False, None, current_time
        
        logger.info(f"Found {len(available_devices)} device(s) after recovery")
        for i, deviceInfo in enumerate(available_devices):
            logger.info(f"Device {i}: {deviceInfo.getMxId()} (state: {deviceInfo.state})")
            
        # Try to create a new device
        logger.info("Creating new device instance")
        
        # Multiple attempts to create device
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                logger.info(f"Device creation attempt {attempt+1}/{max_attempts}")
                new_device = dai.Device(pipeline)
                logger.info("Successfully created new device instance")
                
                # Update status
                status_queue.put({"status": "recovered", "message": "Successfully recovered from error"})
                
                return True, new_device, current_time
            except Exception as e:
                logger.error(f"Device creation attempt {attempt+1} failed: {str(e)}")
                if attempt < max_attempts - 1:
                    logger.info("Waiting before next attempt...")
                    time.sleep(2)
        
        logger.error(f"Failed to create device after {max_attempts} attempts")
        status_queue.put({"status": "error", "message": f"Recovery failed: could not create device after {max_attempts} attempts"})
        return False, None, current_time
        
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
    
    # Create a frame buffer
    frame_buffer = FrameBuffer(max_size=5)
    
    # Create position tracker for smoother object detection
    position_tracker = PositionTracker(smoothing_factor=0.7, history_size=10)
    
    # Add frame processing control variables
    target_fps = 30
    min_frame_interval = 1.0 / target_fps
    last_frame_time = 0
    last_sent_frame_time = 0
    frame_send_interval = 1.0 / 25  # Send frames at 25 FPS max to reduce queue traffic
    
    try:
        # Set up the pipeline
        pipeline = setup_vision_pipeline()
        
        # Initialize status
        status_queue.put({"status": "initializing"})
        
        # Connect to device with improved error handling and retry
        max_retries = 5
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
                                    # Get latest frame from buffer
                                    frozen_frame = frame_buffer.get_latest_frame()
                                    if frozen_frame is None and qVideo.has():
                                        # If buffer is empty, get a fresh frame
                                        current_frame = qVideo.get().getCvFrame()
                                        frozen_frame = cv2.resize(current_frame, (640, 400))
                                    
                                    # Add paused indicator if we have a frame
                                    if frozen_frame is not None:
                                        # Create a copy to avoid reference issues
                                        frozen_frame = frozen_frame.copy()
                                        # Add paused indicator
                                        cv2.putText(frozen_frame, "CAMERA PAUSED", (50, 50), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
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
                        # Also reset position tracker
                        position_tracker = PositionTracker(smoothing_factor=0.7, history_size=10)
                    elif cmd.get("command") == "force_recovery":
                        logger.info("Received force recovery command")
                        # Force device recovery
                        success, device, last_recovery_time = recover_from_error(
                            device, "Manual recovery requested", 0, status_queue
                        )
                        
                        if success and device is not None:
                            # Recreate queues for the new device
                            try:
                                logger.info("Creating new output queues for recovered device")
                                qVideo = device.getOutputQueue(name="video", maxSize=4, blocking=False)
                                qDet = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
                                logger.info("Manual recovery successful")
                                # Reset error counters
                                consecutive_errors = 0
                                status_queue.put({"status": "running", "message": "Camera reconnected after double power cycle"})
                            except Exception as queue_error:
                                logger.error(f"Error creating queues for new device: {str(queue_error)}")
                                status_queue.put({"status": "error", "message": f"Queue creation failed: {str(queue_error)}"})
                        else:
                            logger.error("Manual recovery failed")
                            running = False
                            break
                    elif cmd.get("command") == "status":
                        # Report current status
                        status = "running"
                        if paused:
                            status = "paused"
                        status_queue.put({"status": status})
                
                # Get current time once per loop
                current_time = time.time()
                
                # If paused, send the frozen frame but don't process new frames
                if paused:
                    try:
                        # Send the frozen frame to main process only if it changed or periodically
                        if frozen_frame is not None and (current_time - last_sent_frame_time) >= 0.5:
                            if frame_queue.qsize() < frame_queue._maxsize - 1:
                                frame_queue.put(frozen_frame, block=False)
                                last_sent_frame_time = current_time
                    except Exception as e:
                        logger.warning(f"Error sending frozen frame: {str(e)}")
                        
                    # Empty the input queues to prevent backlog
                    while qVideo.has():
                        qVideo.get()
                    while qDet.has():
                        qDet.get()
                        
                    # Sleep longer while paused to reduce CPU usage
                    time.sleep(0.1)
                    continue
                
                # Process frames with rate limiting and error handling
                try:
                    # Always get detections for tracking when available
                    detections = []
                    if qDet.has():
                        inDet = qDet.get()
                        detections = inDet.detections
                        
                        # Process each detection
                        # Track best detection
                        best_detection = None
                        best_confidence = 0
                        best_object_id = None
                        
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
                            object_id = f"{round(x / 10) * 10}_{round(y / 10) * 10}_{class_name}"
                            
                            # Apply Z-value filter before tracking
                            filtered_z = apply_median_z_filter(position_tracker, object_id, z)
                            
                            # Update the position tracker with this detection
                            position = [x, y, filtered_z]  # Use filtered Z value
                            updated_obj = position_tracker.update_object(
                                object_id, 
                                class_name, 
                                position, 
                                detection.confidence,
                                detection.label,
                                detection
                            )
                            
                            # Use the smoothed position from the tracker
                            smoothed_position = updated_obj['smoothed_position']
                            
                            # Track best detection for picking
                            if detection.confidence > best_confidence:
                                best_confidence = detection.confidence
                                best_object_id = object_id
                                best_detection = {
                                    "coordinates": smoothed_position,  # Use smoothed position
                                    "class_name": class_name,
                                    "confidence": detection.confidence,
                                    "object_id": object_id,
                                    "raw_coordinates": [x, y, z],  # Store raw coordinates for reference
                                    "frame": frame_count,
                                    "stable_count": updated_obj['detections_count']  # Add stability indicator
                                }
                        
                        # Clean up old objects
                        position_tracker.clean_old_objects(max_age_seconds=OBJECT_CLEANUP_TIME)
                        
                        # Send best detection to main process
                        if best_detection:
                            detection_queue.put(best_detection)
                    
                    # Get video frames at the target processing rate
                    should_process_frame = (current_time - last_frame_time) >= min_frame_interval
                    
                    if should_process_frame and qVideo.has():
                        inVideo = qVideo.get()
                        frame = inVideo.getCvFrame()
                        last_frame_time = current_time
                        frame_count += 1
                        
                        # Process the frame
                        # Create a smaller version for display
                        display_frame = cv2.resize(frame, (640, 400))
                        
                        # Add detection visualization to the display frame
                        if len(detections) > 0:
                            for detection in detections:
                                class_name = "Battery" if detection.label == 0 else "CBattery"
                                
                                # Skip if below threshold
                                if class_name not in CLASS_SETTINGS or detection.confidence < CLASS_SETTINGS[class_name]["threshold"]:
                                    continue
                                
                                # Get spatial coordinates for object ID
                                x = detection.spatialCoordinates.x
                                y = detection.spatialCoordinates.y
                                object_id = f"{round(x / 10) * 10}_{round(y / 10) * 10}_{class_name}"
                                
                                # Get color based on class
                                color = CLASS_SETTINGS[class_name]["color"]
                                
                                # Use smoothed bounding box if available
                                smoothed_box = position_tracker.get_smoothed_box(object_id)
                                
                                if smoothed_box:
                                    # Convert normalized coordinates to pixel coordinates
                                    xmin = int(smoothed_box['xmin'] * display_frame.shape[1])
                                    ymin = int(smoothed_box['ymin'] * display_frame.shape[0])
                                    xmax = int(smoothed_box['xmax'] * display_frame.shape[1])
                                    ymax = int(smoothed_box['ymax'] * display_frame.shape[0])
                                else:
                                    # Fall back to original detection box if no smoothed box
                                    xmin = int(detection.xmin * display_frame.shape[1])
                                    ymin = int(detection.ymin * display_frame.shape[0])
                                    xmax = int(detection.xmax * display_frame.shape[1])
                                    ymax = int(detection.ymax * display_frame.shape[0])
                                
                                # Draw rectangle and text
                                cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), color, 2)
                                
                                # Add stability indicator to the label
                                if object_id in position_tracker.tracked_objects:
                                    det_count = position_tracker.tracked_objects[object_id]['detections_count']
                                    confidence = detection.confidence
                                    
                                    # Add stability icon based on detection count
                                    stability_icon = ""
                                    if det_count > 15:
                                        stability_icon = "★"  # Very stable
                                    elif det_count > 8:
                                        stability_icon = "☆"  # Stable
                                    
                                    cv2.putText(display_frame, f"{class_name} {confidence:.2f} {stability_icon}",
                                                (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                else:
                                    cv2.putText(display_frame, f"{class_name} {detection.confidence:.2f}",
                                                (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Add the frame to the buffer
                        frame_buffer.add_frame(display_frame)
                    
                    # Send frames to the main process at a more controlled rate
                    should_send_frame = (current_time - last_sent_frame_time) >= frame_send_interval
                    
                    if should_send_frame:
                        # Get the latest frame from the buffer
                        latest_frame = frame_buffer.get_latest_frame()
                        
                        if latest_frame is not None:
                            # Only send if the queue isn't full
                            if frame_queue.qsize() < frame_queue._maxsize - 1:
                                try:
                                    frame_queue.put(latest_frame, block=False)
                                    last_sent_frame_time = current_time
                                except Exception as e:
                                    logger.warning(f"Error sending frame to main process: {str(e)}")
                    
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
                            # Recreate queues for the new device
                            try:
                                logger.info("Creating new output queues for recovered device")
                                qVideo = device.getOutputQueue(name="video", maxSize=4, blocking=False)
                                qDet = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
                                # Reset error counter
                                consecutive_errors = 0
                                logger.info("Device recovery complete - successfully reconnected")
                                status_queue.put({"status": "running", "message": "Camera reconnected after double power cycle"})
                            except Exception as queue_error:
                                logger.error(f"Error creating queues for new device: {str(queue_error)}")
                                status_queue.put({"status": "error", "message": f"Queue creation failed: {str(queue_error)}"})
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
                
                # Sleep to reduce CPU usage - adaptive based on frame rate
                sleep_time = max(0.001, min_frame_interval - (time.time() - current_time))
                time.sleep(sleep_time)
                
                # Periodic garbage collection 
                if current_time - last_gc_time > 30:  # Reduced from 10 to 30 seconds
                    gc.collect()
                    last_gc_time = current_time
                
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
    frame_queue = multiprocessing.Queue(maxsize=4)  # Increased from 2 to 4 for better buffering
    
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
        
        # Last valid frame for display
        last_valid_frame = None
        
        while time.time() - start_time < run_time:
            if not detection_queue.empty():
                detection = detection_queue.get()
                print(f"Detection: {detection['class_name']} at {detection['coordinates']}")
            
            # Display video feed with improved handling
            if not frame_queue.empty():
                frame = frame_queue.get()
                if frame is not None:
                    last_valid_frame = frame.copy()
            
            if last_valid_frame is not None:
                cv2.imshow("Test Feed", last_valid_frame)
                
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
