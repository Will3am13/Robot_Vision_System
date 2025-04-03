import time
import cv2
import numpy as np
import depthai as dai
import gc
import logging
import multiprocessing
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


def run_oak_camera(detection_queue, command_queue, status_queue):
    """
    Run the Oak-D-SR camera in a separate process
    
    Args:
        detection_queue (multiprocessing.Queue): Queue to send detections to main process
        command_queue (multiprocessing.Queue): Queue to receive commands from main process
        status_queue (multiprocessing.Queue): Queue to send status updates to main process
    """
    logger.info("Starting Oak camera process")
    
    # Object tracking data
    class_objects = {
        "Battery": {},
        "CBattery": {}
    }
    
    try:
        # Set up the pipeline
        pipeline = setup_vision_pipeline()
        
        # Initialize status
        status_queue.put({"status": "initializing"})
        
        # Connect to device with error handling and retry
        max_retries = 3
        retry_count = 0
        device = None
        
        while retry_count < max_retries:
            try:
                logger.info(f"Connecting to Oak device (attempt {retry_count + 1}/{max_retries})")
                device = dai.Device(pipeline)
                break
            except Exception as e:
                retry_count += 1
                logger.error(f"Failed to connect to Oak device: {str(e)}")
                if retry_count >= max_retries:
                    status_queue.put({"status": "error", "message": f"Failed to connect to Oak device: {str(e)}"})
                    return
                time.sleep(2)  # Wait before retrying
        
        # Output queues with limited size to prevent memory leaks
        qVideo = device.getOutputQueue(name="video", maxSize=4, blocking=False)
        qDet = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        
        # Update status
        status_queue.put({"status": "running"})
        logger.info("Oak camera is now running")
        
        running = True
        frame_count = 0
        last_gc_time = time.time()
        
        while running:
            try:
                # Check for commands from main process (non-blocking)
                if not command_queue.empty():
                    cmd = command_queue.get()
                    if cmd.get("command") == "stop":
                        logger.info("Received stop command")
                        running = False
                        break
                    elif cmd.get("command") == "reset":
                        logger.info("Received reset command, clearing object tracking data")
                        class_objects = {
                            "Battery": {},
                            "CBattery": {}
                        }
                
                # Get frames and detections
                inVideo = qVideo.get()
                inDet = qDet.get()
                
                frame = inVideo.getCvFrame()
                detections = inDet.detections
                
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
        tuple: (process, detection_queue, command_queue, status_queue)
    """
    # Create queues for communication between processes
    detection_queue = multiprocessing.Queue()
    command_queue = multiprocessing.Queue()
    status_queue = multiprocessing.Queue()
    
    # Create and start the process
    process = multiprocessing.Process(
        target=run_oak_camera,
        args=(detection_queue, command_queue, status_queue)
    )
    process.daemon = True  # Process will terminate when main process exits
    process.start()
    
    logger.info(f"Started Oak camera process (PID: {process.pid})")
    
    return process, detection_queue, command_queue, status_queue


# For testing the Oak camera as a standalone module
if __name__ == "__main__":
    # Set up multiprocessing
    multiprocessing.set_start_method('spawn')
    
    # Start Oak camera process
    process, detection_queue, command_queue, status_queue = start_oak_camera_process()
    
    try:
        print("Oak camera started. Press Ctrl+C to exit.")
        
        # Wait for process to initialize
        status = status_queue.get(timeout=10)
        print(f"Oak camera status: {status}")
        
        # Monitor detections for a while
        start_time = time.time()
        while time.time() - start_time < 30:  # Run for 30 seconds
            if not detection_queue.empty():
                detection = detection_queue.get()
                print(f"Detection: {detection['class_name']} at {detection['coordinates']}")
            
            time.sleep(0.1)
        
        # Send stop command
        print("Sending stop command...")
        command_queue.put({"command": "stop"})
        
        # Wait for process to stop
        process.join(timeout=5)
        
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up
        if process.is_alive():
            print("Terminating Oak camera process...")
            process.terminate()
            process.join()
        
        print("Oak camera test complete")
