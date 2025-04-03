# main.py - Entry point that manages all processes and UI

import time
import os
import cv2
import multiprocessing
import multiprocessing.resource_tracker
import signal
import logging
import queue
import gc
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

# Import from other modules
from config import (
    SCREENSHOT_FOLDER, CLASS_SETTINGS, COOLDOWN_TIME,
    DISTANCE_SETTINGS
)
from oak_camera import start_oak_camera_process
from robot_control import start_robot_control_process

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("battery_sorting.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Main")


def init_system():
    """
    Initialize system components
    
    Returns:
        tuple: Various process objects and communication queues
    """
    logger.info("Initializing system")
    
    # Create screenshot folder if it doesn't exist
    if not os.path.exists(SCREENSHOT_FOLDER):
        os.makedirs(SCREENSHOT_FOLDER)
        logger.info(f"Created screenshot folder: {SCREENSHOT_FOLDER}")
    
    # Set up multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    # Start Oak camera process
    oak_process, detection_queue, oak_cmd_queue, oak_status_queue, frame_queue = start_oak_camera_process()
    
    # Start robot control process
    robot_process, robot_cmd_queue, robot_status_queue = start_robot_control_process()
    
    # Wait for processes to start with increased timeouts and proper status checking
    oak_running = False
    robot_ready = False
    max_wait_time = 120  # Increased timeout to 120 seconds total
    start_time = time.time()
    
    logger.info("Waiting for system components to initialize...")
    
    try:
        while not (oak_running and robot_ready) and (time.time() - start_time < max_wait_time):
            # Check Oak camera status (non-blocking)
            try:
                if not oak_status_queue.empty():
                    oak_status = oak_status_queue.get_nowait()
                    logger.info(f"Oak camera status update: {oak_status}")
                    if oak_status.get("status") == "running":
                        oak_running = True
            except Exception as e:
                logger.warning(f"Error checking Oak status: {str(e)}")
            
            # Check robot status (non-blocking)
            try:
                if not robot_status_queue.empty():
                    robot_status = robot_status_queue.get_nowait()
                    logger.info(f"Robot status update: {robot_status}")
                    if robot_status.get("status") == "ready":
                        robot_ready = True
            except Exception as e:
                logger.warning(f"Error checking robot status: {str(e)}")
            
            # If not ready yet, wait a bit
            if not (oak_running and robot_ready):
                time.sleep(1)
                
                # Periodically log waiting status
                elapsed = time.time() - start_time
                if elapsed % 10 < 0.1:  # Log approximately every 10 seconds
                    logger.info(f"Still waiting for initialization: Oak running: {oak_running}, Robot ready: {robot_ready}, Elapsed: {elapsed:.1f}s")
        
        # Check if timeout occurred
        if not (oak_running and robot_ready):
            logger.error(f"Timeout after {max_wait_time}s waiting for system components to initialize")
            # Clean up
            shutdown_system(oak_process, oak_cmd_queue, frame_queue, robot_process, robot_cmd_queue)
            return None
            
        # Both components are ready
        logger.info("All system components initialized successfully")
        
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        # Clean up
        shutdown_system(oak_process, oak_cmd_queue, frame_queue, robot_process, robot_cmd_queue)
        return None
    
    logger.info("System initialization complete")
    return (
        oak_process, detection_queue, oak_cmd_queue, oak_status_queue, frame_queue,
        robot_process, robot_cmd_queue, robot_status_queue
    )


def shutdown_system(oak_process, oak_cmd_queue, frame_queue, robot_process, robot_cmd_queue):
    """
    Shutdown all system components
    
    Args:
        oak_process: Oak camera process
        oak_cmd_queue: Oak command queue
        frame_queue: Frame queue from Oak camera
        robot_process: Robot control process
        robot_cmd_queue: Robot command queue
    """
    logger.info("Shutting down system")
    
    # Close all queues to avoid semaphore leaks
    def close_queue_safely(q):
        try:
            if q is not None:
                q.close()
                logger.debug("Queue closed successfully")
        except Exception as e:
            logger.warning(f"Error closing queue: {str(e)}")
    
    # Send stop commands to processes
    if oak_process.is_alive():
        logger.info("Sending stop command to Oak camera")
        try:
            oak_cmd_queue.put({"command": "stop"}, timeout=2)
        except Exception as e:
            logger.warning(f"Error sending stop command to Oak camera: {str(e)}")
            
        oak_process.join(timeout=5)
        
        if oak_process.is_alive():
            logger.warning("Oak camera process did not stop gracefully, terminating")
            oak_process.terminate()
            # Additional cleanup - kill if terminate doesn't work
            try:
                if oak_process.is_alive():
                    logger.warning("Process still alive after terminate, killing...")
                    import signal
                    os.kill(oak_process.pid, signal.SIGKILL)
            except Exception as e:
                logger.error(f"Error killing process: {str(e)}")
            oak_process.join(timeout=1)
    
    if robot_process.is_alive():
        logger.info("Sending stop command to robot control")
        try:
            robot_cmd_queue.put({"command": "stop"}, timeout=2)
        except Exception as e:
            logger.warning(f"Error sending stop command to robot control: {str(e)}")
            
        robot_process.join(timeout=10)
        
        if robot_process.is_alive():
            logger.warning("Robot control process did not stop gracefully, terminating")
            robot_process.terminate()
            # Additional cleanup - kill if terminate doesn't work
            try:
                if robot_process.is_alive():
                    logger.warning("Process still alive after terminate, killing...")
                    import signal
                    os.kill(robot_process.pid, signal.SIGKILL)
            except Exception as e:
                logger.error(f"Error killing process: {str(e)}")
            robot_process.join(timeout=1)
    
    # Close all queues
    logger.info("Closing queues...")
    close_queue_safely(oak_cmd_queue)
    close_queue_safely(robot_cmd_queue)
    close_queue_safely(frame_queue)
    
    # Final garbage collection
    gc.collect()
    
    # Explicitly clear shared memory
    try:
        multiprocessing.resource_tracker._resource_tracker.clear()
        logger.info("Cleared multiprocessing resources")
    except Exception as e:
        logger.warning(f"Error clearing multiprocessing resources: {str(e)}")
    
    logger.info("System shutdown complete")


# Removed create_status_window function since we now use update_combined_frame

def update_combined_frame(combined_frame, video_frame=None, best_detection=None, auto_mode=False, robot_status="unknown", oak_status="unknown"):
    """
    Update the combined frame with both status information and video feed
    
    Args:
        combined_frame (np.ndarray): Combined frame to update
        video_frame (np.ndarray): Current video frame from Oak camera
        best_detection (dict): Best detection information
        auto_mode (bool): Whether auto mode is enabled
        robot_status (str): Current robot status
        oak_status (str): Current Oak camera status
        
    Returns:
        np.ndarray: Updated combined frame
    """
    # Clear the frame
    combined_frame.fill(0)
    
    # Define layout regions
    video_region = combined_frame[50:450, 50:690]  # 640x400 video region
    status_region = combined_frame[50:450, 720:950]  # Status information region
    info_region = combined_frame[500:750, 50:950]  # Bottom information region
    
    # Add title
    cv2.putText(combined_frame, "Battery Sorting System", (50, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Add video feed if available
    if video_frame is not None:
        # Make sure video_frame is the right size
        if video_frame.shape[:2] != (400, 640):
            video_frame = cv2.resize(video_frame, (640, 400))
        
        # Copy video frame to video region
        video_region[:video_frame.shape[0], :video_frame.shape[1]] = video_frame
    else:
    
    # Draw border around video region
    cv2.rectangle(combined_frame, (49, 49), (691, 451), (100, 100, 100), 1)
    
    # Add mode and status information to status region
    mode_text = "AUTO MODE" if auto_mode else "MANUAL MODE"
    cv2.putText(status_region, mode_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.putText(status_region, f"Robot: {robot_status}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.putText(status_region, f"Camera: {oak_status}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Add best detection information
    if best_detection:
        class_name = best_detection.get("class_name", "Unknown")
        confidence = best_detection.get("confidence", 0)
        coords = best_detection.get("coordinates", [0, 0, 0])
        
        color = CLASS_SETTINGS.get(class_name, {}).get("color", (255, 255, 255))
        
        cv2.putText(status_region, f"Detection: {class_name}", 
                    (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.putText(status_region, f"Confidence: {confidence:.2f}", 
                    (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        cv2.putText(status_region, f"X: {coords[0]:.1f}mm", 
                    (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        cv2.putText(status_region, f"Y: {coords[1]:.1f}mm", 
                    (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        cv2.putText(status_region, f"Z: {coords[2]:.1f}mm", 
                    (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    else:
        cv2.putText(status_region, "No valid detections", (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    
    # Draw border around status region
    cv2.rectangle(combined_frame, (719, 49), (951, 451), (100, 100, 100), 1)
    
    # Add instructions to info region
    cv2.putText(info_region, "Controls:", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.putText(info_region, "p: pick object  a: toggle auto mode  q: quit", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Add distance ranges info
    cv2.putText(info_region, "Distance Ranges:", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    
    y_pos = 130
    range_info = [
        f"<{DISTANCE_SETTINGS['too_close']['max_distance']}mm: Too close",
        f"{DISTANCE_SETTINGS['short_range']['min_distance']}-{DISTANCE_SETTINGS['short_range']['max_distance']}mm: Short range",
        f"{DISTANCE_SETTINGS['normal_range']['min_distance']}-{DISTANCE_SETTINGS['normal_range']['max_distance']}mm: Normal range",
        f"{DISTANCE_SETTINGS['long_range']['min_distance']}-{DISTANCE_SETTINGS['long_range']['max_distance']}mm: Long range",
        f">{DISTANCE_SETTINGS['too_far']['min_distance']}mm: Too far"
    ]
    
    for info in range_info:
        cv2.putText(info_region, info, (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_pos += 25
    
    # Add system metrics
    mem_usage = get_memory_usage_mb()
    cpu_usage = get_cpu_usage()
    
    cv2.putText(info_region, f"Memory: {mem_usage:.1f} MB | CPU: {cpu_usage}% | Screenshots: {SCREENSHOT_FOLDER}", 
                (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    # Draw border around info region
    cv2.rectangle(combined_frame, (49, 499), (951, 751), (100, 100, 100), 1)
    
    return combined_frame


def get_memory_usage_mb():
    """Get current process memory usage in MB"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # Convert to MB
    except ImportError:
        return 0  # psutil not installed


def get_cpu_usage():
    """Get current CPU usage"""
    try:
        import psutil
        return psutil.cpu_percent()
    except ImportError:
        return 0  # psutil not installed


def main():
    """Main application entry point"""
    logger.info("Starting Battery Sorting System")
    
    # Initialize system
    system_info = init_system()
    if system_info is None:
        logger.error("System initialization failed")
        return
    
    (
        oak_process, detection_queue, oak_cmd_queue, oak_status_queue, frame_queue,
        robot_process, robot_cmd_queue, robot_status_queue
    ) = system_info
    
    # Create a combined window for status and video feed
    cv2.namedWindow("Battery Sorting System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Battery Sorting System", 1000, 800)
    
    # Initialize a combined frame
    combined_frame = np.zeros((800, 1000, 3), dtype=np.uint8)
    
    # Tracking variables
    last_processed_time = 0
    auto_mode = False
    best_detection = None
    robot_status = "ready"
    oak_status = "running"
    last_error_time = 0
    
    # Signal handler for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Received interrupt signal, shutting down")
        shutdown_system(oak_process, oak_cmd_queue, frame_queue, robot_process, robot_cmd_queue)
        cv2.destroyAllWindows()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    running = True
    while running:
        try:
            # Check for Oak status updates (non-blocking)
            try:
                while not oak_status_queue.empty():
                    oak_status_update = oak_status_queue.get_nowait()
                    new_oak_status = oak_status_update.get("status", oak_status)
                    
                    # Check for communication errors
                    if new_oak_status == "error":
                        error_message = oak_status_update.get("message", "Unknown error")
                        logger.error(f"Oak camera error: {error_message}")
                        
                        # Check if it's a communication error
                        if ("Communication exception" in error_message or 
                            "X_LINK_ERROR" in error_message) and time.time() - last_error_time > 60:
                            logger.info("Detected communication error - initiating recovery")
                            oak_cmd_queue.put({"command": "force_recovery"})
                            last_error_time = time.time()
                    
                    # Update status
                    oak_status = new_oak_status
                    
            except Exception as e:
                logger.error(f"Error checking Oak status: {str(e)}")
            
            # Check for robot status updates (non-blocking)
            try:
                while not robot_status_queue.empty():
                    robot_status_update = robot_status_queue.get_nowait()
                    new_robot_status = robot_status_update.get("status", robot_status)
                    
                    # If robot status changed, handle camera pausing/resuming
                    if new_robot_status != robot_status:
                        logger.info(f"Robot status changed from {robot_status} to {new_robot_status}")
                        
                        # If robot is now busy, pause the camera
                        if new_robot_status == "busy" and robot_status != "busy":
                            logger.info("Robot is busy. Pausing camera.")
                            oak_cmd_queue.put({"command": "pause"})
                        
                        # If robot is now ready (and was previously busy), resume the camera
                        elif new_robot_status == "ready" and robot_status == "busy":
                            logger.info("Robot is ready. Resuming camera.")
                            oak_cmd_queue.put({"command": "resume"})
                    
                    # Update status
                    robot_status = new_robot_status
                    
                    if robot_status == "error":
                        logger.error(f"Robot error: {robot_status_update.get('message', 'Unknown error')}")
            except Exception as e:
                logger.error(f"Error checking robot status: {str(e)}")
            
            # Get latest detection (non-blocking)
            try:
                while not detection_queue.empty():
                    new_detection = detection_queue.get_nowait()
                    best_detection = new_detection  # Always use the latest detection
            except Exception as e:
                logger.error(f"Error getting detection: {str(e)}")
            
            # Get video frame if available
            video_frame = None
            try:
                if not frame_queue.empty():
                    video_frame = frame_queue.get_nowait()
            except Exception as e:
                logger.warning(f"Error getting video frame: {str(e)}")
            
            # Update combined frame
            combined_frame = update_combined_frame(
                combined_frame, video_frame, best_detection, auto_mode, robot_status, oak_status
            )
            cv2.imshow("Battery Sorting System", combined_frame)
            
            # Handle key presses
            key = cv2.waitKey(1)
            
            # Toggle auto mode
            if key == ord('a'):
                auto_mode = not auto_mode
                logger.info(f"Auto mode {'enabled' if auto_mode else 'disabled'}")
            
            # Force camera recovery
            elif key == ord('r'):
                logger.info("User requested camera recovery")
                oak_cmd_queue.put({"command": "force_recovery"})
            
            # Quit on 'q' key
            elif key == ord('q'):
                logger.info("User requested exit")
                running = False
                break
            
            # Check if we should process a battery
            current_time = time.time()
            should_process = (
                (key == ord('p') and robot_status == "ready") or
                (auto_mode and best_detection and current_time - last_processed_time > COOLDOWN_TIME and robot_status == "ready")
            )
            
            # Process the best detection if needed
            if should_process and best_detection:
                battery_type = best_detection["class_name"]
                is_cbattery = (battery_type == "CBattery")
                
                logger.info(f"Processing {battery_type} (confidence: {best_detection['confidence']:.2f})")
                
                # Delay
                time.sleep(3)
                
                # Send pick and place command to robot process
                robot_cmd_queue.put({
                    "command": "pick_and_place",
                    "coordinates": best_detection["coordinates"],
                    "is_cbattery": is_cbattery,
                    "camera_index": 0
                })
                
                # Update tracking variables
                last_processed_time = current_time
                
                # Reset best detection after sending to robot
                best_detection = None
            
            # Sleep to reduce CPU usage
            time.sleep(0.01)
            
            # Periodic garbage collection
            if current_time % 30 < 0.1:  # Every ~30 seconds
                gc.collect()
            
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            time.sleep(1)  # Prevent tight error loop
    
    # Shutdown system
    logger.info("Shutting down Battery Sorting System")
    shutdown_system(oak_process, oak_cmd_queue, frame_queue, robot_process, robot_cmd_queue)
    cv2.destroyAllWindows()
    logger.info("Shutdown complete")


def cleanup_multiprocessing():
    """
    Clean up multiprocessing resources including leaked semaphores
    """
    try:
        # Clear process resources
        multiprocessing.resource_tracker._resource_tracker.clear()
        logger.info("Cleared multiprocessing resources")
        
        # For Python 3.8+ specific issue with leaked semaphores
        try:
            # Access private _fd2ref to find and clear leaked semaphores
            if hasattr(multiprocessing.resource_tracker._resource_tracker, "_fd2ref"):
                leaked_sems = list(multiprocessing.resource_tracker._resource_tracker._fd2ref.items())
                logger.info(f"Found {len(leaked_sems)} resource tracker references")
                
                for fd, (resource_type, _) in leaked_sems:
                    if resource_type == "semaphore":
                        logger.info(f"Cleaning up semaphore with fd {fd}")
                        try:
                            multiprocessing.resource_tracker._resource_tracker._remove_resource(
                                "semaphore", fd
                            )
                        except Exception as e:
                            logger.warning(f"Error removing semaphore {fd}: {str(e)}")
        except Exception as e:
            logger.warning(f"Error cleaning up leaked semaphores: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error in cleanup_multiprocessing: {str(e)}")


if __name__ == "__main__":
    try:
        # Check for pre-existing leaked resources
        try:
            cleanup_multiprocessing()
        except Exception as e:
            logger.warning(f"Pre-cleanup warning: {str(e)}")
            
        # Set spawn method explicitly at program start
        multiprocessing.set_start_method('spawn', force=True)
        
        # Run main application
        main()
        
        # Final cleanup
        cleanup_multiprocessing()
        
    except Exception as e:
        logger.critical(f"Unhandled exception in main: {str(e)}", exc_info=True)
        
        try:
            # Ensure cleanup happens even after exceptions
            cleanup_multiprocessing()
        except:
            pass
            
        # Exit with error code
        import sys
        sys.exit(1)
