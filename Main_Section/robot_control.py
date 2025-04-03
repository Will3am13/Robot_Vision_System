import time
import logging
from pymycobot.genre import Angle, Coord
from pymycobot import MyCobot280
import multiprocessing
from typing import Dict, List, Tuple, Any, Optional

# Import from other modules
from config import (
    MYCOBOT_PORT, BAUDRATE, STANDBY_ANGLES, BATTERY_BIN_COORD, CBATTERY_BIN_COORD,
    PLACE_ORIENTATION, SCREENSHOT_FOLDER
)
from coordinate_transform import (
    transform_point, is_valid_coord, get_current_pick_orientation
)
from mini_cam import take_single_screenshot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RobotControl")


def initialize_robot():
    """
    Initialize and test the robot arm
    
    Returns:
        MyCobot280: Initialized robot object
    """
    logger.info(f"Initializing robot connection on port {MYCOBOT_PORT} at {BAUDRATE} baud")
    
    try:
        # Initialize connection
        mc = MyCobot280(MYCOBOT_PORT, BAUDRATE)
        time.sleep(2)  # Allow time for connection stabilization
        
        # Check connection
        angles = mc.get_angles()
        if angles is None:
            logger.error("Failed to get angles from robot. Check connection.")
            return None
        
        logger.info(f"Robot connected. Current angles: {angles}")
        
        # Initialize gripper
        logger.info("Calibrating gripper...")
        mc.set_gripper_calibration()
        time.sleep(3)
        
        # Test open gripper
        logger.info("Testing gripper (opening)...")
        mc.set_gripper_state(1, 70)
        time.sleep(2)
        
        # Test close gripper
        logger.info("Testing gripper (closing)...")
        mc.set_encoder(7, 4000, 30)
        time.sleep(2)
        
        # Move to standby position
        logger.info(f"Moving to standby position: {STANDBY_ANGLES}")
        mc.send_angles(STANDBY_ANGLES, 30)
        time.sleep(2)
        
        logger.info("Robot initialization complete")
        return mc
        
    except Exception as e:
        logger.error(f"Error initializing robot: {str(e)}")
        return None


def pick_and_place_battery(mc, camera_coords, is_cbattery=False, camera_index=0):
    """
    Pick up a battery at the given coordinates and place it in the appropriate bin
    
    Args:
        mc (MyCobot280): Robot object
        camera_coords (list): Camera coordinates [x, y, z]
        is_cbattery (bool): True if CBattery, False if regular Battery
        camera_index (int): Index of the camera to use for screenshots
        
    Returns:
        bool: True if operation was successful, False otherwise
    """
    battery_type = "CBattery" if is_cbattery else "Battery"
    logger.info(f"Starting pick and place operation for {battery_type}")
    logger.info(f"Camera coordinates: {camera_coords}")
    
    try:
        # Transform camera coordinates to robot coordinates with distance-based settings
        # Now also get the original distance directly from the transform_point function
        robot_xyz, settings, original_distance = transform_point(camera_coords)
        
        # Print information about the coordinates and settings
        logger.info(f"Transformed robot coordinates: {robot_xyz}")
        logger.info(f"Original distance from origin (before offsets): {original_distance:.1f}mm")
        logger.info(f"Range classification: {settings['range']}")
        logger.info(f"Settings: {settings['message']}")
        
        # Get the current pick orientation from coordinate transform module
        current_pick_orientation = get_current_pick_orientation()
        logger.info(f"Pick orientation: {current_pick_orientation}")
        
        # If the object is out of valid range, return with error
        if not settings['valid']:
            logger.error(f"ERROR: {settings['message']}")
            return False
        
        # Create coordinates for hovering position (60 units above the target)
        hover_xyz = robot_xyz.copy()
        hover_xyz[2] += 60  # Add 60 units to the Z coordinate for hovering
        hover_coords = list(hover_xyz) + current_pick_orientation
        
        # Create full 6D coordinates with distance-based orientation for picking
        pick_coords = list(robot_xyz) + current_pick_orientation
        
        # Select the appropriate bin based on battery type
        place_coords = CBATTERY_BIN_COORD if is_cbattery else BATTERY_BIN_COORD
        
        # Validate coordinates are within robot's working range
        if not is_valid_coord(pick_coords) or not is_valid_coord(hover_coords):
            logger.error(f"ERROR: Pick or hover coordinates out of range: {hover_coords} -> {pick_coords}")
            return False
        
        # STEP 1: Move to hover position above the target
        logger.info(f"Moving to hover position {hover_coords} (60 units above target)")
        mc.send_coords(hover_coords, 30, 1)
        time.sleep(2)
        
        # Take a screenshot at hover position (in a separate thread to not block robot operation)
        try:
            logger.info("Taking hover screenshot...")
            screenshot_path = take_single_screenshot(camera_index, SCREENSHOT_FOLDER)
            if screenshot_path:
                logger.info(f"Saved hover screenshot: {screenshot_path}")
            else:
                logger.warning("Failed to save hover screenshot")
        except Exception as e:
            logger.error(f"Error taking screenshot: {str(e)}")
            # Continue with operation even if screenshot fails
        
        # STEP 2: Move down to the actual target position
        logger.info(f"Moving down to pick battery at: {pick_coords}")
        mc.send_coords(pick_coords, 20, 1)  # Slower speed for precision
        time.sleep(2)
        
        # Close gripper to grab the battery
        logger.info("Grabbing battery (closing gripper)...")
        mc.set_gripper_state(1, 70)
        time.sleep(2)

        logger.info(f"Moving to hover position {hover_coords} (60 units above target)")
        mc.send_coords(hover_coords, 30, 1)
        time.sleep(0.5)
        
        # Return to standby position with battery
        logger.info("Returning to standby position...")
        mc.send_angles(STANDBY_ANGLES, 30)
        time.sleep(3)
        
        # Move to the appropriate bin position
        logger.info(f"Moving to {battery_type} bin...")
        mc.send_coords(place_coords, 30, 1)
        time.sleep(3)
        
        # Open gripper to release the battery
        logger.info("Releasing battery (opening gripper)...")
        mc.set_encoder(7, 4000, 30)
        time.sleep(2)
        
        # Return to standby position
        logger.info("Returning to standby position...")
        mc.send_angles(STANDBY_ANGLES, 30)
        time.sleep(2)
        
        logger.info(f"Pick and place operation for {battery_type} completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in pick and place operation: {str(e)}")
        
        # Try to return to safe position in case of error
        try:
            logger.info("Attempting to return to standby position after error...")
            mc.send_angles(STANDBY_ANGLES, 30)
        except Exception as recovery_error:
            logger.error(f"Error in recovery: {str(recovery_error)}")
        
        return False


def run_robot_control(command_queue, status_queue):
    """
    Run the robot control in a separate process
    
    Args:
        command_queue (multiprocessing.Queue): Queue to receive commands from main process
        status_queue (multiprocessing.Queue): Queue to send status updates to main process
    """
    logger.info("Starting robot control process")
    
    try:
        # Initialize robot
        status_queue.put({"status": "initializing"})
        mc = initialize_robot()
        
        if mc is None:
            status_queue.put({"status": "error", "message": "Failed to initialize robot"})
            return
        
        # Update status
        status_queue.put({"status": "ready"})
        logger.info("Robot control is now ready")
        
        running = True
        while running:
            try:
                # Wait for commands from main process
                cmd = command_queue.get()
                
                if cmd.get("command") == "stop":
                    logger.info("Received stop command")
                    running = False
                    break
                    
                elif cmd.get("command") == "pick_and_place":
                    logger.info(f"Received pick and place command: {cmd}")
                    status_queue.put({"status": "busy"})
                    
                    success = pick_and_place_battery(
                        mc,
                        cmd.get("coordinates"),
                        cmd.get("is_cbattery", False),
                        cmd.get("camera_index", 0)
                    )
                    
                    status_queue.put({
                        "status": "ready",
                        "operation_result": success
                    })
                    
                elif cmd.get("command") == "move_to_standby":
                    logger.info("Moving to standby position")
                    status_queue.put({"status": "busy"})
                    
                    mc.send_angles(STANDBY_ANGLES, 30)
                    time.sleep(2)
                    
                    status_queue.put({"status": "ready"})
                
            except Exception as e:
                logger.error(f"Error in robot control process: {str(e)}")
                status_queue.put({"status": "error", "message": str(e)})
                time.sleep(1)  # Prevent tight error loop
        
        # Cleanup
        logger.info("Cleaning up robot control resources")
        
        # Return to neutral position
        logger.info("Returning to neutral position...")
        mc.send_angles([0, 0, 0, 0, 0, 0], 30)
        time.sleep(5)
        
        # Update status
        status_queue.put({"status": "stopped"})
        logger.info("Robot control process stopped")
        
    except Exception as e:
        logger.error(f"Fatal error in robot control process: {str(e)}")
        status_queue.put({"status": "error", "message": str(e)})


def start_robot_control_process():
    """
    Start the robot control in a separate process
    
    Returns:
        tuple: (process, command_queue, status_queue)
    """
    # Create queues for communication between processes
    command_queue = multiprocessing.Queue()
    status_queue = multiprocessing.Queue()
    
    # Create and start the process
    process = multiprocessing.Process(
        target=run_robot_control,
        args=(command_queue, status_queue)
    )
    process.daemon = True  # Process will terminate when main process exits
    process.start()
    
    logger.info(f"Started robot control process (PID: {process.pid})")
    
    return process, command_queue, status_queue


# For testing the robot control as a standalone module
if __name__ == "__main__":
    # Set up multiprocessing
    multiprocessing.set_start_method('spawn')
    
    # Start robot control process
    process, command_queue, status_queue = start_robot_control_process()
    
    try:
        print("Robot control started. Press Ctrl+C to exit.")
        
        # Wait for process to initialize
        status = status_queue.get(timeout=30)
        print(f"Robot status: {status}")
        
        if status["status"] == "ready":
            # Test moving to standby
            print("Moving to standby position...")
            command_queue.put({"command": "move_to_standby"})
            
            # Wait for operation to complete
            status = status_queue.get(timeout=10)
            print(f"Operation result: {status}")
            
            # Wait for user input
            input("Press Enter to continue to next test or Ctrl+C to exit...")
            
            # Test coordinates for pick and place
            test_coords = [50, -100, 400]  # Example camera coordinates
            print(f"Testing pick and place with coordinates: {test_coords}")
            
            command_queue.put({
                "command": "pick_and_place",
                "coordinates": test_coords,
                "is_cbattery": False,
                "camera_index": 0
            })
            
            # Wait for operation to complete
            status = status_queue.get(timeout=60)
            print(f"Operation result: {status}")
        
        # Send stop command
        print("Sending stop command...")
        command_queue.put({"command": "stop"})
        
        # Wait for process to stop
        process.join(timeout=10)
        
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up
        if process.is_alive():
            print("Terminating robot control process...")
            process.terminate()
            process.join()
        
        print("Robot control test complete")
