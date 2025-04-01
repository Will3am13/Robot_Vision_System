import time
import os
import sys
import numpy as np
import depthai as dai
import cv2
from pymycobot.genre import Angle, Coord
from pymycobot import MyCobot280
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Define a standby (home) position
STANDBY_COORDS = [111.7, -53.4, 251.2, 172.67, -9.25, 46.06]
STANDBY_ANGLES = [0.52, 23.9, -75.93, -18.45, 0.08, -131.48]

# Bin locations (fixed positions for sorting)
BATTERY_BIN_COORD = [194.0, -142.8, 144.9, 152.06, -20.98, 1.67]  # Regular battery bin
CBATTERY_BIN_COORD = [99.8, -186.7, 162.6, 172.39, -2.74, -14.16]  # CBattery bin

# Fixed orientation angles for picking and placing
PICK_ORIENTATION = [-142, 30, 59]  # Fixed pitch, roll, yaw for picking
PLACE_ORIENTATION = [152.06, -20.98, 1.67]  # Fixed pitch, roll, yaw for placing

# Default coordinate offsets (can be adjusted during runtime)
X_OFFSET = 75  # mm offset in X direction
Y_OFFSET = 10  # mm offset in Y direction
Z_OFFSET = -20  # mm offset in Z direction


# Linear Ridge transformation setup (replacing polynomial method)
def transform_camera_to_robot(eye_coords, hand_coords):
    """
    Transforms camera coordinates to robot coordinates using a linear regression model.

    Args:
        eye_coords (np.array): Array of camera coordinates (N x 3).
        hand_coords (np.array): Array of corresponding robot coordinates (N x 3).

    Returns:
        function: A function that takes camera coordinates and returns transformed robot coordinates.
    """
    # Train separate Ridge regression models for each axis (x, y, z)
    model_x = Ridge(alpha=1.0)
    model_y = Ridge(alpha=1.0)
    model_z = Ridge(alpha=1.0)

    model_x.fit(eye_coords, hand_coords[:, 0])
    model_y.fit(eye_coords, hand_coords[:, 1])
    model_z.fit(eye_coords, hand_coords[:, 2])

    def transform(camera_points):
        """
        Transforms camera points to robot points using the trained models.

        Args:
            camera_points (np.array): Array of camera coordinates to transform (M x 3).

        Returns:
            np.array: Array of transformed robot coordinates (M x 3).
        """
        # Ensure camera_points is a 2D array
        if camera_points.ndim == 1:
            camera_points = camera_points.reshape(1, -1)

        transformed_x = model_x.predict(camera_points)
        transformed_y = model_y.predict(camera_points)
        transformed_z = model_z.predict(camera_points)

        return np.column_stack((transformed_x, transformed_y, transformed_z))

    return transform


# Example calibration data (replace with your actual data if needed)
hand_coords = np.array([
    [98.3, 176.4, 101.4],
    [246.0, 150.3, 101.2],
    [246.0, 150.3, 101.2],
    # [120.0, 16.5, 94.6], * # Really Close to robotic base
    [98.7, 150.8, 102.6],
    [239.8, -56.2, 99.5],
    [141.3, -82.5, 104.5],
    [139.3, -71.0, 103.0],
    [37.1, 193.6, 99.8],
    [90.5, 99.8, 104.5],
    [206.9, 66.0, 176.6],
    [65.4, 155.4, 176.5],
    [182.2, -42.8, 146.5],
    [196.1, 163.6, 142.1],
    [262.9, -88.5, 96.0],
    [197.1, 14.7, 103.6]
])

eye_coords = np.array([
    [1, -112, 343],
    [-55, -81, 480],
    [71, -74, 510],
    # [149, -94, 436]*, # Really Close to robotic base
    [24, -108, 358],
    [122.5, -61, 597],
    [203, -73.5, 539],
    [197, -76, 520],
    [23, -111, 283],
    [70, -105, 384],
    [39.5, -7, 474],
    [35, -43.5, 316],
    [149, -29.5, 530],
    [-35.5, -51.5, 416],
    [138, -51, 625],
    [86.5, -74.5, 514.5]
])

# Create the transformation function (updated to use Ridge method)
transform_func = transform_camera_to_robot(eye_coords, hand_coords)


# Function to adjust offsets during runtime
def adjust_offsets(x_offset=None, y_offset=None, z_offset=None):
    """
    Adjust the coordinate offsets for robot movement.

    Args:
        x_offset (float, optional): Offset in mm for X coordinate. None means no change.
        y_offset (float, optional): Offset in mm for Y coordinate. None means no change.
        z_offset (float, optional): Offset in mm for Z coordinate. None means no change.

    Returns:
        tuple: Current (x_offset, y_offset, z_offset) values after adjustment
    """
    global X_OFFSET, Y_OFFSET, Z_OFFSET

    if x_offset is not None:
        X_OFFSET = float(x_offset)
    if y_offset is not None:
        Y_OFFSET = float(y_offset)
    if z_offset is not None:
        Z_OFFSET = float(z_offset)

    print(f"Current offsets: X={X_OFFSET}mm, Y={Y_OFFSET}mm, Z={Z_OFFSET}mm")
    return (X_OFFSET, Y_OFFSET, Z_OFFSET)


# Function to transform coordinates from camera to robot space (updated with offsets)
def transform_point(cam_point):
    """Transform point from camera coordinates to robot coordinates using Ridge regression and apply offsets"""
    # Ensure cam_point is a numpy array
    cam_point_np = np.array(cam_point)
    # Apply transformation using the ridge regression model
    result = transform_func(cam_point_np.reshape(1, -1))

    # Apply offsets to the coordinates
    result[0][0] += X_OFFSET
    result[0][1] += Y_OFFSET
    result[0][2] += Z_OFFSET

    # Enforce minimum Z value of 105
    if result[0][2] < 65:
        result[0][2] = 65

    # Return the result as a 1D array
    return result[0]


# Function to check if coordinate values are within the safe working range
def is_valid_coord(coord):
    x, y, z, rx, ry, rz = coord
    if not (-281.45 <= x <= 281.45):  # x range
        print("out of range in x")
        return False
    if not (-281.45 <= y <= 281.45):  # y range
        print("out of range in y")
        return False
    if not (-70 <= z <= 412.67):  # z range
        print("out of range in z")
        return False
    if not (-180 <= rx <= 180):  # rx range
        print("out of range in roll")
        return False
    if not (-180 <= ry <= 180):  # ry range
        print("out of range in pitch")
        return False
    if not (-180 <= rz <= 180):  # rz range
        print("out of range in yaww!")
        return False
    return True


def setup_vision_pipeline():
    """Setup and configure the DepthAI vision pipeline"""
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
    spatialDetectionNetwork.setBlobPath(r"/home/er/Downloads/BatteryV2.blob")
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

    return pipeline


def initialize_robot():
    """Initialize and test the robot arm"""
    # Initialize connection
    MYCOBOT_PORT = "/dev/ttyAMA0"
    BAUDRATE = 1000000

    mc = MyCobot280(MYCOBOT_PORT, BAUDRATE)
    time.sleep(2)  # Allow time for connection stabilization

    # Initialize gripper
    print("Calibrating gripper...")
    mc.set_gripper_calibration()
    time.sleep(3)

    # Test open gripper
    print("Testing gripper (opening)...")
    mc.set_gripper_state(1, 70)
    time.sleep(2)

    # Test close gripper
    print("Testing gripper (closing)...")
    mc.set_encoder(7, 4000, 30)
    time.sleep(2)

    # Move to standby position
    print("Moving to standby position...")
    mc.send_angles(STANDBY_ANGLES, 30)
    time.sleep(2)

    return mc


def pick_and_place_battery(mc, camera_coords, is_cbattery=False):
    """Pick up a battery at the given coordinates and place it in the appropriate bin"""
    # Transform camera coordinates to robot coordinates
    robot_xyz = transform_point(camera_coords)
    print(f"Camera coordinates: {camera_coords}")
    print(f"Transformed robot coordinates: {robot_xyz}")
    print(f"Applied offsets: X={X_OFFSET}mm, Y={Y_OFFSET}mm, Z={Z_OFFSET}mm")

    # Create coordinates for hovering position (60 units above the target)
    hover_xyz = robot_xyz.copy()
    hover_xyz[2] += 60  # Add 60 units to the Z coordinate for hovering
    hover_coords = list(hover_xyz) + PICK_ORIENTATION

    # Create full 6D coordinates with fixed orientation for picking
    pick_coords = list(robot_xyz) + PICK_ORIENTATION

    # Select the appropriate bin based on battery type
    place_coords = CBATTERY_BIN_COORD if is_cbattery else BATTERY_BIN_COORD

    # Validate coordinates are within robot's working range
    if not is_valid_coord(pick_coords) or not is_valid_coord(hover_coords):
        print(f"ERROR: Pick or hover coordinates out of range: {hover_coords} -> {pick_coords}")
        return False

    # STEP 1: Move to hover position above the target
    print(f"Moving to hover position {hover_coords} (60 units above target)")
    mc.send_coords(hover_coords, 30, 1)
    time.sleep(2)

    # STEP 2: Move down to the actual target position
    print(f"Moving down to pick battery at: {pick_coords}")
    mc.send_coords(pick_coords, 20, 1)  # Slower speed for precision
    time.sleep(2)

    # Close gripper to grab the battery
    print("Grabbing battery (closing gripper)...")
    mc.set_gripper_state(1, 70)
    time.sleep(2)

    # Return to standby position with battery
    print("Returning to standby position...")
    mc.send_angles(STANDBY_ANGLES, 30)
    time.sleep(3)

    # Move to the appropriate bin position
    print(f"Moving to {'CBattery' if is_cbattery else 'Battery'} bin...")
    mc.send_coords(place_coords, 30, 1)
    time.sleep(3)

    # Open gripper to release the battery
    print("Releasing battery (opening gripper)...")
    mc.set_encoder(7, 4000, 30)
    time.sleep(2)

    # Return to standby position
    print("Returning to standby position...")
    mc.send_angles(STANDBY_ANGLES, 30)
    time.sleep(2)

    return True


def main():
    # Initialize robot
    mc = initialize_robot()

    # Setup vision pipeline
    pipeline = setup_vision_pipeline()

    # Class-specific settings
    class_settings = {
        "Battery": {
            "color": (0, 255, 0),  # Green for Battery
            "threshold": 0.5,  # Detection threshold for Battery
            "objects": {}  # Dictionary to store multiple objects of this class
        },
        "CBattery": {
            "color": (255, 0, 0),  # Blue for CBattery
            "threshold": 0.1,  # Detection threshold for CBattery
            "objects": {}  # Dictionary to store multiple objects of this class
        }
    }

    # Function to calculate average Z value for a specific object
    def get_avg_z(class_name, object_id):
        if class_name not in class_settings:
            return None
        if object_id not in class_settings[class_name]["objects"]:
            return None

        z_history = class_settings[class_name]["objects"][object_id]["z_history"]
        if not z_history:
            return None
        return sum(z_history) / len(z_history)

    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:
        # Output queues
        qVideo = device.getOutputQueue(name="video", maxSize=4, blocking=False)
        qDet = device.getOutputQueue(name="detections", maxSize=4, blocking=False)

        print("Vision system running. Press 'p' to pick up detected battery, 'a' to enable auto-mode, 'q' to quit.")
        print("Offset controls: x/z: X offset, c/v: Y offset, b/n: Z offset, r: reset offsets")

        # Tracking variables
        last_processed_time = 0
        cooldown_time = 10  # Seconds between auto-processing
        auto_mode = False

        while True:
            # Get frames and detections
            inVideo = qVideo.get()
            inDet = qDet.get()

            frame = inVideo.getCvFrame()
            detections = inDet.detections

            # Track best detection
            best_detection = None
            best_confidence = 0

            # Process each detection
            for detection in detections:
                # Determine class name
                class_name = "Battery" if detection.label == 0 else "CBattery"

                # Skip if below class-specific threshold
                if class_name not in class_settings or detection.confidence < class_settings[class_name]["threshold"]:
                    continue

                # Get bounding box coordinates
                xmin, ymin = int(detection.xmin * frame.shape[1]), int(detection.ymin * frame.shape[0])
                xmax, ymax = int(detection.xmax * frame.shape[1]), int(detection.ymax * frame.shape[0])

                # Get spatial coordinates (in millimeters)
                x = detection.spatialCoordinates.x
                y = detection.spatialCoordinates.y
                z = detection.spatialCoordinates.z

                # Create a unique object identifier based on its approximate position in 3D space
                # Round position to nearest 10mm to account for small movements
                object_id = f"{round(x / 10) * 10}_{round(y / 10) * 10}"

                # Initialize object data if this is a new object
                if object_id not in class_settings[class_name]["objects"]:
                    class_settings[class_name]["objects"][object_id] = {
                        "z_history": [],
                        "last_seen": time.time()
                    }

                # Update last seen time
                class_settings[class_name]["objects"][object_id]["last_seen"] = time.time()

                # Update Z history for this specific object (keep only last 5 values)
                class_settings[class_name]["objects"][object_id]["z_history"].append(z)
                if len(class_settings[class_name]["objects"][object_id]["z_history"]) > 50:
                    class_settings[class_name]["objects"][object_id]["z_history"].pop(0)

                # Calculate average Z for this specific object
                avg_z = get_avg_z(class_name, object_id)

                # Draw bounding box and information
                color = class_settings[class_name]["color"]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(frame, f"{class_name} {detection.confidence:.2f}",
                            (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Display spatial coordinates with both raw and averaged Z
                cv2.putText(frame, f"X: {x:.0f}mm  Y: {y:.0f}mm  Z: {z:.0f}mm",
                            (xmin, ymin - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if avg_z is not None:
                    cv2.putText(frame,
                                f"Avg Z: {avg_z:.0f}mm ({len(class_settings[class_name]['objects'][object_id]['z_history'])}/5 frames)",
                                (xmin, ymin - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Also display object ID for debugging
                cv2.putText(frame, f"ID: {object_id}",
                            (xmin, ymin - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Track best detection for picking
                if detection.confidence > best_confidence:
                    best_confidence = detection.confidence
                    best_detection = {
                        "coordinates": [x, y, avg_z if avg_z is not None else z],  # Use averaged Z if available
                        "class_name": class_name,
                        "confidence": detection.confidence,
                        "object_id": object_id
                    }

            # Display status and instructions
            mode_text = "AUTO MODE" if auto_mode else "MANUAL MODE"
            cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "p: pick  a: toggle auto  q: quit  r: reset offsets", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, "Adjust: x/z: X  c/v: Y  b/n: Z", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Display current offsets
            cv2.putText(frame, f"Offsets: X={X_OFFSET}mm Y={Y_OFFSET}mm Z={Z_OFFSET}mm",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Show current Z history status for each object
            y_pos = 130
            for class_name in ["Battery", "CBattery"]:
                object_count = len(class_settings[class_name]["objects"])
                cv2.putText(frame, f"{class_name} Objects: {object_count}", (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_settings[class_name]["color"], 1)
                y_pos += 20

                # Display the first 3 objects for each class
                count = 0
                for object_id, object_data in class_settings[class_name]["objects"].items():
                    if count >= 3:  # Limit to 3 objects to prevent cluttering the display
                        break
                    avg_z = get_avg_z(class_name, object_id)
                    history_count = len(object_data["z_history"])
                    status = f"  ID {object_id}: Z-Buffer {history_count}/5"
                    if avg_z is not None:
                        status += f" (Avg: {avg_z:.0f}mm)"
                    cv2.putText(frame, status, (20, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_settings[class_name]["color"], 1)
                    y_pos += 20
                    count += 1

            # Clean up old objects (not seen for more than 10 seconds)
            current_time = time.time()
            for class_name in ["Battery", "CBattery"]:
                # Create a copy of the keys to avoid modifying during iteration
                object_ids = list(class_settings[class_name]["objects"].keys())
                for object_id in object_ids:
                    if current_time - class_settings[class_name]["objects"][object_id]["last_seen"] > 10:
                        del class_settings[class_name]["objects"][object_id]

            # Show frame
            cv2.imshow("Battery Sorting System", frame)

            # Handle key presses
            key = cv2.waitKey(1)

            # Toggle auto mode
            if key == ord('a'):
                auto_mode = not auto_mode
                print(f"Auto mode {'enabled' if auto_mode else 'disabled'}")

            # Offset adjustment keys
            if key == ord('x'):  # Increase X offset
                adjust_offsets(x_offset=X_OFFSET + 5)
            elif key == ord('z'):  # Decrease X offset
                adjust_offsets(x_offset=X_OFFSET - 5)
            elif key == ord('c'):  # Increase Y offset
                adjust_offsets(y_offset=Y_OFFSET + 5)
            elif key == ord('v'):  # Decrease Y offset
                adjust_offsets(y_offset=Y_OFFSET - 5)
            elif key == ord('b'):  # Increase Z offset
                adjust_offsets(z_offset=Z_OFFSET + 5)
            elif key == ord('n'):  # Decrease Z offset
                adjust_offsets(z_offset=Z_OFFSET - 5)
            elif key == ord('r'):  # Reset all offsets
                adjust_offsets(0, 0, 0)

            # Check if we should process a battery
            current_time = time.time()
            should_process = (
                    (key == ord('p')) or
                    (auto_mode and best_detection and current_time - last_processed_time > cooldown_time)
            )

            # Process the best detection if needed
            if should_process and best_detection:
                battery_type = best_detection["class_name"]
                is_cbattery = (battery_type == "CBattery")

                print(f"\nProcessing {battery_type} (confidence: {best_detection['confidence']:.2f})")

                # Log the Z value being used (raw or averaged)
                camera_coords = best_detection["coordinates"]
                object_id = best_detection["object_id"]
                avg_z = get_avg_z(battery_type, object_id)
                if avg_z is not None and avg_z == camera_coords[2]:
                    print(
                        f"Using averaged Z value: {avg_z:.0f}mm from {len(class_settings[battery_type]['objects'][object_id]['z_history'])} samples")

                success = pick_and_place_battery(
                    mc,
                    camera_coords,
                    is_cbattery
                )

                if success:
                    print(f"Successfully sorted {battery_type}")
                    last_processed_time = current_time

                    # Clear object from tracking after successful pick
                    if "object_id" in best_detection and best_detection["object_id"] in class_settings[battery_type][
                        "objects"]:
                        del class_settings[battery_type]["objects"][best_detection["object_id"]]
                    print(f"Removed object from tracking")
                else:
                    print(f"Failed to sort {battery_type}")

            # Quit on 'q' key
            if key == ord('q'):
                break

        # Clean up
        cv2.destroyAllWindows()

        # Return to neutral position
        print("Returning to neutral position...")
        mc.send_angles([0, 0, 0, 0, 0, 0], 30)
        time.sleep(5)
        print("System shutdown complete.")


if __name__ == '__main__':
    main()