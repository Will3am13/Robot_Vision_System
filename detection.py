import time
import os
import sys
import numpy as np
import depthai as dai
import cv2
from pymycobot.genre import Angle, Coord
from pymycobot import MyCobot280
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Define a standby (home) position
STANDBY_COORDS = [100.6, -29.8, 318.0, 147.27, -26.28, 55.77]

# Bin locations (fixed positions for sorting)
BATTERY_BIN_COORD = [194.0, -142.8, 144.9, 152.06, -20.98, 1.67]  # Regular battery bin
CBATTERY_BIN_COORD = [99.8, -186.7, 162.6, 172.39, -2.74, -14.16]  # CBattery bin

# Fixed orientation angles for picking and placing
PICK_ORIENTATION = [175.82, -0.64, -3.97]  # Fixed pitch, roll, yaw for picking
PLACE_ORIENTATION = [152.06, -20.98, 1.67]  # Fixed pitch, roll, yaw for placing


# Polynomial transformation setup
def transform_camera_to_robot_poly(eye_coords, hand_coords, degree=2):
    """
    Creates a polynomial regression model for camera to robot coordinate transformation

    Args:
        eye_coords: Camera coordinates as a numpy array of shape (n, 3)
        hand_coords: Robot coordinates as a numpy array of shape (n, 3)
        degree: Degree of polynomial features (default=2)

    Returns:
        transform_func: Function to transform new camera coordinates to robot coordinates
        models: The trained models for each axis
    """
    poly = PolynomialFeatures(degree=degree)
    eye_poly = poly.fit_transform(eye_coords)

    model_x = LinearRegression()
    model_y = LinearRegression()
    model_z = LinearRegression()

    model_x.fit(eye_poly, hand_coords[:, 0])
    model_y.fit(eye_poly, hand_coords[:, 1])
    model_z.fit(eye_poly, hand_coords[:, 2])

    def transform(camera_points):
        """Transform camera coordinates to robot coordinates using the trained models"""
        # Ensure camera_points is a 2D array
        if camera_points.ndim == 1:
            camera_points = camera_points.reshape(1, -1)

        camera_points_poly = poly.transform(camera_points)
        transformed_x = model_x.predict(camera_points_poly)
        transformed_y = model_y.predict(camera_points_poly)
        transformed_z = model_z.predict(camera_points_poly)

        return np.column_stack((transformed_x, transformed_y, transformed_z))

    return transform, (model_x, model_y, model_z, poly)


# Example calibration data (replace with your actual data if needed)
hand_coords = np.array([
    [275.0, -12.1, 103.3],
    [269.3, -16.5, 135.2],
    [92.8, 9.2, 113.3],
    [197.4, -97.2, 104.2],
    [207.2, -91.7, 172.0],
    [229.0, 165.9, 100.1],
    [228.6, 164.4, 133.0],
    [149.0, 152.3, 233.7],
    [277.1, 18.8, 85.9],
    [95.5, 158.0, 91.8]
])

eye_coords = np.array([
    [66, -56, 575],
    [75, -16, 575],
    [170, -97, 435],
    [191, -66, 575],
    [191, 10, 597],
    [-51, -79, 460],
    [-53, -42, 460],
    [14, 40, 375],
    [54, -66, 575],
    [30, -116, 343]
])

# Create the transformation function
transform_func, models = transform_camera_to_robot_poly(eye_coords, hand_coords, degree=2)


# Function to transform coordinates from camera to robot space
def transform_point(cam_point):
    """Transform point from camera coordinates to robot coordinates using polynomial regression"""
    # Ensure cam_point is a numpy array
    cam_point_np = np.array(cam_point)

    # Apply transformation using the polynomial regression model
    result = transform_func(cam_point_np.reshape(1, -1))

    # Return the result as a 1D array
    return result[0]


# Function to check if coordinate values are within the safe working range
def is_valid_coord(coord):
    x, y, z, rx, ry, rz = coord
    if not (-281.45 <= x <= 281.45):  # x range
        return False
    if not (-281.45 <= y <= 281.45):  # y range
        return False
    if not (-70 <= z <= 412.67):  # z range
        return False
    if not (-180 <= rx <= 180):  # rx range
        return False
    if not (-180 <= ry <= 180):  # ry range
        return False
    if not (-180 <= rz <= 180):  # rz range
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

    # YOLO specific settings
    spatialDetectionNetwork.setBlobPath(r"/home/er/Downloads/battery.blob")
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
    mc.send_coords(STANDBY_COORDS, 30, 1)
    time.sleep(2)

    return mc


def pick_and_place_battery(mc, camera_coords, is_cbattery=False):
    """Pick up a battery at the given coordinates and place it in the appropriate bin"""
    # Transform camera coordinates to robot coordinates
    robot_xyz = transform_point(camera_coords)
    print(f"Camera coordinates: {camera_coords}")
    print(f"Transformed robot coordinates: {robot_xyz}")

    # Create full 6D coordinates with fixed orientation for picking
    pick_coords = list(robot_xyz) + PICK_ORIENTATION

    # Select the appropriate bin based on battery type
    place_coords = CBATTERY_BIN_COORD if is_cbattery else BATTERY_BIN_COORD

    # Validate coordinates are within robot's working range
    if not is_valid_coord(pick_coords):
        print(f"ERROR: Pick coordinates out of range: {pick_coords}")
        return False

    # Move to target position
    print(f"Moving to pick battery at: {pick_coords}")
    mc.send_coords(pick_coords, 30, 1)
    time.sleep(3)

    # Close gripper to grab the battery
    print("Grabbing battery (closing gripper)...")
    mc.set_gripper_state(1, 70)
    time.sleep(2)

    # Return to standby position with battery
    print("Returning to standby position...")
    mc.send_coords(STANDBY_COORDS, 30, 1)
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
    mc.send_coords(STANDBY_COORDS, 30, 1)
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
            "threshold": 0.5  # Detection threshold for Battery
        },
        "CBattery": {
            "color": (255, 0, 0),  # Blue for CBattery
            "threshold": 0.1  # Detection threshold for CBattery
        }
    }

    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:
        # Output queues
        qVideo = device.getOutputQueue(name="video", maxSize=4, blocking=False)
        qDet = device.getOutputQueue(name="detections", maxSize=4, blocking=False)

        print("Vision system running. Press 'p' to pick up detected battery, 'a' to enable auto-mode, 'q' to quit.")

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

                # Draw bounding box and information
                color = class_settings[class_name]["color"]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(frame, f"{class_name} {detection.confidence:.2f}",
                            (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Display spatial coordinates
                cv2.putText(frame, f"X: {x:.0f}mm  Y: {y:.0f}mm  Z: {z:.0f}mm",
                            (xmin, ymin - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Track best detection for picking
                if detection.confidence > best_confidence:
                    best_confidence = detection.confidence
                    best_detection = {
                        "coordinates": [x, y, z],
                        "class_name": class_name,
                        "confidence": detection.confidence
                    }

            # Display status and instructions
            mode_text = "AUTO MODE" if auto_mode else "MANUAL MODE"
            cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "p: pick  a: toggle auto  q: quit", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Show frame
            cv2.imshow("Battery Sorting System", frame)

            # Handle key presses
            key = cv2.waitKey(1)

            # Toggle auto mode
            if key == ord('a'):
                auto_mode = not auto_mode
                print(f"Auto mode {'enabled' if auto_mode else 'disabled'}")

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
                success = pick_and_place_battery(
                    mc,
                    best_detection["coordinates"],
                    is_cbattery
                )

                if success:
                    print(f"Successfully sorted {battery_type}")
                    last_processed_time = current_time
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