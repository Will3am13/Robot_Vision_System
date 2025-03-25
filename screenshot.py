import depthai as dai
import cv2
import numpy as np
import time

# Class-specific settings
class_settings = {
    "Battery": {
        "color": (0, 255, 0),  # Green for Battery
        "threshold": 0.5       # Detection threshold for Battery
    },
    "CBattery": {
        "color": (255, 0, 0),  # Blue for CBattery
        "threshold": 0.1       # Detection threshold for CBattery
    }
}

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

# Stereo depth settings - Set to HIGH_ACCURACY as requested
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
stereo.setLeftRightCheck(True)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_B)

# YOLO specific settings
spatialDetectionNetwork.setBlobPath(r"/home/er/Downloads/battery.blob")
spatialDetectionNetwork.setConfidenceThreshold(0.2)  # Set low threshold, we'll filter by class later
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

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    # Output queues
    qVideo = device.getOutputQueue(name="video", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
   
    # FPS calculation variables
    start_time = time.time()
    frame_count = 0
    fps = 0
   
    while True:
        # Get frames and detections
        inVideo = qVideo.get()
        inDet = qDet.get()

        frame = inVideo.getCvFrame()
        detections = inDet.detections
       
        # FPS calculation
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - start_time
       
        # Update FPS every second
        if elapsed_time > 1:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = current_time
       
        # Process each detection
        for detection in detections:
            # Determine class name
            class_name = "Battery" if detection.label == 0 else "CBattery"
           
            # Skip if below class-specific threshold
            if class_name not in class_settings or detection.confidence < class_settings[class_name]["threshold"]:
                continue
           
            # Get normalized coordinates
            xmin = detection.xmin
            ymin = detection.ymin
            xmax = detection.xmax
            ymax = detection.ymax
           
            # Convert normalized coordinates to pixel coordinates
            height, width = frame.shape[:2]
            x1 = int(xmin * width)
            y1 = int(ymin * height)
            x2 = int(xmax * width)
            y2 = int(ymax * height)
           
            # Get class-specific color
            color = class_settings[class_name]["color"]
           
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
           
            # Display class name and confidence at top of bounding box
            cv2.putText(frame, f"{class_name} {detection.confidence:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
           
            # Get spatial coordinates directly from detection
            x = detection.spatialCoordinates.x
            y = detection.spatialCoordinates.y
            z = detection.spatialCoordinates.z
           
            # Display spatial coordinates
            text_y_pos = y1 - 100
            cv2.putText(frame, f"X: {x:.0f}mm", (x1, text_y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, f"Y: {y:.0f}mm", (x1, text_y_pos + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, f"Z: {z:.0f}mm", (x1, text_y_pos + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
           
            # Draw a small circle at the center point where depth is measured
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)
       
        # Add FPS counter in the top-left corner
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
       
        # Add simple instructions
        cv2.putText(frame, "Press 'q' to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Show frame
        cv2.imshow("Battery Detection", frame)

        # Check for key presses
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

cv2.destroyAllWindows()
