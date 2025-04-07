import time
import os
import sys
import numpy as np
import depthai as dai
import cv2
from mini_cam_controller import MiniCamController


class CameraManager:
    def __init__(self,
                 oak_d_model_path="/home/er/Downloads/BatteryV2.blob",
                 mini_cam_model_path=None,
                 mini_cam_id=1,
                 oak_camera_active=False):
        """
        Initialize the camera manager to handle both Oak-D-SR and mini camera resources.

        Args:
            oak_d_model_path (str): Path to the Oak-D-SR model blob
            mini_cam_model_path (str): Path to the mini camera model
            mini_cam_id (int): Camera ID for the mini camera
            oak_camera_active (bool): Whether the Oak-D-SR camera should be active immediately
        """
        self.oak_d_model_path = oak_d_model_path
        self.mini_cam_model_path = mini_cam_model_path
        self.mini_cam_id = mini_cam_id
        self.oak_camera_active = oak_camera_active

        # Initialize variables for Oak-D-SR camera
        self.device = None
        self.pipeline = None
        self.qVideo = None
        self.qDet = None
        self.qControl = None

        # Initialize the mini camera controller but don't start camera
        self.mini_cam = None
        if mini_cam_model_path:
            self.mini_cam = MiniCamController(
                model_path=mini_cam_model_path,
                camera_id=mini_cam_id,
                x_pixel_to_mm=0.1,  # Calibrate this value
                y_pixel_to_mm=0.1,  # Calibrate this value
                angle_to_robot=1.0  # Calibrate this value
            )

    def setup_vision_pipeline(self):
        """Setup and configure the DepthAI vision pipeline with Script node for control"""
        # Create pipeline
        self.pipeline = dai.Pipeline()

        # Define sources and outputs
        camRgb = self.pipeline.create(dai.node.ColorCamera)
        spatialDetectionNetwork = self.pipeline.create(dai.node.YoloSpatialDetectionNetwork)
        stereo = self.pipeline.create(dai.node.StereoDepth)
        monoRight = self.pipeline.create(dai.node.MonoCamera)

        # Create a script node to control camera
        script = self.pipeline.create(dai.node.Script)

        # Create output nodes
        xoutVideo = self.pipeline.create(dai.node.XLinkOut)
        xoutNN = self.pipeline.create(dai.node.XLinkOut)
        xinControl = self.pipeline.create(dai.node.XLinkIn)

        # Set stream names
        xoutVideo.setStreamName("video")
        xoutNN.setStreamName("detections")
        xinControl.setStreamName("control")

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
        spatialDetectionNetwork.setBlobPath(self.oak_d_model_path)
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

        # Script node to control frame processing
        script.setScript("""
        # Initial camera state
        active = """ + str(self.oak_camera_active).lower() + """

        # Main loop
        while True:
            # Check for control messages
            ctrl = node.io['control'].tryGet()
            if ctrl is not None:
                cmd = ctrl.getText()
                if cmd == "START":
                    active = True
                    print("Camera processing activated")
                elif cmd == "STOP":
                    active = False
                    print("Camera processing paused")
                elif cmd == "TOGGLE":
                    active = not active
                    print(f"Camera processing {'activated' if active else 'paused'}")

            # If active, process frames normally
            preview = node.io['preview'].tryGet()
            if preview is not None and active:
                # Pass the frame to YOLO for processing
                node.io['nn_in'].send(preview)
                # Also pass to video output
                node.io['video'].send(preview)
        """)

        # Connect Script node
        camRgb.preview.link(script.inputs['preview'])
        script.outputs['nn_in'].link(spatialDetectionNetwork.input)
        script.outputs['video'].link(xoutVideo.input)
        xinControl.out.link(script.inputs['control'])

        # Connect other nodes
        camRgb.video.link(stereo.left)
        monoRight.out.link(stereo.right)
        stereo.depth.link(spatialDetectionNetwork.inputDepth)
        spatialDetectionNetwork.out.link(xoutNN.input)

        return self.pipeline

    def start_device(self):
        """Start the Oak-D-SR device (but not necessarily activate frame processing)"""
        if self.device is not None:
            print("Oak-D-SR device already running")
            return True

        try:
            print("Starting Oak-D-SR device...")
            pipeline = self.setup_vision_pipeline()
            self.device = dai.Device(pipeline)

            # Output queues
            self.qVideo = self.device.getOutputQueue(name="video", maxSize=4, blocking=False)
            self.qDet = self.device.getOutputQueue(name="detections", maxSize=4, blocking=False)
            self.qControl = self.device.getInputQueue(name="control")

            # Set initial camera state
            self.set_camera_active(self.oak_camera_active)

            print("Oak-D-SR device started successfully")
            return True
        except Exception as e:
            print(f"Error starting Oak-D-SR device: {e}")
            return False

    def stop_device(self):
        """Stop the Oak-D-SR device and release resources"""
        if self.device is None:
            print("Oak-D-SR device not running")
            return True

        try:
            print("Stopping Oak-D-SR device...")
            # Close the device to release resources
            self.device.close()
            self.device = None
            self.qVideo = None
            self.qDet = None
            self.qControl = None
            self.oak_camera_active = False
            print("Oak-D-SR device stopped successfully")
            return True
        except Exception as e:
            print(f"Error stopping Oak-D-SR device: {e}")
            return False

    def set_camera_active(self, active=True):
        """Activate or deactivate frame processing without stopping the device"""
        if self.device is None or self.qControl is None:
            print("Oak-D-SR device not running, cannot change camera state")
            return False

        try:
            # Create a control message
            ctrl = dai.Buffer()
            ctrl.setData(b"START" if active else b"STOP")

            # Send control message to script node
            self.qControl.send(ctrl)

            # Update state
            self.oak_camera_active = active

            print(f"Oak-D-SR camera {'activated' if active else 'paused'}")
            return True
        except Exception as e:
            print(f"Error changing camera state: {e}")
            return False

    def toggle_camera_active(self):
        """Toggle camera processing on/off without stopping the device"""
        if self.device is None or self.qControl is None:
            print("Oak-D-SR device not running, cannot toggle camera state")
            return False

        try:
            # Create a control message
            ctrl = dai.Buffer()
            ctrl.setData(b"TOGGLE")

            # Send control message to script node
            self.qControl.send(ctrl)

            # Update state (as we don't know the current state in the script)
            self.oak_camera_active = not self.oak_camera_active

            print(f"Oak-D-SR camera toggled to {'active' if self.oak_camera_active else 'paused'}")
            return True
        except Exception as e:
            print(f"Error toggling camera state: {e}")
            return False

    def get_frame(self):
        """Get the latest frame and detections from the Oak-D-SR camera"""
        if self.device is None or not self.oak_camera_active:
            return None, None

        # Try to get the latest video frame
        video = self.qVideo.tryGet()
        if video is None:
            return None, None

        # Try to get the latest detections
        det = self.qDet.tryGet()
        detections = det.detections if det is not None else []

        return video.getCvFrame(), detections

    def use_mini_cam_for_fine_tuning(self, mc, current_xyz, threshold_distance=10, threshold_angle=5, max_attempts=3,
                                     visualization=False):
        """
        Fine-tune robot position using mini camera

        Args:
            mc: MyCobot instance
            current_xyz: Current robot XYZ position
            threshold_distance: Pixel threshold for centering
            threshold_angle: Angle threshold
            max_attempts: Maximum adjustment attempts
            visualization: Whether to show camera view

        Returns:
            tuple: (success, adjusted_xyz, adjusted_angle)
        """
        if self.mini_cam is None:
            print("Mini camera controller not initialized")
            return False, current_xyz, None

        # Temporary pause Oak-D-SR camera if it's active
        was_active = self.oak_camera_active
        if was_active:
            print("Temporarily pausing Oak-D-SR camera processing...")
            self.set_camera_active(False)
            time.sleep(0.5)  # Give time for processing to stop

        # Use mini camera for fine-tuning
        success, adjusted_xyz, adjusted_angle = self.mini_cam.fine_tune_position(
            mc=mc,
            current_xyz=current_xyz,
            threshold_distance=threshold_distance,
            threshold_angle=threshold_angle,
            max_attempts=max_attempts,
            visualization=visualization
        )

        # Resume Oak-D-SR camera if it was active before
        if was_active:
            print("Resuming Oak-D-SR camera processing...")
            self.set_camera_active(True)

        return success, adjusted_xyz, adjusted_angle

    def cleanup(self):
        """Clean up resources for both cameras"""
        if self.device is not None:
            self.stop_device()

        # Mini camera should already be stopped after use, but make sure
        if self.mini_cam is not None:
            self.mini_cam.stop_camera()

        print("All camera resources cleaned up")