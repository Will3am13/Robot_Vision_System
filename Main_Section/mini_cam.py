import os
import cv2
from datetime import datetime
import gc
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MiniCam")

class MiniCam:
    def __init__(self, camera_index=0, screenshot_folder='hover_screenshots'):
        """
        Initialize the camera with screenshot capability
       
        Args:
            camera_index (int): Index of the camera to use
            screenshot_folder (str): Folder to save hover screenshots
        """
        self.camera_index = camera_index
        self.cap = None
        self.screenshot_folder = screenshot_folder
        self.is_initialized = False
       
        # Create the screenshot folder if it doesn't exist
        if not os.path.exists(self.screenshot_folder):
            os.makedirs(self.screenshot_folder)
            logger.info(f"Created screenshot folder: {self.screenshot_folder}")

    def initialize(self):
        """Initialize the camera capture with error handling"""
        logger.info(f"Initializing camera at index {self.camera_index}")
        try:
            # Make sure we close any previous capture if it exists
            if self.cap is not None:
                self.release()
                
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise Exception(f"Failed to open camera at index {self.camera_index}")
                
            # Set properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.is_initialized = True
            logger.info(f"Camera at index {self.camera_index} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Camera initialization error: {str(e)}")
            self.is_initialized = False
            return False

    def get_frame(self, apply_filters=True):
        """
        Capture a frame from the camera with error handling
       
        Args:
            apply_filters (bool): Whether to apply filters to the frame
           
        Returns:
            tuple: (success, frame) where success is a boolean and frame is the captured image
        """
        if not self.is_initialized:
            logger.warning("Camera not initialized. Attempting to initialize...")
            if not self.initialize():
                logger.error("Failed to initialize camera")
                return False, None
               
        try:
            success, frame = self.cap.read()
            
            if not success:
                logger.warning("Failed to read frame. Attempting to reinitialize camera...")
                self.release()
                if self.initialize():
                    success, frame = self.cap.read()
                else:
                    return False, None
           
            if success and apply_filters:
                # Apply any filters here if needed
                pass
                
            return success, frame
            
        except Exception as e:
            logger.error(f"Error capturing frame: {str(e)}")
            return False, None
   
    def take_screenshot(self):
        """
        Capture an unfiltered screenshot from the camera and save it to the screenshot folder
       
        Returns:
            str: Path to the saved screenshot file, or None if failed
        """
        try:
            logger.info("Taking screenshot")
            success, frame = self.get_frame(apply_filters=False)
           
            if success:
                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{self.screenshot_folder}/hover_screenshot_{timestamp}.jpg"
               
                # Save the screenshot
                cv2.imwrite(filename, frame)
                logger.info(f"Screenshot saved to {filename}")
                return filename
            else:
                logger.warning("Failed to capture screenshot - no frame available")
                return None
                
        except Exception as e:
            logger.error(f"Screenshot error: {str(e)}")
            return None

    def release(self):
        """Release the camera resources with error handling"""
        try:
            if self.cap is not None:
                logger.info("Releasing camera resources")
                self.cap.release()
                self.cap = None
                self.is_initialized = False
                
                # Force garbage collection to ensure resources are freed
                gc.collect()
                logger.info("Camera resources released successfully")
                
        except Exception as e:
            logger.error(f"Error releasing camera: {str(e)}")


# Standalone function for taking a single screenshot with automatic cleanup
def take_single_screenshot(camera_index=0, screenshot_folder='hover_screenshots'):
    """
    Take a single screenshot and ensure all resources are properly released
    
    Args:
        camera_index (int): Index of the camera to use
        screenshot_folder (str): Folder to save the screenshot
        
    Returns:
        str: Path to the saved screenshot, or None if failed
    """
    camera = MiniCam(camera_index, screenshot_folder)
    
    try:
        if camera.initialize():
            # Take screenshot
            screenshot_path = camera.take_screenshot()
            return screenshot_path
        else:
            logger.error("Failed to initialize camera for screenshot")
            return None
    except Exception as e:
        logger.error(f"Error taking screenshot: {str(e)}")
        return None
    finally:
        # Always release camera resources, even if an exception occurs
        camera.release()
        # Force garbage collection
        gc.collect()


# For testing the camera standalone
if __name__ == "__main__":
    # Test the standalone screenshot function
    screenshot_path = take_single_screenshot()
    if screenshot_path:
        print(f"Screenshot saved to: {screenshot_path}")
    else:
        print("Failed to take screenshot")
