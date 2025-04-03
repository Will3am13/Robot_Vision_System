import numpy as np
import math
import logging
import os
import pickle
from sklearn.linear_model import Ridge
from typing import Tuple, Dict, List, Union, Any

# Import settings from config
from config import (
    HAND_COORDS, EYE_COORDS, DISTANCE_SETTINGS,
    PICK_ORIENTATION, X_OFFSET, Y_OFFSET, Z_OFFSET, MIN_Z
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CoordinateTransform")

# Global variables (these will be updated based on distance)
global_pick_orientation = PICK_ORIENTATION.copy()
global_x_offset = X_OFFSET
global_y_offset = Y_OFFSET
global_z_offset = Z_OFFSET
global_min_z = MIN_Z

# Directory for saved models
MODELS_DIR = "saved_models"
MODEL_X_PATH = os.path.join(MODELS_DIR, "ridge_model_x.pkl")
MODEL_Y_PATH = os.path.join(MODELS_DIR, "ridge_model_y.pkl")
MODEL_Z_PATH = os.path.join(MODELS_DIR, "ridge_model_z.pkl")

# Ensure models directory exists
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)
    logger.info(f"Created models directory: {MODELS_DIR}")


def save_model(model, path):
    """Save a trained model to disk"""
    try:
        with open(path, 'wb') as file:
            pickle.dump(model, file)
        logger.info(f"Saved model to {path}")
        return True
    except Exception as e:
        logger.error(f"Error saving model to {path}: {str(e)}")
        return False


def load_model(path):
    """Load a trained model from disk"""
    try:
        with open(path, 'rb') as file:
            model = pickle.load(file)
        logger.info(f"Loaded model from {path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {path}: {str(e)}")
        return None


def train_and_save_models(eye_coords, hand_coords):
    """
    Train Ridge regression models and save them to disk
    
    Args:
        eye_coords (np.array): Array of camera coordinates (N x 3)
        hand_coords (np.array): Array of corresponding robot coordinates (N x 3)
        
    Returns:
        tuple: (model_x, model_y, model_z) trained models
    """
    logger.info("Training coordinate transformation models...")
    
    # Convert inputs to numpy arrays if they aren't already
    eye_coords_np = np.array(eye_coords)
    hand_coords_np = np.array(hand_coords)
    
    # Train separate Ridge regression models for each axis (x, y, z)
    model_x = Ridge(alpha=1.0)
    model_y = Ridge(alpha=1.0)
    model_z = Ridge(alpha=1.0)

    model_x.fit(eye_coords_np, hand_coords_np[:, 0])
    model_y.fit(eye_coords_np, hand_coords_np[:, 1])
    model_z.fit(eye_coords_np, hand_coords_np[:, 2])
    
    # Save the trained models
    save_model(model_x, MODEL_X_PATH)
    save_model(model_y, MODEL_Y_PATH)
    save_model(model_z, MODEL_Z_PATH)
    
    logger.info("Coordinate transformation models trained and saved")
    
    return model_x, model_y, model_z


def load_or_train_models(eye_coords, hand_coords):
    """
    Load models from disk if they exist, otherwise train new ones
    
    Args:
        eye_coords (np.array): Array of camera coordinates (N x 3)
        hand_coords (np.array): Array of corresponding robot coordinates (N x 3)
        
    Returns:
        tuple: (model_x, model_y, model_z) models
    """
    # Try to load models
    model_x = load_model(MODEL_X_PATH)
    model_y = load_model(MODEL_Y_PATH)
    model_z = load_model(MODEL_Z_PATH)
    
    # If any model failed to load, train all models
    if model_x is None or model_y is None or model_z is None:
        logger.info("One or more models not found or failed to load, training new models")
        return train_and_save_models(eye_coords, hand_coords)
    
    return model_x, model_y, model_z


def create_transform_function(model_x, model_y, model_z):
    """
    Create a transformation function using the provided models
    
    Args:
        model_x, model_y, model_z: Ridge regression models for each dimension
        
    Returns:
        function: A function that takes camera coordinates and returns transformed robot coordinates
    """
    def transform(camera_points):
        """
        Transforms camera points to robot points using the trained models.

        Args:
            camera_points (np.array): Array of camera coordinates to transform (M x 3).

        Returns:
            np.array: Array of transformed robot coordinates (M x 3).
        """
        # Ensure camera_points is a 2D array
        camera_points_np = np.array(camera_points)
        if camera_points_np.ndim == 1:
            camera_points_np = camera_points_np.reshape(1, -1)

        transformed_x = model_x.predict(camera_points_np)
        transformed_y = model_y.predict(camera_points_np)
        transformed_z = model_z.predict(camera_points_np)

        return np.column_stack((transformed_x, transformed_y, transformed_z))

    return transform


# Load or train models and create the transformation function once at module initialization
model_x, model_y, model_z = load_or_train_models(EYE_COORDS, HAND_COORDS)
transform_func = create_transform_function(model_x, model_y, model_z)


def calculate_xy_distance(x, y):
    """
    Calculate the Euclidean distance in the XY plane from the origin (0,0)
    
    Args:
        x (float): X coordinate
        y (float): Y coordinate
        
    Returns:
        float: Euclidean distance in the XY plane
    """
    return math.sqrt(x ** 2 + y ** 2)


def update_settings_based_on_distance(distance):
    """
    Update the global orientation and offset settings based on the calculated distance

    Args:
        distance (float): Euclidean distance in XY plane from origin

    Returns:
        dict: Contains 'valid' (bool), 'message' (str), and 'range' (str) fields
    """
    global global_pick_orientation, global_x_offset, global_y_offset, global_z_offset, global_min_z

    # Initialize return values
    result = {
        'valid': False,
        'message': "",
        'range': ""
    }

    # Check if object is too close to robot base
    if distance < DISTANCE_SETTINGS["too_close"]["max_distance"]:
        result['message'] = DISTANCE_SETTINGS["too_close"]["message"]
        result['range'] = "too_close"
        return result

    # Check if object is too far from robot base
    if distance > DISTANCE_SETTINGS["too_far"]["min_distance"]:
        result['message'] = DISTANCE_SETTINGS["too_far"]["message"]
        result['range'] = "too_far"
        return result

    # Object is in valid range, determine which range
    if DISTANCE_SETTINGS["short_range"]["min_distance"] <= distance <= DISTANCE_SETTINGS["short_range"]["max_distance"]:
        # Short range settings
        global_pick_orientation = DISTANCE_SETTINGS["short_range"]["pick_orientation"].copy()
        global_x_offset, global_y_offset, global_z_offset = DISTANCE_SETTINGS["short_range"]["offsets"]
        global_min_z = DISTANCE_SETTINGS["short_range"]["min_z"]  # Update minimum Z value
        result['valid'] = True
        result['message'] = f"Using short range settings (distance: {distance:.1f}mm)"
        result['range'] = "short_range"

    elif DISTANCE_SETTINGS["normal_range"]["min_distance"] <= distance <= DISTANCE_SETTINGS["normal_range"]["max_distance"]:
        # Normal range settings
        global_pick_orientation = DISTANCE_SETTINGS["normal_range"]["pick_orientation"].copy()
        global_x_offset, global_y_offset, global_z_offset = DISTANCE_SETTINGS["normal_range"]["offsets"]
        global_min_z = DISTANCE_SETTINGS["normal_range"]["min_z"]  # Update minimum Z value
        result['valid'] = True
        result['message'] = f"Using normal range settings (distance: {distance:.1f}mm)"
        result['range'] = "normal_range"

    elif DISTANCE_SETTINGS["long_range"]["min_distance"] <= distance <= DISTANCE_SETTINGS["long_range"]["max_distance"]:
        # Long range settings
        global_pick_orientation = DISTANCE_SETTINGS["long_range"]["pick_orientation"].copy()
        global_x_offset, global_y_offset, global_z_offset = DISTANCE_SETTINGS["long_range"]["offsets"]
        global_min_z = DISTANCE_SETTINGS["long_range"]["min_z"]  # Update minimum Z value
        result['valid'] = True
        result['message'] = f"Using long range settings (distance: {distance:.1f}mm)"
        result['range'] = "long_range"

    return result


def transform_point(cam_point):
    """
    Transform point from camera coordinates to robot coordinates using pre-trained models
    
    Args:
        cam_point (list): Camera coordinates [x, y, z]
        
    Returns:
        tuple: (robot_coords, settings, distance)
            - robot_coords (np.array): Transformed robot coordinates with offsets applied
            - settings (dict): Distance-based settings used
            - distance (float): Original distance calculation before offsets
    """
    global global_pick_orientation, global_x_offset, global_y_offset, global_z_offset, global_min_z
    
    logger.debug(f"Transforming camera point: {cam_point}")
    
    # Ensure cam_point is a numpy array
    cam_point_np = np.array(cam_point)
    
    # Apply transformation using the pre-trained transform_func
    result = transform_func(cam_point_np.reshape(1, -1))

    # Store the original coordinates before applying any offsets
    original_coords = result[0].copy()
   
    # Calculate distance from origin in XY plane BEFORE applying any offsets
    distance = calculate_xy_distance(original_coords[0], original_coords[1])

    # Update settings based on distance
    settings = update_settings_based_on_distance(distance)

    # If valid range, apply offsets AFTER distance calculation and settings update
    if settings['valid']:
        result[0][0] += global_x_offset
        result[0][1] += global_y_offset
        result[0][2] += global_z_offset

    # Enforce minimum Z value based on the current range setting
    if result[0][2] < global_min_z:
        result[0][2] = global_min_z

    logger.debug(f"Transformed to robot coordinates: {result[0]} with {settings}")
    
    # Return the result as a 1D array, settings, and the original distance calculation
    return result[0], settings, distance


def is_valid_coord(coord):
    """
    Check if coordinate values are within the safe working range of the robot
    
    Args:
        coord (list): 6D coordinates [x, y, z, rx, ry, rz]
        
    Returns:
        bool: True if coordinates are valid, False otherwise
    """
    x, y, z, rx, ry, rz = coord
    
    if not (-281.45 <= x <= 281.45):  # x range
        logger.warning(f"Out of range in x: {x}")
        return False
    
    if not (-281.45 <= y <= 281.45):  # y range
        logger.warning(f"Out of range in y: {y}")
        return False
    
    if not (-70 <= z <= 412.67):  # z range
        logger.warning(f"Out of range in z: {z}")
        return False
    
    if not (-180 <= rx <= 180):  # rx range
        logger.warning(f"Out of range in roll: {rx}")
        return False
    
    if not (-180 <= ry <= 180):  # ry range
        logger.warning(f"Out of range in pitch: {ry}")
        return False
    
    if not (-180 <= rz <= 180):  # rz range
        logger.warning(f"Out of range in yaw: {rz}")
        return False
    
    return True


def get_current_pick_orientation():
    """
    Get the current pick orientation
    
    Returns:
        list: Current pick orientation [rx, ry, rz]
    """
    return global_pick_orientation.copy()


# For testing coordinate transform and model saving/loading
if __name__ == "__main__":
    # Force retrain models
    print("Testing model training and saving...")
    model_x, model_y, model_z = train_and_save_models(EYE_COORDS, HAND_COORDS)
    
    # Test loading models
    print("Testing model loading...")
    loaded_model_x, loaded_model_y, loaded_model_z = load_or_train_models(EYE_COORDS, HAND_COORDS)
    
    # Create transform function with loaded models
    test_transform_func = create_transform_function(loaded_model_x, loaded_model_y, loaded_model_z)
    
    # Test coordinate transformation
    test_cam_point = [50, -80, 400]
    robot_xyz = test_transform_func(test_cam_point)
    
    print(f"Camera coordinates: {test_cam_point}")
    print(f"Robot coordinates: {robot_xyz[0]}")
    
    # Test the full transform_point function
    robot_xyz, settings, distance = transform_point(test_cam_point)
    
    print(f"Transformed coordinates with settings: {robot_xyz}")
    print(f"Distance: {distance}")
    print(f"Settings: {settings}")
    print(f"Current pick orientation: {get_current_pick_orientation()}")
