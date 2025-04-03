# config.py - Configuration settings and constants

# Robot settings
MYCOBOT_PORT = "/dev/ttyAMA0"
BAUDRATE = 1000000

# Define a standby (home) position
STANDBY_COORDS = [111.7, -53.4, 251.2, 172.67, -9.25, 46.06]
STANDBY_ANGLES = [0.52, 23.9, -75.93, -18.45, 0.08, -131.48]

# Bin locations (fixed positions for sorting)
BATTERY_BIN_COORD = [194.0, -142.8, 144.9, 152.06, -20.98, 1.67]  # Regular battery bin
CBATTERY_BIN_COORD = [99.8, -186.7, 162.6, 172.39, -2.74, -14.16]  # CBattery bin

# Distance-based configuration settings
DISTANCE_SETTINGS = {
    "too_close": {
        "max_distance": 65,
        "message": "Object too close to robot base"
    },
    "short_range": {
        "min_distance": 65,
        "max_distance": 130,
        "pick_orientation": [-142, 30, 59],
        "offsets": [75, 10, -20],
        "min_z": 60  # Minimum Z value for short range
    },
    "normal_range": {
        "min_distance": 130,
        "max_distance": 280,
        "pick_orientation": [180, 0, 45],
        "offsets": [0, 5, 0],
        "min_z": 105  # Minimum Z value for normal range
    },
    "long_range": {
        "min_distance": 280,
        "max_distance": 320,
        "pick_orientation": [150, -24, 50],
        "offsets": [-50, 5, 0],
        "min_z": 70  # Minimum Z value for long range
    },
    "too_far": {
        "min_distance": 320,
        "message": "Object out of range"
    }
}

# Default orientation and offsets (will be updated based on distance)
PICK_ORIENTATION = [180, 0, 45]  # Default - will be updated based on distance
PLACE_ORIENTATION = [152.06, -20.98, 1.67]  # Fixed pitch, roll, yaw for placing

# Default coordinate offsets (will be updated based on distance)
X_OFFSET = 0  # mm offset in X direction
Y_OFFSET = 5  # mm offset in Y direction
Z_OFFSET = 0  # mm offset in Z direction
MIN_Z = 105  # Default minimum Z value - will be updated based on distance

# Screenshot directory
SCREENSHOT_FOLDER = 'hover_screenshots'

# Object tracking settings
COOLDOWN_TIME = 25  # Seconds between auto-processing
Z_HISTORY_MAX_SIZE = 50  # Maximum size of Z value history
OBJECT_CLEANUP_TIME = 10  # Time in seconds after which objects are removed from tracking

# Class-specific settings
CLASS_SETTINGS = {
    "Battery": {
        "color": (0, 255, 0),  # Green for Battery
        "threshold": 0.5,  # Detection threshold for Battery
    },
    "CBattery": {
        "color": (255, 0, 0),  # Blue for CBattery
        "threshold": 0.1,  # Detection threshold for CBattery
    }
}

# Example calibration data
HAND_COORDS = [
    [98.3, 176.4, 101.4],
    [246.0, 150.3, 101.2],
    [246.0, 150.3, 101.2],
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
]

EYE_COORDS = [
    [1, -112, 343],
    [-55, -81, 480],
    [71, -74, 510],
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
]

# Oak-D camera settings
OAK_BLOB_PATH = "/home/er/Downloads/BatteryV2.blob"

# IPC settings
DETECTION_BUFFER_SIZE = 10  # Maximum number of detections to buffer
