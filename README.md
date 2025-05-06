# Integrated Robotic Vision System

## Project Overview
This repository contains the code and documentation for an integrated robotic vision system that combines stereoscopic camera technology with a dual-camera approach for real-time object detection, classification, and manipulation. The system is designed specifically for detecting and sorting 9-V batteries, distinguishing between corroded and non-corroded batteries.

## System Architecture
The system architecture is built on a Raspberry Pi platform that serves as the central controller, coordinating between:
- **OAK-D-SR camera**: Primary vision sensor with integrated Robotics Vision Core 2 (RVC2) for on-device processing
- **Secondary USB camera**: Provides fine-tuning alignment and orientation detection
- **MyCobot 280 robotic arm**: 6-axis robot with adaptive gripper for object manipulation

## Key Features
- **Dual-camera vision system**: Primary camera for coarse object detection and depth estimation, secondary camera for fine-grained alignment
- **Real-time object detection**: 90% accuracy for standard objects, 60% accuracy for detecting specific defects
- **Distributed processing**: Vision processing on the OAK-D-SR's RVC2, robot control and fine-tuning on Raspberry Pi
- **Adaptive distance-based configuration**: Different gripper orientations based on object distance
- **Z-depth averaging**: Multi-frame averaging for stable position estimation
- **Orientation-adaptive gripping**: Detects and adjusts to arbitrary object orientations
- **Safety validation**: Comprehensive coordinate checking and range verification

## Hardware Requirements
- Raspberry Pi 4 (4GB)
- Luxonis OAK-D-SR Camera
- USB Camera for fine alignment 
- Elephant Robotics MyCobot 280
- Adaptive Gripper for MyCobot 280
- Power supply
- 9-V batteries (for testing)

## Software Dependencies
- Python 3.7+
- OpenCV
- NumPy
- DepthAI API
- MyCobot Python SDK
- scikit-learn (for coordinate transformation)

## Key Algorithms

### Multi-Camera Object Detection and Position Calculation
1. Initialize OAK-D-SR camera and RVC2 processing
2. Capture stereoscopic image data for coarse detection
3. Process raw data through detection algorithm on RVC2
4. Detect object boundaries and classification
5. Transform coordinates from the camera's frame to the robot's frame
6. Send coordinates to the robot
7. Activate secondary USB camera for fine-tuning
8. Calculate object centroid and orientation
9. Adjust final grip coordinates and angle
10. Output refined position and orientation data

### Object Grasping Algorithm
1. Initialize in a neutral position, away from objects with gripper open
2. Receive the coordinates for object and class of object
3. Calculate path to object
4. Execute navigation
5. Receive orientation and final grip coordinates
6. Navigate to orientation and grip coordinates
7. Close the gripper

### Sorting Control System
1. Based on object class, select the appropriate destination bin (Corroded vs. Non-Corroded)
2. Calculate the path to the destination
3. Execute sorting movement
4. Open the gripper to release the object at the destination
5. Return to neutral position

## Authors
- William Dayton
- Antonio Coelho
