import cv2
import numpy as np

hand_coords = np.array([[275.0, -12.1, 103.3],
[269.3, -16.5, 135.2],
[92.8, 9.2, 113.3],
[197.4, -97.2, 104.2],
[207.2, -91.7, 172.0],
[229.0, 165.9, 100.1],
[228.6, 164.4, 133.0],
[149.0, 152.3, 233.7],
[277.1, 18.8, 85.9],
[95.5, 158.0, 91.8]])

eye_coords = np.array([[66, -56, 575],
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

# rotation matrix between the target and camera
R_target2cam = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [
						0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])

# translation vector between the target and camera
t_target2cam = np.array([0.0, 0.0, 0.0, 0.0])

# transformation matrix
T, _ = cv2.calibrateHandEye(hand_coords, eye_coords,
							R_target2cam, t_target2cam)

print(T)
