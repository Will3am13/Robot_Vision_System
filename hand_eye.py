import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

def transform_camera_to_robot_poly(eye_coords, hand_coords, degree=1):
    """Transforms using polynomial regression."""
    poly = PolynomialFeatures(degree=degree)
    eye_poly = poly.fit_transform(eye_coords)

    model_x = LinearRegression()
    model_y = LinearRegression()
    model_z = LinearRegression()

    model_x.fit(eye_poly, hand_coords[:, 0])
    model_y.fit(eye_poly, hand_coords[:, 1])
    model_z.fit(eye_poly, hand_coords[:, 2])

    def transform(camera_points):
      camera_points_poly = poly.transform(camera_points)
      transformed_x = model_x.predict(camera_points_poly)
      transformed_y = model_y.predict(camera_points_poly)
      transformed_z = model_z.predict(camera_points_poly)
      return np.column_stack((transformed_x, transformed_y, transformed_z))

    return transform, model_x, model_y, model_z, poly

# Example usage with your provided data:
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
                       [30, -116, 343]])

# Create the transformation function and models
transform_func, model_x, model_y, model_z, poly = transform_camera_to_robot_poly(eye_coords, hand_coords, degree=2)

# Transform the original camera coordinates
transformed_all = transform_func(eye_coords)

# Calculate the error for each point and axis
errors = hand_coords - transformed_all

# Print the errors
print("Error for each point (x, y, z):")
for i, error in enumerate(errors):
    print(f"Point {i+1}: {error}")

# Calculate the mean squared error for each axis
mse_x = mean_squared_error(hand_coords[:, 0], transformed_all[:, 0])
mse_y = mean_squared_error(hand_coords[:, 1], transformed_all[:, 1])
mse_z = mean_squared_error(hand_coords[:, 2], transformed_all[:, 2])

print("\nMean Squared Error (MSE) for each axis:")
print(f"MSE X: {mse_x}")
print(f"MSE Y: {mse_y}")
print(f"MSE Z: {mse_z}")

# Add a way to input a new eye coordinate and see the robot frame output:
new_eye_coord_input = input("\nEnter a new eye coordinate (x, y, z), separated by commas: ")

try:
    new_eye_coord = np.array([list(map(float, new_eye_coord_input.split(",")))]) #convert to numpy array.
    transformed_new_coord = transform_func(new_eye_coord)
    print("\nTransformed robot coordinate for the new eye coordinate:")
    print(transformed_new_coord)
except ValueError:
    print("Invalid input. Please enter coordinates in the format 'x,y,z'.")