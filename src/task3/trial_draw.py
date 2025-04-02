import os
import cv2
import numpy as np

# Initialize global variables to store origin, axis data, and rotation angles
origin = (250, 250)
x_axis = None
y_axis = None
x_angle = 0  # Initial X-axis rotation angle
y_angle = 0  # Initial Y-axis rotation angle

# Function to rotate points around the origin by an angle in degrees
def rotate_point(point, angle, origin):
    angle_rad = np.deg2rad(angle)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    return np.dot(rotation_matrix, np.array(point) - np.array(origin)) + np.array(origin)

# Function to draw axes and store origin with rotations
def draw_axes(image, origin, x_angle):
    # Rotate the X and Y axes independently
    x_end_rot = (0, origin[1] - origin[0]*np.tan(np.deg2rad(x_angle)) )  # X-axis points to the left
    y_end_rot = (origin[0] + origin[1]*np.tan(np.deg2rad(x_angle)), 0)  # Y-axis points upwards
    
    print(f"x_angle: {x_angle}")
    print(f"origin: {origin}")
    print("Axes x and y: ",tuple(map(int, x_end_rot)), tuple(map(int, y_end_rot)))

    # Draw the rotated axes
    cv2.line(image, origin, tuple(map(int, x_end_rot)), (0, 0, 255), 2)  # Red X-axis
    cv2.line(image, origin, tuple(map(int, y_end_rot)), (0, 255, 0), 2)  # Green Y-axis

    # Mark the origin
    cv2.circle(image, origin, 5, (255, 0, 0), -1)  # Blue origin

    # Store the coordinates of the axes
    x_axis = (origin, tuple(map(int, x_end_rot)))  # Rotated coordinates of x-axis
    y_axis = (origin, tuple(map(int, y_end_rot)))  # Rotated coordinates of y-axis
    

    return image, x_axis, y_axis

# Slider callback function to update the origin and rotation angles
def update_origin(x):
    global origin, x_angle
    origin = (cv2.getTrackbarPos("X Origin", "Image"), cv2.getTrackbarPos("Y Origin", "Image"))
    x_angle = cv2.getTrackbarPos("X Angle", "Image")
        
    updated_image, x_axis, y_axis = draw_axes(image.copy(), origin, x_angle)
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", updated_image.shape[1], updated_image.shape[0])
    cv2.imshow("Image", updated_image)

img2_path = os.path.join(os.getcwd(), "src/task3/test_images/d15.png")
image = cv2.imread(img2_path)
if image is None:
    print("Image not found")
    exit()

# Create a window and add sliders to control the origin and axis rotation
cv2.namedWindow("Image")
cv2.createTrackbar("X Origin", "Image", origin[0], image.shape[1] - 1, update_origin)
cv2.createTrackbar("Y Origin", "Image", origin[1], image.shape[0] - 1, update_origin)
cv2.createTrackbar("X Angle", "Image", x_angle, 360, update_origin)  # Slider for rotating X axis

# Draw initial axes
updated_image, x_axis, y_axis = draw_axes(image, origin, x_angle)

# Show the image
# cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Image", updated_image.shape[1], updated_image.shape[0])
cv2.imshow("Image", updated_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

# You can now access the stored axes and origin
print("X Axis:", x_axis)
print("Y Axis:", y_axis)
print("Origin:", origin)
