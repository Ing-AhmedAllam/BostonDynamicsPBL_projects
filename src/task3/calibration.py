"""
Calibrate the camera for PnP
"""
import numpy as np
import cv2
import glob

# Checkerboard dimensions (internal corners, not squares)
CHECKERBOARD = (10, 4)  # Adjust to match your pattern

# Arrays to store object points and image points
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Prepare object points (0,0,0), (1,0,0), (2,0,0) ...
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Get list of calibration images
images = glob.glob('calibration_images/*.jpg')
print(f"Found {len(images)} calibration images.")

for image in images:
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    assert ret, f"Failed to find chessboard corners in {image}"

    objpoints.append(objp)
    # Refine corner positions
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    imgpoints.append(corners2)

    # Draw and display the corners (optional)
    cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
    cv2.imshow('Chessboard Detection', img)
    cv2.waitKey(500)

# Calibrate camera
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

# Save calibration results
np.savez('camera_calibration.npz',
         camera_matrix=camera_matrix,
         dist_coeffs=dist_coeffs)

print("Camera matrix:")
print(camera_matrix)
print("\nDistortion coefficients:")
print(dist_coeffs)

cv2.destroyAllWindows()
