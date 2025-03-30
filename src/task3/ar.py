import cv2
import numpy as np
import apriltag

def setup_video_pipeline(video_path=0):
    cap = cv2.VideoCapture(video_path)

    assert cap.isOpened()
    return cap

def process_video(cap, processing_functions=None):
    """
    Process video frames with optional processing functions

    Args:
        cap: OpenCV VideoCapture object
        processing_functions: List of functions to apply to each frame
    """
    if processing_functions is None:
        processing_functions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame captured")
            break

        processed_frame = frame
        for func in processing_functions:
            processed_frame = func(processed_frame)

        cv2.imshow('Processed Frame', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()

def load_calibration():
    import os
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Create the full path to the calibration file
    calibration_path = os.path.join(current_dir, 'camera_calibration.npz')
    calibration = np.load(calibration_path)
    camera_matrix = calibration['camera_matrix']
    dist_coeffs = calibration['dist_coeffs']

    return camera_matrix, dist_coeffs

camera_matrix, dist_coeffs = load_calibration()

def draw_cube(frame):
    """
    Detects AprilTags in a frame and draw a cube on it

    Args:
        frame: Input image frame

    Returns:
        Frame with cube drawn
    """
    processed_frame = frame

    # Convert to grayscale
    gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)

    # Create detector with default parameters
    detector = apriltag.Detector()

    # Detect AprilTags
    results = detector.detect(gray)

    for r in results:
        # 1. Draw detection outlines

        # Convert corners to integer points
        pts = np.array(r.corners, np.int32)
        pts = pts.reshape((-1, 1, 2))

        # Draw outline
        cv2.polylines(processed_frame, [pts], True, (0, 255, 0), 2)

        # Draw center point
        center = (int(r.center[0]), int(r.center[1]))
        cv2.circle(processed_frame, center, 5, (0, 0, 255), -1)

        # Display tag ID
        tag_id = str(r.tag_id)
        cv2.putText(processed_frame, tag_id, (center[0] - 10, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 2. PnP
        half_size = 71 / 2 # 18mm
        object_points = np.array([
            [-half_size, -half_size, 0],  # bottom-left
            [ half_size, -half_size, 0],  # bottom-right
            [ half_size,  half_size, 0],  # top-right
            [-half_size,  half_size, 0]   # top-left
        ], dtype=np.float32)

        image_points = np.array([r.corners for r in results], dtype=np.float32)
        success, rotation_vec, translation_vec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
        rotation_matrix, _ = cv2.Rodrigues(rotation_vec)

        # 3. Draw coordinate axis
        axis_length = 30  # Length of the axis lines
        axis_points = np.float32([[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]])
        axis_img_points, _ = cv2.projectPoints(axis_points, rotation_vec, translation_vec, camera_matrix, dist_coeffs)

        origin = tuple(map(int, axis_img_points[0].ravel()))
        x_point = tuple(map(int, axis_img_points[1].ravel()))
        y_point = tuple(map(int, axis_img_points[2].ravel()))
        z_point = tuple(map(int, axis_img_points[3].ravel()))

        cv2.line(processed_frame, origin, x_point, (0, 0, 255), 2)  # X-axis: Red
        cv2.line(processed_frame, origin, y_point, (0, 255, 0), 2)  # Y-axis: Green
        cv2.line(processed_frame, origin, z_point, (255, 0, 0), 2)  # Z-axis: Blue

        # Draw cube
        cube_size = 25  # Size of the cube
        cube_vertices = np.float32([
            [-cube_size, -cube_size, -cube_size],  # 0: back bottom left
            [cube_size, -cube_size, -cube_size],   # 1: back bottom right
            [cube_size, cube_size, -cube_size],    # 2: back top right
            [-cube_size, cube_size, -cube_size],   # 3: back top left
            [-cube_size, -cube_size, cube_size],   # 4: front bottom left
            [cube_size, -cube_size, cube_size],    # 5: front bottom right
            [cube_size, cube_size, cube_size],     # 6: front top right
            [-cube_size, cube_size, cube_size]     # 7: front top left
        ])

        # Project cube vertices to image plane
        cube_img_points, _ = cv2.projectPoints(cube_vertices, rotation_vec, translation_vec, camera_matrix, dist_coeffs)
        cube_img_points = [tuple(map(int, point.ravel())) for point in cube_img_points]

        # Define edges of the cube
        cube_edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # back face
            (4, 5), (5, 6), (6, 7), (7, 4),  # front face
            (0, 4), (1, 5), (2, 6), (3, 7)   # connecting edges
        ]

        # Draw the edges of the cube
        for edge in cube_edges:
            cv2.line(processed_frame, cube_img_points[edge[0]], cube_img_points[edge[1]], (0, 255, 255), 2)


    return processed_frame

# Example usage
if __name__ == "__main__":
    cap = setup_video_pipeline(0)  # Use 0 for webcam
    if cap:
        # Process with multiple functions
        process_video(cap, [detect_apriltags])
