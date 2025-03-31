import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_dataset(csv_file_path: str):
    #Load Dataset
    try:
        dataset = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []
    
    required_columns = ["Figure", "origin_x", "origin_y", "major2x", "major2y", "minor2x", "minor2y"]
    if not all(col in dataset.columns for col in required_columns):
        print("Dataset format incorrect or missing required columns")
        return []
    
    # Convert each row into a structured dictionary
    parsed_data = []
    for _, row in dataset.iterrows():
        entry = {
            "figure": row["Figure"],
            "origin": [row["origin_x"], row["origin_y"]],
            "major": {
                "start": [row["origin_x"], row["origin_y"]],
                "end": [row["major2x"], row["major2y"]]
            },
            "minor": {
                "start": [row["origin_x"], row["origin_y"]],
                "end": [row["minor2x"], row["minor2y"]]
            }
        }
        parsed_data.append(entry)

    return parsed_data
            
def detect_feats(img1: np.array, visualize=False):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    
    if visualize:
        #Display the keypoints
        img1_kp = cv2.drawKeypoints(img1, kp1, None)
        
        
        #Display Image
        cv2.namedWindow("Image Keypoints", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image Keypoints", img1_kp.shape[1], img1_kp.shape[0])
        cv2.imshow("Image Keypoints", img1_kp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        save_path = get_incremented_filename("out")
        cv2.imwrite(save_path,img1_kp)
        print(f"Images saved to {save_path}")
    
    return kp1, des1

def detect_feats_orb(img1: np.array, visualize=False):
    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    
    if visualize:
        #Display the keypoints
        img1_kp = cv2.drawKeypoints(img1, kp1, None)
        
        #Display Image
        cv2.namedWindow("Image Keypoints", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image Keypoints", img1_kp.shape[1], img1_kp.shape[0])
        cv2.imshow("Image Keypoints", img1_kp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        save_path = get_incremented_filename("out")
        cv2.imwrite(save_path,img1_kp)
        print(f"Images saved to {save_path}")
    
    return kp1, des1

def match_feats(img1, img2, kp1, kp2, des1, des2, visulize=False):
    # Use BFMatcher to find matches
    # bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    # matches = bf.match(des1, des2)
    # # Sort matches by distance
    # matches = sorted(matches, key=lambda x: x.distance)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Lowe’s ratio test
    matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    
    print(f"Number of matches: {len(matches)}")

    if visulize:
        # Display the matches
        img_matches = cv2.drawMatches(
            img1, kp1, img2, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Display matches using matplotlib
        #Display Image
        cv2.namedWindow("Matches", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Matches", img_matches.shape[1], img_matches.shape[0])
        cv2.imshow("Matches", img_matches)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        save_path = get_incremented_filename("out")
        cv2.imwrite(save_path,img_matches)
        print(f"Images saved to {save_path}")
    
    return matches

def find_homography(kp1, kp2, matches):
    if len(matches) < 4:
        return None
    
    # Extract location of good matches
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    print(f"Homography matrix after:\n{H}")

    return H

def adjust_homography_for_shift(H, shift):
    dx, dy = shift
    T1 = np.array([[1, 0, -dx], [0, 1, -dy], [0, 0, 1]])
    T2 = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])
    return T2 @ H @ T1

def blend(img1, img2, H):
    result = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))
    # result = cv2.addWeighted(img1, 0.5, result, 0.5, 0)
    
    # Blend images by averaging non-zero pixels
    mask = (result == 0)
    result[mask] = img1[mask]

    return result

def get_incremented_filename(base_name, extension=".png"):
    """Generate an incremented filename in the 'tests' folder if the base name already exists."""
    # Go back to the previous directory and then into the 'tests' folder
    save_dir = os.path.join(os.getcwd(), "results/task_2")
    os.makedirs(save_dir, exist_ok=True)  # Create 'results' folder if it doesn't exist

    index = 1
    new_name = os.path.join(save_dir, f"{base_name}{extension}")
    
    while os.path.exists(new_name):
        new_name = os.path.join(save_dir, f"{base_name}_{index}{extension}")
        index += 1
    
    return new_name

def show_images(image1, save_flag= False):    
    if save_flag:
        save_path = get_incremented_filename("out")
        cv2.imwrite(save_path,image1[0])
        print(f"Images saved to {save_path}")
    
    # Display images side by side with titles
    cv2.namedWindow(image1[1], cv2.WINDOW_NORMAL)
    cv2.resizeWindow(image1[1], image1[0].shape[1], image1[0].shape[0])
    cv2.imshow(image1[1], image1[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        

def draw_directional_arrow(image, center, axis_direction, axis_length, length_factor, color, thickness=2):
    arrow_start_x = int((center[0] + length_factor * axis_length * axis_direction[0]).item())
    arrow_start_y = int((center[1] + length_factor * axis_length * axis_direction[1]).item())

    arrow_length = 20
    arrow_end_x = int((arrow_start_x + arrow_length * axis_direction[0]).item())
    arrow_end_y = int((arrow_start_y + arrow_length * axis_direction[1]).item())
    
    try:
        cv2.arrowedLine(image, (arrow_start_x, arrow_start_y), (arrow_end_x, arrow_end_y), 
                    color, thickness, tipLength=0.3)
    except:
        print("failed to draw arrows")

def normalize_vector(vector):
    magnitude = np.linalg.norm(vector)
    if magnitude == 0:
        return vector  # Avoid division by zero
    return vector / magnitude

def visualize_pose(image, pose, obj_id, arrow_length_factor=0.8, save_path=None):
    vis_image = image.copy()

    center_color = (0, 0, 255)
    major_axis_color = (0, 255, 0)
    minor_axis_color = (255, 0, 0)
    arrow_color = (255, 255, 0)

    center, major_axis_info, minor_axis_info, angle = pose

    center_x, center_y = int(center[0].item()), int(center[1].item())

    cv2.circle(vis_image, (center_x, center_y), 5, center_color, -1)

    major_axis, major_length = major_axis_info
    minor_axis, minor_length = minor_axis_info

    major_end_x = int(major_axis[0].item())
    major_end_y = int(major_axis[1].item())
    cv2.line(vis_image, (center_x, center_y), (major_end_x, major_end_y), major_axis_color, 2)

    minor_end_x = int(minor_axis[0].item())
    minor_end_y = int(minor_axis[1].item())
    cv2.line(vis_image, (center_x, center_y), (minor_end_x, minor_end_y), minor_axis_color, 2)

    draw_directional_arrow(vis_image, center, major_axis, major_length, arrow_length_factor, arrow_color, thickness=2)
    draw_directional_arrow(vis_image, center, minor_axis, minor_length, arrow_length_factor, arrow_color, thickness=2)

    
    
    # # Draw the gripper
    # major_axis = normalize_vector(major_axis)
    # minor_axis = normalize_vector(minor_axis)
    # print(major_axis)
    # print(minor_axis)
    # print(minor_length)
    # print(center)
    # cv2.line(vis_image,
    #             (int((center[0]+minor_axis[0]*(minor_length + 10) -major_axis[0]*30).item()),
    #             int((center[1]+minor_axis[1]*(minor_length + 10) -major_axis[1]*30).item())),
    #             (int((center[0]+minor_axis[0]*(minor_length + 10) +major_axis[0]*30).item()),
    #             int((center[1]+minor_axis[1]*(minor_length + 10) +major_axis[1]*30).item())),
    #             (255,255,255),2)
    # cv2.line(vis_image,
    #          (int((center[0]-minor_axis[0]*(minor_length + 10) -major_axis[0]*30).item()),
    #           int((center[1]-minor_axis[1]*(minor_length + 10) -major_axis[1]*30).item())),
    #          (int((center[0]-minor_axis[0]*(minor_length + 10) +major_axis[0]*30).item()),
    #           int((center[1]-minor_axis[1]*(minor_length + 10) +major_axis[1]*30).item())),
    #          (255,255,255),2)

    cv2.putText(vis_image, f"Object {obj_id}", (center_x + 10, center_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    angle = angle[0].item()
    cv2.putText(vis_image, f"Angle: {angle * 180 / np.pi:.1f} deg", (center_x + 10, center_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return vis_image


def searcher(dataset, img1, detector, kp1, des1):
    cnt = 0
    for i, entry in enumerate(dataset):
        img2_path = os.path.join(os.getcwd(), "src/task2/test_images/"+ entry["figure"])
        img2 = cv2.imread(img2_path)
        if img2 is None:
            print("Image not found")
            continue
        
        if detector == "sift":
            kp2, des2 = detect_feats(img2, visualize=True)
        elif detector == "orb":
            kp2, des2 = detect_feats_orb(img2, visualize=True)
        
        matches = match_feats(img1, img2, kp1, kp2, des1, des2, visulize=True)
        H = find_homography(kp1, kp2, matches)
        if H is not None:
            cnt += 1         
            
            old_center = np.array((entry["origin"] + [1])).reshape((-1,1))
            center = np.dot(H, old_center)[:2]
            
            old_major_axis_end = np.array((entry["major"]["end"]+[1])).reshape((-1,1))
            major_axis_end = np.dot(H,old_major_axis_end)[:2]
            major_axis_length = np.sqrt( np.square(major_axis_end[0] - center[0]) + np.square(major_axis_end[1] - center[1]) )
            major_axis_info = (major_axis_end, major_axis_length)
            
            old_minor_axis_end = np.array((entry["minor"]["end"]+[1])).reshape((-1,1))
            minor_axis_end = np.dot(H,old_minor_axis_end)[:2]
            minor_axis_length = np.sqrt( (minor_axis_end[0] - center[0])**2 + (minor_axis_end[1] - center[1])**2 )
            minor_axis_info = (minor_axis_end, minor_axis_length)
            
            angle = np.arctan2(major_axis_end[1] - center[1], major_axis_end[0] - center[0])
            
            
            pose = center, major_axis_info, minor_axis_info, angle
            
            # img1 = blend(img1, img2, H)
            
            img1 = visualize_pose(img1,pose,cnt)
            
            
        
    return img1
        
if __name__ == "__main__":
    default_name = "t3"
    if len(sys.argv) > 1:
        detector = sys.argv[1]
        if len(sys.argv) == 4:
            detector = sys.argv[1]
            img1_path = sys.argv[2]
            dataset_path = sys.argv[3]
        else:
            img1_path = os.path.join(os.getcwd(), "src/task2/test_images/"+default_name+".jpg")
            dataset_path = os.path.join(os.getcwd(), "src/task2/dataset.csv")
            print("Usage: python trial.py <detector> <image1>")
    else:
        detector = "sift"
        img1_path = os.path.join(os.getcwd(), "src/task2/test_images/"+default_name+".jpg")
        dataset_path = os.path.join(os.getcwd(), "src/task2/dataset.csv")
        
    img1 = cv2.imread(img1_path)
    if img1 is None:
        print("Image not found")
        exit()
    
    dataset = load_dataset(dataset_path)
    
    print(f"Using {detector} detector")
    if detector == "sift":
        kp1, des1 = detect_feats(img1, visualize=True)
    elif detector == "orb":
        kp1, des1 = detect_feats_orb(img1, visualize=True)
    
    
    result = searcher(dataset, img1, detector, kp1, des1)
    print("Displaying the images...")
    
    show_images((result, "Result"), save_flag=True)