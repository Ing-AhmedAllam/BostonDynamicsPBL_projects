import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyrealsense2 as rs

from src.task3.saver import get_incremented_filename
from src.task3.feat_extractor import sift, orb
from src.task3.dataset import load_dataset
from src.task3.matcher import match_feats, find_homography
from src.task3.visualizer import visualize_pose, show_images, blend
    
def convert_to_intrinsics(intrinsics_dict):
    intrinsics = rs.intrinsics()
    intrinsics.width = intrinsics_dict['width']
    intrinsics.height = intrinsics_dict['height']
    intrinsics.ppx = intrinsics_dict['ppx']
    intrinsics.ppy = intrinsics_dict['ppy']
    intrinsics.fx = intrinsics_dict['fx']
    intrinsics.fy = intrinsics_dict['fy']
    intrinsics.model = intrinsics_dict['distortion']
    intrinsics.coeffs = intrinsics_dict['coeffs']
    return intrinsics

def get_depth(depth_image: np.array, x: int, y: int, cam_param: list) -> float:
    #Get the depth value at the pixel (x, y)
    # Get the depth value at the pixel (x, y)
    intrinsics = convert_to_intrinsics(cam_param[1])
    depth_value = depth_image[y, x]  # Depth at (x, y)

    if depth_value == 0:  # No valid depth at this pixel
        return None

    # Compute the 3D point in the camera coordinate system
    # Using depth_intrinsics to convert from pixel coordinates to camera space (3D)
    depth_pixel = [x, y]
    depth_point = rs.rs2_deproject_pixel_to_point(intrinsics, depth_pixel, depth_value)

    return depth_point

def searcher(dataset: list, img1: np.array, detector: str, kp1: list, des1: list,depth_image, cam_param: list) -> list:
    """
    Searches for matching features between a given image and a dataset of images, and computes the poses of matched objects.
    Args:
        dataset (list[dict]): A list of dictionaries, where each dictionary contains information about an image 
                              and its associated geometric properties (e.g., origin, major axis, minor axis).
        img1 (np.array): The input image (query image) represented as a NumPy array.
        detector (str): The feature detector to use. Options are "sift" or "orb".
        kp1 (list): Keypoints detected in the input image `img1`.
        des1 (list): Descriptors corresponding to the keypoints in `img1`.
    Returns:
        list[tuple]: A list of poses for matched objects. Each pose is represented as a tuple containing:
                     - center (np.array): The transformed center of the object.
                     - major_axis_info (tuple): A tuple containing:
                         - major_axis_end (np.array): The transformed endpoint of the major axis.
                         - major_axis_length (float): The length of the major axis.
                     - minor_axis_info (tuple): A tuple containing:
                         - minor_axis_end (np.array): The transformed endpoint of the minor axis.
                         - minor_axis_length (float): The length of the minor axis.
                     - angle (float): The angle of the major axis with respect to the horizontal axis, in radians.
    Notes:
        - The function uses homography to transform the geometric properties of objects in the dataset to match the input image.
        - If an image in the dataset cannot be loaded, it is skipped with a warning message.
        - Visualization of keypoints and matches can be enabled by setting the `visualize` parameter in the respective functions.
    """
    cnt = 0
    for i, entry in enumerate(dataset):
        img2_path = os.path.join(os.getcwd(), "src/task3/test_images/"+ entry["figure"])
        img2 = cv2.imread(img2_path)
        if img2 is None:
            print("Image not found")
            continue
        
        if detector == "sift":
            kp2, des2 = sift(img2, visualize=True)
        elif detector == "orb":
            kp2, des2 = orb(img2, visualize=True)
        
        if len(kp2) == 0 or len(des2) == 0:
            print("No keypoints or descriptors found in image 2")
            continue
        if des1.shape[0] <=1 or des2.shape[0] <= 1:
            print("Not enough descriptors found")
            continue
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
            
            
            
            #get depth of pose center
            if depth_image is not None and cam_param is not None:
                depth = get_depth(depth_image, int(center[0]), int(center[1]), cam_param)
            else:
                depth = [0,0,0]
                
            if depth is not None:
                center = np.append(center, depth[2])  # Append the depth (z-coordinate) to the center array
            else:
                center = np.append(center, 0)  # Append 0 if depth is not available
            
            pose = center, major_axis_info, minor_axis_info, angle
                        
            blended_img = blend(img1, img2, H)
            
            #Visualize the blend and save it
            plt.imshow(cv2.cvtColor(blended_img, cv2.COLOR_BGR2RGB))
            plt.title(f"Blended Image {cnt}")
            plt.axis('off')

            # Save the blended image with an incremented filename
            filename = get_incremented_filename("blend", "png")
            plt.savefig(filename)
            plt.close()
            
            img1 = visualize_pose(img1,pose,cnt)       
                    
    return img1

def load_camera_parameters_np(file_name="camera_params.npy"):
    # Load the numpy structured array containing the camera parameters
    camera_params = np.load(file_name, allow_pickle=True)

    # Extract color and depth intrinsics
    color_intrinsics = camera_params['color_intrinsics'][0]
    depth_intrinsics = camera_params['depth_intrinsics'][0]

    return color_intrinsics, depth_intrinsics
        
if __name__ == "__main__":
    default_name = "t6.png"
    camera_param_path = os.path.join(os.getcwd(), "src/task3/test_images/camera_params.npy")
    if len(sys.argv) > 1:
        detector = sys.argv[1]
        if len(sys.argv) == 5:
            detector = sys.argv[1]
            img1_path = sys.argv[2]
            dataset_path = sys.argv[3]
            depth_flag = sys.argv[4]
        else:
            img1_path = os.path.join(os.getcwd(), "src/task3/test_images/"+default_name)
            dataset_path = os.path.join(os.getcwd(), "src/task3/dataset.csv")
            print("Usage: python trial.py <detector> <image1>")
            depth_flag = "False"
    else:
        detector = "sift"
        img1_path = os.path.join(os.getcwd(), "src/task3/test_images/"+default_name)
        dataset_path = os.path.join(os.getcwd(), "src/task3/dataset.csv")
        depth_flag = "False"
    
    img1 = cv2.imread(img1_path)
    
    if depth_flag == "True": 
        depth_path = os.path.join(os.getcwd(), "src/task3/test_images/"+default_name)
        depth_image = np.load(depth_path+'.npy')  # Replace with your actual filename
        cam_param = load_camera_parameters_np(camera_param_path)
        
        if depth_image is None:
            print("Depth image not found")
            exit()
    else:
        depth_image = None
        cam_param = None
        
    if img1 is None:
        print("Image not found")
        exit()
    
    
    dataset = load_dataset(dataset_path)
    
    print(f"Using {detector} detector")
    if detector == "sift":
        kp1, des1 = sift(img1, visualize=True)
    elif detector == "orb":
        kp1, des1 = orb(img1, visualize=True)
    
    
    result = searcher(dataset, img1, detector, kp1, des1, depth_image, cam_param)

    #To_DO
    print("Displaying the images...")
    
    show_images((result, "Result"), save_flag=True)