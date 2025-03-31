import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.task3.saver import get_incremented_filename
from src.task3.feat_extractor import sift, orb
from src.task3.dataset import load_dataset
from src.task3.matcher import match_feats, find_homography

def searcher(dataset: list[dict], img1: np.array, detector: str, kp1: list, des1: list) -> list[tuple]:
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
    poses = []
    for i, entry in enumerate(dataset):
        img2_path = os.path.join(os.getcwd(), "src/task2/test_images/"+ entry["figure"])
        img2 = cv2.imread(img2_path)
        if img2 is None:
            print("Image not found")
            continue
        
        if detector == "sift":
            kp2, des2 = sift(img2, visualize=True)
        elif detector == "orb":
            kp2, des2 = orb(img2, visualize=True)
        
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
            
            poses.append(pose)          
                    
    return poses
        
if __name__ == "__main__":
    default_name = "t2"
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
        kp1, des1 = sift(img1, visualize=True)
    elif detector == "orb":
        kp1, des1 = orb(img1, visualize=True)
    
    
    result = searcher(dataset, img1, detector, kp1, des1)

    #To_DO
    # print("Displaying the images...")
    
    # show_images((result, "Result"), save_flag=True)