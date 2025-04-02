import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
# from saver import get_incremented_filename
from src.task3.saver import get_incremented_filename


def sift(img1: np.array, visualize=False):
    """
    Extracts SIFT (Scale-Invariant Feature Transform) keypoints and descriptors from an input image.
    Args:
        img1 (np.array): The input image as a NumPy array. Must not be None.
        visualize (bool, optional): If True, visualizes the detected keypoints on the image 
                                     and saves the visualization to a file. Defaults to False.
    Returns:
        tuple: A tuple containing:
            - kp1 (list): A list of detected keypoints.
            - des1 (np.array): A NumPy array of descriptors corresponding to the keypoints.
    Raises:
        AssertionError: If the input image `img1` is None.
    Notes:
        - The visualization is saved to a file with an incremented filename using the 
          `get_incremented_filename` function.
        - The visualization is displayed using Matplotlib if `visualize` is True.
    """
    
    assert img1 is not None, "Image 1 is None"
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    
    if visualize:
        #Display the keypoints
        img1_kp = cv2.drawKeypoints(img1, kp1, None)
        
        #Plot both images
        plt.figure(figsize=(10, 6))
        plt.imshow(img1_kp)
        plt.title("Image 1 Keypoints")
        plt.axis("off")
        
        #Save the image
        save_path = get_incremented_filename("sift_feats")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Images saved to {save_path}")
        
        plt.show()
    
    return kp1, des1

def orb(img1: np.array, visualize=False):
    """
    Extracts ORB (Oriented FAST and Rotated BRIEF) keypoints and descriptors from an input image.
    Parameters:
    -----------
    img1 : np.array
        The input image as a NumPy array. Must not be None.
    visualize : bool, optional
        If True, visualizes the detected keypoints on the input image and saves the visualization
        as an image file. Default is False.
    Returns:
    --------
    kp1 : list of cv2.KeyPoint
        A list of detected keypoints in the input image.
    des1 : np.array
        A NumPy array of shape (N, 32), where N is the number of keypoints, containing the descriptors
        for the detected keypoints.
    Raises:
    -------
    AssertionError
        If the input image `img1` is None.
    Notes:
    ------
    - The visualization is saved to a file with an incremented filename starting with "orb_feats".
    - The visualization is displayed using Matplotlib if `visualize` is set to True.
    """
    
    assert img1 is not None, "Image 1 is None"
    
    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    
    if visualize:
        #Display the keypoints
        img1_kp = cv2.drawKeypoints(img1, kp1, None)
        
        #Plot both images
        plt.figure(figsize=(10, 6))
        plt.imshow(img1_kp)
        plt.title("Image 1 Keypoints")
        plt.axis("off")

        #Save the image
        save_path = get_incremented_filename("orb_feats")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Images saved to {save_path}")
        plt.show()
        
                
    return kp1, des1
