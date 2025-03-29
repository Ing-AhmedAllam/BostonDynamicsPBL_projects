# Description: Detects contours in an image

import cv2
import numpy as np

def detect_RGBcontours(image: np.array, verbose=False):
    """This function detects contours in an image

    Args:
        image (np.array): This is the image that the contours will be detected in

    Returns:
        List: A list of contours detected in the image
    """
    
    # Convert image to grayscale and detect contours
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if verbose:
        print(f'Number of contours detected: {len(contours)}')

    return contours

def detect_HSVcontours(image: np.array, verbose=False):
    """This function detects contours in an image using HSV color space

    Args:
        image (np.array): This is the image that the contours will be detected in

    Returns:
        List: A list of contours detected in the image
    """
    
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the lower and upper bounds of the color to detect
    lower_bound = np.array([20, 100, 100])
    upper_bound = np.array([30, 255, 255])
    
    # Create a mask for the color
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if verbose:
        print(f'Number of contours detected: {len(contours)}')

    return contours