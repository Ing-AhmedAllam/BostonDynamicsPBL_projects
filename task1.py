from src.task1.Draw_shape import generate_random_shapes, draw_shapes
from src.task1.detect_shapes import detect_RGBcontours, detect_HSVcontours
from src.task1.detect_object import detect
from src.task1.pose_estimate import pose_estimate
from src.task1.draw_poses import visualize_pose

import os
import sys
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt


def get_incremented_filename(base_name, extension=".png"):
    """Generate an incremented filename in the 'tests' folder if the base name already exists."""
    # Go back to the previous directory and then into the 'tests' folder
    save_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(save_dir, exist_ok=True)  # Create 'results' folder if it doesn't exist

    index = 1
    new_name = os.path.join(save_dir, f"{base_name}{extension}")
    
    while os.path.exists(new_name):
        new_name = os.path.join(save_dir, f"{base_name}_{index}{extension}")
        index += 1
    
    return new_name


def show_images(image1, image2, image3, save_flag= False):
    # Convert images from BGR to RGB for correct color display in matplotlib
    img1 = cv2.cvtColor(image1[0], cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(image2[0], cv2.COLOR_BGR2RGB)
    img3 = cv2.cvtColor(image3[0], cv2.COLOR_BGR2RGB)

    # Display images side by side with titles
    plt.figure(figsize=(12, 4))  # Adjust the figure size as needed

    plt.subplot(1, 3, 1)
    plt.imshow(img1)
    plt.title(image1[1])
    plt.axis("off")  # Hide axis for a cleaner look

    plt.subplot(1, 3, 2)
    plt.imshow(img2)
    plt.title(image2[1])
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(img3)
    plt.title(image3[1])
    plt.axis("off")

    plt.tight_layout()  # Ensures the layout looks clean
    
    if save_flag:
        save_path = get_incremented_filename("out")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Images saved to {save_path}")
        
    plt.show()

if __name__ == '__main__':
    
    if len(sys.argv) > 1:
        img_path = sys.argv[1] 
        image = cv2.imread(img_path)

    else:
        image = generate_random_shapes(background_color=(0, 0, 0))
            
    #Detect contours in the image using 3 different methods; RGB, HSV and Canny edge detection
    print("Detecting contours in the image using 3 different methods...")
    print("RGB method:")
    contours_RGB = detect_RGBcontours(image, verbose=True)
    print("HSV method:")
    contours_HSV = detect_HSVcontours(image, verbose=True)
    print("Canny edge detection method:")
    contours_Canny = detect(image)
    
    #Detect poses in the image for the 3 methods
    print("Estimating poses in the image using 3 different methods...")
    print("RGB method:")
    poses_RGB = pose_estimate(contours_RGB, verbose=True)
    print("HSV method:")
    poses_HSV = pose_estimate(contours_HSV, verbose=True)
    print("Canny edge detection method:")
    poses_Canny = pose_estimate(contours_Canny, verbose=True)
    
    #Visualize the poses
    image_RGB = visualize_pose(image, poses_RGB)
    image_HSV = visualize_pose(image, poses_HSV)
    image_Canny = visualize_pose(image, poses_Canny)
        
    #Display the images
    print("Displaying the images...")
    show_images((image_RGB, "RGB"), (image_HSV, "HSV"), (image_Canny, "Canny"), save_flag=True)




