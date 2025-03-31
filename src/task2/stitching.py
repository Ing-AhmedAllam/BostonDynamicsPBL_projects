import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_feats(img1: np.array, img2: np.array, visualize=False):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    if visualize:
        #Display the keypoints
        img1_kp = cv2.drawKeypoints(img1, kp1, None)
        img2_kp = cv2.drawKeypoints(img2, kp2, None)
        
        #Plot both images
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(img1_kp)
        plt.title("Image 1 Keypoints")
        plt.subplot(1, 2, 2)
        plt.imshow(img2_kp)
        plt.title("Image 2 Keypoints")
        plt.axis("off")
        

        save_path = get_incremented_filename("out")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Images saved to {save_path}")
        
        plt.show()
    
    return kp1, des1, kp2, des2

def detect_feats_orb(img1: np.array, img2: np.array, visualize=False):
    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    if visualize:
        #Display the keypoints
        img1_kp = cv2.drawKeypoints(img1, kp1, None)
        img2_kp = cv2.drawKeypoints(img2, kp2, None)
        
        #Plot both images
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(img1_kp)
        plt.title("Image 1 Keypoints")
        plt.subplot(1, 2, 2)
        plt.imshow(img2_kp)
        plt.title("Image 2 Keypoints")
        plt.axis("off")

        save_path = get_incremented_filename("out")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Images saved to {save_path}")
        plt.show()
        
                
    return kp1, des1, kp2, des2

def match_feats(kp1, kp2, des1, des2, visulize=False):
    # Use BFMatcher to find matches
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    print(f"Number of matches: {len(matches)}")

    if visulize:
        # Display the matches
        img_matches = cv2.drawMatches(
            img1, kp1, img2, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Display matches using matplotlib
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        plt.title("Matches")
        plt.axis("off")  # Hides axis for cleaner display
        
        save_path = get_incremented_filename("out")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Images saved to {save_path}")
        plt.show()
        
    
    return matches

def find_homography(kp1, kp2, matches):
    # Extract location of good matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    print(f"Homography matrix after:\n{H}")

    return H

def blend(img1,img2, H):
    # Compute the bounding box of the warped image
    h, w = img1.shape[:2]
    pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    warped_pts = cv2.perspectiveTransform(pts, H)

    x_min, y_min = np.int32(warped_pts.min(axis=0).ravel())
    x_max, y_max = np.int32(warped_pts.max(axis=0).ravel())

    # Calculate canvas size
    canvas_width = img1.shape[1] + img2.shape[1]  
    canvas_height = img1.shape[0] + img2.shape[0]

    # Warp with the expanded canvas size
    result = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Shift the image to the correct position
    # The shift is needed because the warped image may have negative coordinates
    shift_x = max(0, x_min)
    shift_y = max(0, y_min)
    im2x = max(0, -x_min)
    im2y = max(0, -y_min)
    print(f"Shifts:\n  \tFirst Image: ({shift_x}, {shift_y})\n\tSecond Image: ({im2x}, {im2y})")

    # Copy the images to the correct position
    result[shift_y:img1.shape[0]+shift_y, shift_x:img1.shape[1]+shift_x] = img1
    result[im2y:img2.shape[0]+im2y, im2x:img2.shape[1]+im2x] = img2

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

if __name__ == "__main__":
    def1 = "image8.jpg"
    def2 = "image9.jpg"
    if len(sys.argv) > 1:
        detector = sys.argv[1]
        if len(sys.argv) == 4:
            detector = sys.argv[1]
            img1_path = sys.argv[2]
            img2_path = sys.argv[3]
        else:
            img1_path = os.path.join(os.getcwd(), "src/task2/test_images/"+def1)
            img2_path = os.path.join(os.getcwd(), "src/task2/test_images/"+def2)
            print("Usage: python trial.py <detector> <image1> <image2>")
    else:
        detector = "sift"
        img1_path = os.path.join(os.getcwd(), "src/task2/test_images/"+def1)
        img2_path = os.path.join(os.getcwd(), "src/task2/test_images/"+def2)
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        print("Images not found")
        exit()
    
    print(f"Using {detector} detector")
    if detector == "sift":
        kp1, des1, kp2, des2 = detect_feats(img1, img2, visualize=True)
    elif detector == "orb":
        kp1, des1, kp2, des2 = detect_feats_orb(img1, img2, visualize=True)
    matches = match_feats(kp1, kp2, des1, des2, visulize=True)
    H = find_homography(kp1, kp2, matches)
    result = blend(img1, img2, H)
    show_images([img1, "Image 1"], [img2, "Image 2"], [result, "Blended Image"], save_flag=True)











