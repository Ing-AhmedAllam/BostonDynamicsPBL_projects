import cv2
import matplotlib.pyplot as plt
import os
from time import time

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

def show_images(image1, image2, save_flag= False):
    # Convert images from BGR to RGB for correct color display in matplotlib
    img1 = cv2.cvtColor(image1[0], cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(image2[0], cv2.COLOR_BGR2RGB)

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

    plt.tight_layout()  # Ensures the layout looks clean
    
    if save_flag:
        save_path = get_incremented_filename("out")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Images saved to {save_path}")
        
    plt.show()

def sifting(images: list):
    
    assert len(images) == 2, "Error: Please provide exactly 2 images for comparison."
    
    # Initiate SIFT detector and create a list to store keypoints and descriptors
    keys_n_descriptors = []
    for ind, image in enumerate(images):
        # Initiate SIFT detector
        sift = cv2.SIFT_create()
        
        # Find the keypoints and descriptors with SIFT
        keypoints, descriptors = sift.detectAndCompute(image, None)
        
        keys_n_descriptors.append([keypoints,descriptors])
        
    return keys_n_descriptors

def match_images(keys_n_descriptors: list):
    assert len(keys_n_descriptors) == 2, "Error: Please provide exactly 2 images for comparison."
    
    # Use FLANN-based matcher for efficiency
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    
    # Perform matching
    descriptors = keys_n_descriptors[0][1]
    descriptors2 = keys_n_descriptors[1][1]
    matches = flann.knnMatch(descriptors, descriptors2, k=2)
    
    # Lowe's ratio test to filter good matches
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    
    print(f"Number of matches: {len(good_matches)}")
    
    return good_matches

def draw_matches(image1, image2, keypoints1, keypoints2, good_matches):
    image_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return image_matches

def visualize_matches(image_matches):
    plt.imshow(image_matches)
    plt.title('SIFT Feature Matching')
    plt.axis('off') # Hide axis for a cleaner look
    plt.show()


if __name__ == '__main__':
    # Load images
    img1_path = os.path.join(os.getcwd(), "src/task2/test_images/image2.jpg")
    orig_image = cv2.imread(img1_path)
    image = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)

    img2_path = os.path.join(os.getcwd(), "src/task2/test_images/image3.jpg")
    orig_image2 = cv2.imread(img2_path)
    image2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if image is None and image2 is None:
        print("Error: Images not found or could not be loaded.")
    else:
        print("Images loaded successfully.")
    
    tic = time()
    keys_n_descriptors = sifting([image, image2])
    tac = time()
    print(f"Time taken to extract keypoints and descriptors: {tac - tic:.2f} seconds")
    good_matches = match_images(keys_n_descriptors)
    image_matches = draw_matches(orig_image, orig_image2, keys_n_descriptors[0][0], keys_n_descriptors[1][0], good_matches)
    visualize_matches(image_matches)