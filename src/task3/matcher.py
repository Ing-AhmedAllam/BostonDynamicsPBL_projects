import cv2
import numpy as np
from matplotlib import pyplot as plt

# from saver import get_incremented_filename
from src.task3.saver import get_incremented_filename


def expand_inliers(inliers: list,kp1: list,kp2: list, all_matches: list, radius=5 ) -> list:
    """
    Expands a set of inlier matches by including additional matches that are spatially 
    close to the inliers within a specified radius.
    Args:
        inliers (list): A list of inlier matches (cv2.DMatch objects) that are already 
                        identified as good matches.
        kp1 (list): A list of keypoints (cv2.KeyPoint objects) from the first image.
        kp2 (list): A list of keypoints (cv2.KeyPoint objects) from the second image.
        all_matches (list): A list of all potential matches (cv2.DMatch objects) between 
                            the two images.
        radius (float, optional): The spatial radius within which a match is considered 
                                    close to an inlier. Defaults to 5.
    Returns:
        list: An expanded list of matches (cv2.DMatch objects) that includes the original 
                inliers and additional matches that are spatially close to the inliers.
    Raises:
        AssertionError: If any of the following conditions are not met:
            - `inliers` is not empty.
            - `all_matches` is not empty.
            - `radius` is positive.
            - `kp1` and `kp2` are not empty.
            - The number of inliers does not exceed the number of keypoints in either image.
            - The number of matches does not exceed the number of keypoints in either image.
    """
    
    assert len(inliers) > 0, "No inliers to expand"
    assert len(all_matches) > 0, "No matches to expand inliers with"
    assert radius > 0, "Radius must be positive"
    assert len(kp1) > 0, "No keypoints in image 1"
    assert len(kp2) > 0, "No keypoints in image 2"
    assert len(inliers) <= len(kp1), "Inliers exceed number of keypoints in image 1"
    assert len(inliers) <= len(kp2), "Inliers exceed number of keypoints in image 2"
    assert len(all_matches) <= len(kp1), "Matches exceed number of keypoints in image 1"
    assert len(all_matches) <= len(kp2), "Matches exceed number of keypoints in image 2"
    
    expanded_matches = inliers.copy()
    
    # Convert inlier keypoints to numpy arrays for spatial search
    inlier_pts1 = np.float32([kp1[m.queryIdx].pt for m in inliers])
    inlier_pts2 = np.float32([kp2[m.trainIdx].pt for m in inliers])
    
    for match in all_matches:
        p1 = np.array(kp1[match.queryIdx].pt)
        p2 = np.array(kp2[match.trainIdx].pt)

        # Check if the point is near any existing inlier
        if np.any(np.linalg.norm(inlier_pts1 - p1, axis=1) < radius) and \
            np.any(np.linalg.norm(inlier_pts2 - p2, axis=1) < radius):
            expanded_matches.append(match)

    return expanded_matches

def match_feats(img1: np.array, img2: np.array, kp1: list, kp2: list, des1: list, des2: list, visulize=False ) -> list:
    """
    Matches features between two images using FLANN-based matching and applies Lowe's ratio test 
    to filter good matches. Optionally visualizes the matches and saves the visualization.
    Args:
        img1 (np.array): The first input image.
        img2 (np.array): The second input image.
        kp1 (list): List of keypoints detected in the first image.
        kp2 (list): List of keypoints detected in the second image.
        des1 (list): List of descriptors corresponding to the keypoints in the first image.
        des2 (list): List of descriptors corresponding to the keypoints in the second image.
        visulize (bool, optional): If True, visualizes the matches and saves the visualization. Defaults to False.
    Returns:
        list: A list of expanded inlier matches after applying RANSAC and optional expansion of inliers.
                Returns an empty list if not enough matches are found.
    Raises:
        AssertionError: If any of the input keypoints or descriptors lists are empty.
    Notes:
        - The function uses FLANN-based matching with KDTree for SIFT descriptors.
        - Lowe's ratio test is applied to filter good matches.
        - RANSAC is used to estimate the fundamental matrix and filter inliers.
        - If `visulize` is True, the matches are displayed and saved to an incremented filename.
    """
    
    assert len(kp1) > 0, "No keypoints in image 1"
    assert len(kp2) > 0, "No keypoints in image 2"
    assert len(des1) > 0, "No descriptors in image 1"
    assert len(des2) > 0, "No descriptors in image 2"

    # Create FLANN matcher
    index_params = dict(algorithm=1, trees=5)  # KDTree for SIFT
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Match descriptors using KNN
    print(f"des1 shape: {des1.shape}")
    print(f"des2 shape: {des2.shape}")
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe’s ratio test
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    
    # Extract matched keypoints
    if len(good_matches) > 8:  # At least 8 points for fundamental matrix estimation
        # src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        # dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # # Find Fundamental Matrix using RANSAC
        # F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, 3.0, 0.99)

        # # Use only inliers
        # inlier_matches = [good_matches[i] for i in range(len(mask)) if mask[i]]
        
        # # Ensure that the number of inliers doesn't exceed the number of keypoints in kp2
        # assert len(inlier_matches) <= len(kp2), "Inliers exceed number of keypoints in image 2"
        # assert len(good_matches) <= len(kp2), "Good matches exceed number of keypoints in image 2"
        
        # expanded_matches = expand_inliers(inlier_matches, kp1, kp2, good_matches, radius=5)
        
        print(f"Number of matches: {len(good_matches)}")
        # print(f"Number of matches after RANSAC: {len(inlier_matches)}")
        # print(f"Number of matches after expanding: {len(expanded_matches)}")
        
        if visulize:
            # Display the matches
            img_matches = cv2.drawMatches(
                img1, kp1, img2, kp2, good_matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            # Display matches using matplotlib

            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
            plt.title("Matches")
            plt.axis("off")
            plt.show()
            
            save_path = get_incremented_filename("matches")
            plt.savefig(save_path)
            print(f"Images saved to {save_path}")
            plt.close()    
        return good_matches
    
    else:
        print("Not enough matches found")
        return []

def find_homography(kp1: list, kp2: list, matches: list) -> np.array:
    """
    Computes the homography matrix that maps points from one image to another 
    using matched keypoints and the RANSAC algorithm.
    Args:
        kp1 (list): List of keypoints from the first image.
        kp2 (list): List of keypoints from the second image.
        matches (list): List of DMatch objects representing the matches between 
                        keypoints in the two images.
    Returns:
        np.array: The 3x3 homography matrix if at least 4 matches are provided 
                  and the computation is successful. Returns None if there are 
                  fewer than 4 matches.
    """
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
