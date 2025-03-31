import cv2
import numpy as np

def sift_feature_matching_ransac(img1, img2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Create SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect and compute features
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    # Create FLANN matcher
    index_params = dict(algorithm=1, trees=5)  # KDTree for SIFT
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Match descriptors using KNN
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Apply Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    # Extract matched keypoints
    if len(good_matches) > 8:  # At least 8 points for fundamental matrix estimation
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find Fundamental Matrix using RANSAC
        F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, 3.0, 0.99)

        # Use only inliers
        inlier_matches = [good_matches[i] for i in range(len(mask)) if mask[i]]

        return kp1, kp2, inlier_matches

    return kp1, kp2, []

# Example usage
if __name__ == "__main__":
    img1 = cv2.imread("image1.jpg")
    img2 = cv2.imread("image2.jpg")
    
    kp1, kp2, filtered_matches = sift_feature_matching_ransac(img1, img2)
    
    # Draw matches
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, filtered_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Filtered Matches (RANSAC on Fundamental Matrix)", match_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
