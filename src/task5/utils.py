import os
import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans
from scipy.signal import find_peaks
from scipy.ndimage import convolve1d
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

def remove_background(pcd_back: np.array, pcd_curr: np.array, k = 1, thres = 0.05, visualize = False) -> np.array:
    """
    Removes the background points from the current point cloud by comparing it with a reference background point cloud.
    Args:
        pcd_back (np.array): The reference background point cloud as a NumPy array of shape (3, N).
        pcd_curr (np.array): The current point cloud as a NumPy array of shape (3, N).
        k (int, optional): The number of nearest neighbors to consider. Defaults to 1.
        thres (float, optional): The distance threshold to determine whether a point is part of the background. 
                                    Points with distances less than this threshold are removed. Defaults to 0.05.
        visualize (bool, optional): If True, visualizes the original background, current point cloud, and filtered 
                                        point cloud using Open3D. Defaults to False.
    Returns:
        np.array: The filtered current point cloud with background points removed.
    Raises:
        AssertionError: If the shapes of `pcd_back` and `pcd_curr` do not match.
    """
    
    assert pcd_back.shape == pcd_curr.shape, "Point clouds must have the same shape" 
    
    # Create a NearestNeighbors object
    knn = NearestNeighbors(n_neighbors=k)
    
    # Fit the model to the original point cloud
    neighbours = knn.fit(pcd_back)
    # Find the nearest neighbors for the current point cloud
    distances, indices = neighbours.kneighbors(pcd_curr)
    
    #Create Mask to remove points that are too close to the background
    mask = distances[:, 0] > thres
    # Apply the mask to the current point cloud
    pcd_curr_filtered = pcd_curr[mask]
    
    if visualize:
        print("Visualizing the point clouds")
        pcd_back_o3d = o3d.geometry.PointCloud()
        pcd_back_o3d.points = o3d.utility.Vector3dVector(pcd_back)
        pcd_back_o3d.paint_uniform_color([0.5, 0.5, 0.5]) # gray color
        
        pcd_curr_o3d = o3d.geometry.PointCloud()
        pcd_curr_o3d.points = o3d.utility.Vector3dVector(pcd_curr)
        pcd_curr_o3d.paint_uniform_color([1, 0, 0]) # red color
        
        pcd_curr_filtered_o3d = o3d.geometry.PointCloud()
        pcd_curr_filtered_o3d.points = o3d.utility.Vector3dVector(pcd_curr_filtered)
        pcd_curr_filtered_o3d.paint_uniform_color([0, 1, 0]) # green color
        
        # Visualize the point clouds
        o3d.visualization.draw_geometries([pcd_back_o3d, pcd_curr_o3d, pcd_curr_filtered_o3d],
                                          window_name="Point Cloud Visualization",
                                          width=800, height=600)
        print("Press 'q' to close the visualization window")
    return pcd_curr_filtered

def focus_on_platform(pcd: np.array, view_points: np.array) -> np.array:
    """
    Filters a point cloud to retain only the points within specified view points.
    
    Args:
        pcd (np.array): A numpy array of shape (N, 3) representing the point cloud, 
                        where N is the number of points and each point has (x, y, z) coordinates.
        view_points (np.array): A numpy array of shape (3, 2) representing the view points, 
                                where each row corresponds to the minimum and maximum bounds 
                                for the x, y, and z dimensions respectively.
    
    Returns:
        np.array: A filtered numpy array of shape (M,3) containing only the points from the input point cloud 
                  that lie within the specified view points.
    
    Raises:
        AssertionError: If the input `pcd` does not have shape (N, 3).
        AssertionError: If the input `view_points` does not have shape (3, 2).
    """
    
    assert pcd.shape[1] == 3, "Point cloud must have shape (N, 3)"
    assert view_points.shape == (3,2), "View points must have shape (3,2)"
    
    #remove points that are not in the view points
    mask = np.all(np.logical_and(pcd >= view_points[:,0], pcd <= view_points[:,1]), axis=1)
    pcd_filtered = pcd[mask]
    
    return pcd_filtered
    
def get_clusters(pcd_seg: np.array, objects_num: int, visualize: bool = False) -> tuple:
    """
    Perform K-means clustering on a point cloud dataset and return the clusters and their centers.
    Args:
        pcd_seg (np.array): A NumPy array representing the segmented point cloud data.
        objects_num (int): The number of clusters (objects) to form.
        visualize (bool, optional): A flag to indicate whether to visualize the clustering process. 
                                     Defaults to False.
    Returns:
        tuple: A tuple containing:
            - clusters (dict): A dictionary where the keys are cluster indices (int) and the values 
                               are NumPy arrays of points belonging to each cluster.
            - centers (np.array): A NumPy array of cluster center coordinates.
    """
    
    kmeans = KMeans(n_clusters=objects_num, random_state=0)
    kmeans.fit(pcd_seg)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    clusters = {}
    for i in range(objects_num):
        clusters[i] = pcd_seg[labels == i]
    return clusters, centers

def get_objects() -> dict:
    """
    Loads and returns point cloud data for objects stored in the "data/objects" directory.
    This function reads all `.ply` files in the "data/objects" directory relative to the 
    current script's location, loads their point cloud data, and returns a dictionary 
    where the keys are indices and the values are lists containing the file name and 
    the corresponding point cloud data as a NumPy array.
    Returns:
        dict: A dictionary where each key is an integer index, and each value is a list 
              containing:
              - str: The file name of the point cloud.
              - numpy.ndarray: The point cloud data as a NumPy array.
    Raises:
        FileNotFoundError: If the "data/objects" directory or any `.ply` file is not found.
        Exception: If there is an error reading a `.ply` file.
    """
    
    #Objects paths
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    pcd_path = os.path.join(cur_dir, "data", "objects")
    pcd_files = os.listdir(pcd_path)
    pcd_files_paths = [os.path.join(pcd_path, f) for f in pcd_files if f.endswith(".ply")]
    #Load point clouds
    objects_pcd = {}
    for i, file_path in enumerate(pcd_files_paths):
        pcd = o3d.io.read_point_cloud(file_path)
        pcd = np.asarray(pcd.points)
        objects_pcd[i] = [pcd_files[i], pcd]
    return objects_pcd

def downsample_pcd(pcd: np.array, voxel_size = 0.002 ) -> np.array:
    """
    Downsample a point cloud using a voxel grid filter.
    Parameters:
    -----------
    pcd : np.array
        A numpy array of shape (N, 3) representing the input point cloud, 
        where N is the number of points.
    voxel_size : float, optional
        The size of the voxel grid used for downsampling. Default is 0.002.
    Returns:
    --------
    np.array or None
        A numpy array of shape (M, 3) representing the downsampled point cloud, 
        where M <= N. Returns None if the input point cloud is empty.
    Raises:
    -------
    AssertionError
        If the input point cloud does not have the correct shape (N, 3).
    Notes:
    ------
    This function uses Open3D for the voxel grid downsampling process.
    """

    assert pcd.shape[0] > 0, "Point cloud must have shape (N, 3)"
    assert pcd.shape[1] == 3, "Point cloud must have shape (N, 3)"
    
    if len(pcd) == 0:
        return None
    # Downsample the point cloud using Open3D
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    pcd_downsampled = pcd_o3d.voxel_down_sample(voxel_size=voxel_size)
    pcd_downsampled = np.asarray(pcd_downsampled.points)
        
    return pcd_downsampled

def compute_range(aligned_dim) -> list:
    """
    Computes the range of values for a given array by analyzing its histogram 
    and identifying peaks and valleys.
    Parameters:
    aligned_dim (numpy.ndarray): A 1D array of numerical values to compute the range for.
    Returns:
    list: A list containing two float values [lower_bound, upper_bound] that represent 
          the computed range of the input array.
    The function performs the following steps:
    1. Sorts the input array and creates histogram bins with a step size of 0.01.
    2. Smooths the histogram counts using a 1D convolution with equal weights.
    3. Identifies the peak closest to the midpoint of the histogram.
    4. Finds valleys (local minima) in the inverted histogram to determine the lower 
       and upper bounds of the range.
    5. Returns the computed range based on the identified bounds.
    Notes:
    - If no significant peaks are found, the function defaults to returning the 
      minimum and maximum values of the input array.
    - The function assumes the input array contains numerical values and is not empty.
    """
    
    sorted_vals = np.sort(aligned_dim)
    bins = np.arange(sorted_vals.min(), sorted_vals.max() + 0.01, 0.01)
    hist_counts, _ = np.histogram(sorted_vals, bins=bins)
    smoothed = convolve1d(hist_counts.astype(float), weights=np.array([1/3, 1/3, 1/3]), mode='constant')

    mid_bin_idx = int(np.ceil((0 - sorted_vals.min()) / 0.01))

    max_peaks, max_locs = find_peaks(smoothed)
    max_locs = max_locs[max_peaks > 0.3 * np.max(max_peaks)]
    if len(max_locs) == 0:
        return [sorted_vals.min(), sorted_vals.max()]
    nearest_peak_idx = np.argmin(np.abs(max_locs - mid_bin_idx))
    mid_bin_idx = max_locs[nearest_peak_idx]

    inv_hist = np.max(smoothed) - smoothed
    min_peaks, min_locs = find_peaks(inv_hist)
    min_locs = min_locs[inv_hist[min_locs] < 0.05 * np.max(smoothed)]

    lower = np.max([1] + [loc for loc in min_locs if loc < mid_bin_idx])
    upper = np.min([len(smoothed)-1] + [loc for loc in min_locs if loc > mid_bin_idx])
    value_range = [sorted_vals.min() - 0.005 + (lower - 1) * 0.01,
                    sorted_vals.min() - 0.005 + (upper - 1) * 0.01]
    return value_range

def pca_denoise_pcd(objSegmPts: np.ndarray) -> np.ndarray:
    """
    Denoises a 3D point cloud using Principal Component Analysis (PCA).
    This function takes a 3D point cloud represented as a 3xN numpy array, 
    aligns it using PCA, and filters out points that fall outside the 
    computed range for each principal component.
    Args:
        objSegmPts (np.ndarray): A 3xN numpy array representing the 3D point cloud, 
                                 where N is the number of points.
    Returns:
        np.ndarray: A 3xM numpy array of the filtered 3D points, where M <= N.
    Raises:
        AssertionError: If the input point cloud does not have the shape (3, N) 
                        or if N is not greater than 0.
    Notes:
        - The input point cloud is first transposed to an Nx3 format for PCA processing.
        - The function computes the range of values for each principal component 
          and filters out points that fall outside these ranges.
        - The output is returned in the original 3xN format.
    """
    
    assert objSegmPts.shape[0] == 3, "Point cloud must have shape (3, N)"
    assert objSegmPts.shape[1] > 0, "Point cloud must have shape (3, N)"
    
    # objSegmPts: 3xN numpy array of 3D points
    objSegmPts = objSegmPts.T  # Make it Nx3 for PCA

    # Perform PCA
    pca = PCA(n_components=3)
    aligned_pts = pca.fit_transform(objSegmPts - np.median(objSegmPts, axis=0))
    
    # Compute ranges for all 3 principal components
    ranges = [compute_range(aligned_pts[:, i]) for i in range(3)]

    # Create mask for filtering
    mask = np.ones(aligned_pts.shape[0], dtype=bool)
    for i in range(3):
        mask &= (aligned_pts[:, i] > ranges[i][0]) & (aligned_pts[:, i] < ranges[i][1])

    # Apply mask and return in 3xN format
    return objSegmPts[mask].T

def ngh_denoise_pcd(pcd_np, nb_neighbors=4, std_ratio=1.0) -> np.ndarray:
    """
    Denoise a 3D point cloud using statistical outlier removal.

    Parameters:
        pcd_np (numpy.ndarray): Nx3 array of 3D points
        nb_neighbors (int): Number of neighbors to analyze (like 'NumNeighbors')
        std_ratio (float): Threshold based on standard deviation

    Returns:
        numpy.ndarray: Denoised point cloud as Nx3 array
    """
    # Convert to Open3D PointCloud
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd_np)

    # Apply statistical outlier removal
    pcd_clean, ind = pcd_o3d.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                                        std_ratio=std_ratio)
    # Return denoised numpy array
    return np.asarray(pcd_clean.points)

def compute_centroid(obj_pcd: np.array, R: np.array, t: np.array) -> tuple:
    """
    Computes the centroid and range of a 3D point cloud in the world coordinate frame.
    Parameters:
    -----------
    obj_pcd : np.array
        A 3xN numpy array representing the 3D point cloud, where N is the number of points.
        The array must have a shape of (3, N).
    R : np.array
        A 3x3 numpy array representing the rotation matrix for transforming the point cloud
        to the world coordinate frame.
    t : np.array
        A 3x1 numpy array representing the translation vector for transforming the point cloud
        to the world coordinate frame.
    Returns:
    --------
    tuple
        A tuple containing:
        - surf_centroid_world (np.array): A 3x1 numpy array representing the centroid of the 
          surface in the world coordinate frame.
        - surf_range_world (np.array): A 3x2 numpy array representing the range of the surface 
          in the world coordinate frame. Each row corresponds to the minimum and maximum values 
          along the x, y, and z axes, respectively.
    Raises:
    -------
    AssertionError
        If the input `obj_pcd` does not have a shape of (3, N) or contains no points.
        If the input `R` is not a 3x3 matrix.
        If the input `t` is not a 3x1 vector.
    Notes:
    ------
    - The function computes the centroid of the point cloud in the local coordinate frame
      and filters points within a small range around the centroid along each axis.
    - If no points are found within the range for a specific axis, the entire point cloud
      is used as a fallback for that axis.
    - The centroid and range are transformed to the world coordinate frame using the provided
      rotation matrix `R` and translation vector `t`.
    """
    
    assert obj_pcd.shape[0] == 3, "Point cloud must have shape (3, N)"
    assert obj_pcd.shape[1] > 0, "Point cloud must have shape (3, N)"
    assert R.shape == (3, 3), "Rotation matrix must have shape (3, 3)"
    assert t.shape == (3, 1), "Translation vector must have shape (3, 1)"
    
    surf_centroid = np.mean(obj_pcd, axis=1, keepdims=True)
    
    mask_x = (
        (obj_pcd[1, :] > surf_centroid[1, 0] - 0.005) &
        (obj_pcd[1, :] < surf_centroid[1, 0] + 0.005) &
        (obj_pcd[2, :] > surf_centroid[2, 0] - 0.005) &
        (obj_pcd[2, :] < surf_centroid[2, 0] + 0.005)
    )
    obj_pcd_range_x = obj_pcd[:, mask_x]

    mask_y = (
        (obj_pcd[0, :] > surf_centroid[0, 0] - 0.005) &
        (obj_pcd[0, :] < surf_centroid[0, 0] + 0.005) &
        (obj_pcd[2, :] > surf_centroid[2, 0] - 0.005) &
        (obj_pcd[2, :] < surf_centroid[2, 0] + 0.005)
    )
    obj_pcd_range_y = obj_pcd[:, mask_y]

    mask_z = (
        (obj_pcd[1, :] > surf_centroid[1, 0] - 0.005) &
        (obj_pcd[1, :] < surf_centroid[1, 0] + 0.005) &
        (obj_pcd[0, :] > surf_centroid[0, 0] - 0.005) &
        (obj_pcd[0, :] < surf_centroid[0, 0] + 0.005)
    )
    obj_pcd_range_z = obj_pcd[:, mask_z]
    
        # Fallback if range sets are empty
    if obj_pcd_range_x.shape[1] == 0:
        obj_pcd_range_x = obj_pcd
    if obj_pcd_range_y.shape[1] == 0:
        obj_pcd_range_y = obj_pcd
    if obj_pcd_range_z.shape[1] == 0:
        obj_pcd_range_z = obj_pcd
    
    # Compute surface ranges
    surf_range_x = [np.min(obj_pcd_range_x[0, :]), np.max(obj_pcd_range_x[0, :])]
    surf_range_y = [np.min(obj_pcd_range_y[1, :]), np.max(obj_pcd_range_y[1, :])]
    surf_range_z = [np.min(obj_pcd_range_z[2, :]), np.max(obj_pcd_range_z[2, :])]
    surf_range_world = np.array([surf_range_x, surf_range_y, surf_range_z])
    
    surf_centroid_world = R @ surf_centroid + t
    surf_range_world = R @ surf_range_world + t
    
    return surf_centroid_world, surf_range_world
    
def pca_pose(obj_pcd: np.array, surf_centroid: np.array) -> tuple:
    """
    Computes the PCA-based pose of a 3D object point cloud.

    Args:
        obj_pcd (np.array): A 3xN numpy array representing the 3D point cloud, where N is the number of points.
        surf_centroid (np.array): A 3x1 numpy array representing the centroid of the surface.

    Returns:
        tuple: A tuple containing:
            - surfPCAPoseBin (np.array): A 4x4 numpy array representing the PCA-based pose in homogeneous coordinates.
            - latent_pca (np.array): A 1D numpy array of the explained variances (eigenvalues) of the principal components.
            - score_pca (np.array): A 2D numpy array of the transformed point cloud in the PCA space.
    """
    pca = PCA()
    pca.fit(obj_pcd.T)
    coeff_pca = pca.components_.T
    score_pca = pca.transform(obj_pcd.T)
    latent_pca = pca.explained_variance_

    if latent_pca.shape[0] < 3:
        latent_pca = np.pad(latent_pca, (0, 3 - latent_pca.shape[0]), constant_values=0)

    coeff_pca = np.column_stack([
        coeff_pca[:, 0],
        coeff_pca[:, 1],
        np.cross(coeff_pca[:, 0], coeff_pca[:, 1])
    ])

    surfPCAPoseBin = np.eye(4)
    surfPCAPoseBin[:3, :3] = coeff_pca
    surfPCAPoseBin[:3, 3] = surf_centroid.flatten()

    return surfPCAPoseBin, latent_pca, score_pca

def push_back_object(object_pcd: np.array, homogenous_matrix: np.array, pushBackAxis = np.array([1, 0, -1]).reshape(3, 1)) -> np.array:
    """
    Adjusts the position of a homogeneous transformation matrix to push back an object 
    along a specified axis based on the object's point cloud.
    Parameters:
        object_pcd (np.array): A 3xN numpy array representing the point cloud of the object, 
                                where N is the number of points. The array must have a shape of (3, N).
        homogenous_matrix (np.array): A 4x4 numpy array representing the homogeneous transformation matrix.
        pushBackAxis (np.array, optional): A 3x1 numpy array representing the unit vector along which 
                                            the object should be pushed back. Defaults to np.array([1, 0, -1]).reshape(3, 1).
    Returns:
        np.array: The updated 4x4 homogeneous transformation matrix after applying the push-back operation.
    Raises:
        AssertionError: If the input `object_pcd` does not have a shape of (3, N).
        AssertionError: If the input `homogenous_matrix` does not have a shape of (4, 4).
        AssertionError: If the input `pushBackAxis` does not have a shape of (3, 1).
        AssertionError: If the `pushBackAxis` is not a unit vector.
    Notes:
        - The function computes the maximum x, y, and z limits of the object point cloud and uses the 
            largest value to determine how far to push back the object along the specified axis.
        - The push-back operation modifies the translation component of the homogeneous matrix.
    """
    
    assert object_pcd.shape[0] == 3, "Point cloud must have shape (3, N)"
    assert object_pcd.shape[1] > 0, "Point cloud must have shape (3, N)"
    assert homogenous_matrix.shape == (4, 4), "Homogeneous matrix must have shape (4, 4)"
    assert pushBackAxis.shape == (3, 1), "Push back axis must have shape (3, 1)"
    assert np.linalg.norm(pushBackAxis) == 1, "Push back axis must be a unit vector"
    
    # Compute the maximum x, y, and z limits of the object point cloud
    max_x = np.max(object_pcd[0, :])
    max_y = np.max(object_pcd[1, :])
    max_z = np.max(object_pcd[2, :])
    
    push_back_val = np.max([max_x, max_y, max_z])
    homogenous_matrix[:3, 3] += pushBackAxis * push_back_val
    
    return homogenous_matrix

def icp(obj_pcd: np.array, pcd: np.array, homogenous_matrix: np.array, max_correspondence_distance: float = 0.05) -> tuple:
    """
    Perform Iterative Closest Point (ICP) registration to align a 3D object point cloud to a segmented point cloud.
    Args:
        obj_pcd (np.array): A 3xN numpy array representing the object point cloud, where N is the number of points.
        pcd (np.array): A 3xN numpy array representing the segmented point cloud, where N is the number of points.
        homogenous_matrix (np.array): A 4x4 numpy array representing the initial transformation matrix in homogeneous coordinates.
        max_correspondence_distance (float, optional): The maximum distance between corresponding points to be considered during ICP. Defaults to 0.05.
    Returns:
        tuple: A 4x4 numpy array representing the updated transformation matrix in homogeneous coordinates after ICP alignment.
    Raises:
        AssertionError: If the input point clouds or transformation matrix do not have the expected shapes.
        AssertionError: If the max_correspondence_distance is not greater than 0.
    """
    
    assert obj_pcd.shape[0] == 3, "Point cloud must have shape (3, N)"
    assert obj_pcd.shape[1] > 0, "Point cloud must have shape (3, N)"
    assert pcd.shape[0] == 3, "Point cloud must have shape (3, N)"
    assert pcd.shape[1] > 0, "Point cloud must have shape (3, N)"
    assert homogenous_matrix.shape == (4, 4), "Homogeneous matrix must have shape (4, 4)"
    assert max_correspondence_distance > 0, "Max correspondence distance must be greater than 0"
    
    tmp_obj_model_pts = np.dot(homogenous_matrix[:3, :3], obj_pcd) + np.tile(homogenous_matrix[:3, 3].reshape(-1, 1), (1, obj_pcd.shape[1]))
    
    tmpObjModelCloud = o3d.geometry.PointCloud()
    tmpObjModelCloud.points = o3d.utility.Vector3dVector(tmp_obj_model_pts.T)
    
    objSegCloud = o3d.geometry.PointCloud()
    objSegCloud.points = o3d.utility.Vector3dVector(pcd.T)
    
    icp_result1 = o3d.pipelines.registration.registration_icp(
        objSegCloud, tmpObjModelCloud, 0.1, np.eye(4), 
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    icpRt1 = icp_result1.transformation
    
    tmpObjModelCloud.transform(icpRt1)
    icp_result2 = o3d.pipelines.registration.registration_icp(
        objSegCloud, tmpObjModelCloud, 0.1, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    
    icpRt2 = icp_result2.transformation
    
    icpRT = icpRt2 @ icpRt1
    obj_pose = icpRT @ homogenous_matrix
    
    return obj_pose