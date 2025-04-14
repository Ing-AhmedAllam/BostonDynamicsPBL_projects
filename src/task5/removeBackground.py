import os
import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans
from scipy.signal import find_peaks
from scipy.ndimage import convolve1d
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

def remove_background(pcd_back: np.array, pcd_curr: np.array, k = 1, thres = 0.05, visualize = False):
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

def get_objects():
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

def downsample_pcd(pcd: np.array, voxel_size = 0.002 ):
    if len(pcd) == 0:
        return None
    # Downsample the point cloud using Open3D
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    pcd_downsampled = pcd_o3d.voxel_down_sample(voxel_size=voxel_size)
    pcd_downsampled = np.asarray(pcd_downsampled.points)
        
    return pcd_downsampled

def compute_range(aligned_dim):
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

def ngh_denoise_pcd(pcd_np, nb_neighbors=4, std_ratio=1.0):
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
    
def pca_pose(obj_pcd: np.array, surf_centroid: np.array) -> np.array:
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
    surfPCAPoseBin[:3, 3] = surf_centroid
    
    return surfPCAPoseBin, latent_pca, score_pca

def push_back_object(object_pcd: np.array, homogenous_matrix: np.array, pushBackAxis = np.array([1, 0, -1]).reshape(3, 1)) -> np.array:
    
    # Compute the maximum x, y, and z limits of the object point cloud
    max_x = np.max(object_pcd[0, :])
    max_y = np.max(object_pcd[1, :])
    max_z = np.max(object_pcd[2, :])
    
    push_back_val = np.max([max_x, max_y, max_z])
    homogenous_matrix[:3, 3] += pushBackAxis * push_back_val

def icp(obj_pcd: np.array, pcd: np.array, homogenous_matrix: np.array, max_correspondence_distance: float = 0.05) -> tuple:
    tmpObjModelCloud = o3d.geometry.PointCloud()
    tmpObjModelCloud.points = o3d.utility.Vector3dVector(obj_pcd.T)
    
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
    
    #save pose
    #TODO