import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors


def remove_background(pcd_back: np.array, pcd_curr: np.array, k = 1, thres = 0.05, visualize = False):
    """
    Removes the background points from the current point cloud by comparing it with a reference background point cloud.
    Args:
        pcd_back (np.array): The reference background point cloud as a NumPy array of shape (N, 3).
        pcd_curr (np.array): The current point cloud as a NumPy array of shape (N, 3).
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
    