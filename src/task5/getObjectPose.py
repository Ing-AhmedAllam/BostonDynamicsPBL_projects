import numpy as np
import open3d as o3d

from removeBackground import remove_background

def get_object_pose(pcd_back: np.array, pcd_curr: np.array, visualize: bool = False) -> np.array:
    
    #Remove background points
    pcd_curr_filtered = remove_background(pcd_back, pcd_curr, k=1, thres=0.05, visualize=visualize)

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
    