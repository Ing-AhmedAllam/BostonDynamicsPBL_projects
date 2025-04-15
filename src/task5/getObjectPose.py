import numpy as np
from task5.utils import*

def get_object_pose(pcd_back: np.array, pcd_seg: np.array, objects_num: int, scene_data: dict, visualize: bool = False) -> np.array:
    """
    Computes the pose of objects in a scene using point cloud data.
    Args:
        pcd_back (np.array): Background point cloud of shape (N, 3), where N is the number of points.
        pcd_seg (np.array): Segmented point cloud of shape (N, 3), where N is the number of points.
        objects_num (int): Number of objects to detect in the scene.
        scene_data (dict): Dictionary containing scene transformation data. Must include:
            - 'H': A dictionary with:
                - 'R' (np.array): Rotation matrix of shape (3, 3).
                - 't' (np.array): Translation vector of shape (3, 1).
        visualize (bool, optional): If True, enables visualization of intermediate steps. Defaults to False.
    Returns:
        np.array: Array containing the computed poses of the objects.
    Raises:
        AssertionError: If input data does not meet the required conditions, such as:
            - Point clouds not having the correct shape.
            - Background and segmented point clouds not having the same number of points.
            - Number of objects is not greater than 0.
            - Scene data is missing required keys or has incorrect shapes.
    """
    
    assert pcd_back.shape[1] == 3, "Background point cloud should be Nx3"
    assert pcd_seg.shape[1] == 3, "Segmented point cloud should be Nx3"
    assert pcd_back.shape[0] == pcd_seg.shape[0], "Background and segmented point clouds should have the same number of points"
    assert pcd_back.shape[0] > 0, "Background point cloud should not be empty"
    assert pcd_seg.shape[0] > 0, "Segmented point cloud should not be empty"
    assert objects_num > 0, "Number of objects should be greater than 0"
    assert scene_data is not None, "Scene data should not be None"
    assert 'H' in scene_data, "Scene data should contain 'H' key"
    assert 'R' in scene_data['H'], "Scene data should contain 'R' key"
    assert 't' in scene_data['H'], "Scene data should contain 't' key"
    assert scene_data['H']['R'].shape == (3, 3), "Rotation matrix should be 3x3"
    assert scene_data['H']['t'].shape == (3, 1), "Translation vector should be 3x1"
    
    frames = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    view_bounds = np.array([[-0.01, 0.40], [-0.17, 0.17], [-0.06, 0.20]])
    push_back_axis = np.array([1, 0, -1])
    
    #Remove background points
    pcd_seg_filtered = remove_background(pcd_back, pcd_seg, k=1, thres=0.05, visualize=visualize)
    
    #Remove points that are outside the shelves or tote
    pcd_seg_focused = focus_on_platform(pcd_seg_filtered, view_points=view_bounds, visualize=visualize)
    
    #Get the centroid of the object
    instances_pcd,pcd_seg_centroids = get_clusters(pcd_seg_focused, objects_num, visualize=visualize)
    
    #Get objects models
    objects_pcd = get_objects()
    #downsample the objects point cloud
    objects_pcd_downsampled = downsample_pcd(objects_pcd)
    
    #*****************************DO I HAVE TO DENOISE OBJECTS POINT CLOUDS?*************************
    
    for ind, object_pcd in enumerate(objects_pcd_downsampled):
        #load the object point cloud from the segmented point cloud
        cur_obj_pcd = instances_pcd[ind]
        
        #Denoise and Downsample the segemented object point cloud to the same size as the object model
        denoised_cur_obj_pcd = pca_denoise_pcd(cur_obj_pcd)
        downsampled_cur_obj_pcd =downsample_pcd(denoised_cur_obj_pcd)
        denoised_cur_obj_pcd = ngh_denoise_pcd(downsampled_cur_obj_pcd)
        
        # #Compute surface means and centroids
        # ranges,centroids = compute_centroid(denoised_cur_obj_pcd, scene_data['H']['R'], scene_data['H']['t'])
        
        #perform PCA on the object point cloud
        surfPCAPoseBin, latent_pca, score_pca = pca_pose(denoised_cur_obj_pcd, pcd_seg_centroids[ind])
        
        #Push back the object point cloud to the surface PCA pose bin
        surfPCAPoseBin = push_back_object(denoised_cur_obj_pcd, surfPCAPoseBin,push_back_axis)
        
        #Get the object pose
        obj_pose = icp(objects_pcd_downsampled[ind],denoised_cur_obj_pcd, surfPCAPoseBin)
        
        return obj_pose