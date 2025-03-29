import numpy as np
from sklearn.decomposition import PCA


def pose_estimate(contours: list, verbose=False):
    """This function estimates the pose of objects using PCA

    Args:
        contours (list): A list of numpy arrays representing the contours of the objects
        verbal (bool, optional): A flag to print object stats. Defaults to False.

    Returns:
        list: A list of lists containing the pose of each object
    """
    #Check the input    
    assert isinstance(contours, (list,tuple)), '[Error] contours must be a list of numpy arrays'

    #Initialize the pca
    pca = PCA(n_components=2)
    
    poses = []
    for i,contour in enumerate(contours):   
        #Check the contour     
        assert isinstance(contour, np.ndarray), '[Error] contour must be a numpy array, instead got {}'.format(type(contour))
        assert contour.shape[-1] == 2, '[Error] contour must be a 2D array, instead got {}'.format(contour.shape)
        
        if contour.ndim != 2:
            contour = contour.reshape(-1, 2)
        
        #Reshape the contour        
        contour = contour.reshape(-1, 2)
        
        #Fit the PCA
        contour_pca = pca.fit_transform(contour)
        
        #Get center of the object
        center = pca.mean_
        
        #Get principal axes
        distances = np.max(np.abs(contour_pca), axis=0)
        mean_pca = contour_pca.mean(axis=0)
        
        major_axis = pca.components_[0]
        major_axis_length = distances[0]-mean_pca[0]
        minor_axis = pca.components_[1]
        minor_axis_length = distances[1]-mean_pca[1]
        
        #Get the angle of the first principal component
        angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
        
        poses.append([contour, center, (major_axis, major_axis_length), (minor_axis, minor_axis_length), angle])
        
        if verbose:
            print(f'#Object {i}')
            print(f'    Object {i} has center {center}')
            print(f'    Object {i} has major axis {major_axis} and minor axis {minor_axis}')
            print(f'    Object {i} is rotated by {angle * 180 / np.pi} degrees')
    
    return poses