# Description: This file contains the function to estimate the pose of objects using PCA

import numpy as np
from sklearn.decomposition import PCA


def pose_estimate(contours, verbal=False):
    """This function estimates the pose of objects using PCA

    Args:
        contours (list): A list of numpy arrays representing the contours of the objects
        verbal (bool, optional): A flag to print object stats. Defaults to False.

    Returns:
        list: A list of lists containing the pose of each object
    """
    #Check the input    
    assert isinstance(contours, list), '[Error] contours must be a list of numpy arrays'

    #Initialize the pca
    pca = PCA(n_components=2)
    
    poses = []
    for i,contour in enumerate(contours):   
        #Check the contour     
        assert isinstance(contour, np.ndarray), '[Error] contour must be a numpy array'
        assert contour.shape[1] == 2, '[Error] contour must be a 2D array'
        
        #Reshape the contour        
        contour = contour.reshape(-1, 2)
        
        #Fit the PCA
        pca.fit_transform(contour)
        
        #Get center of the object
        center = pca.mean_
        
        #Get principal axes
        major_axis = pca.components_[0]
        minor_axis = pca.components_[1]
        
        #Get the angle of the first principal component
        angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
        
        poses.append([contour, pca.mean_, pca.components_[0], pca.components_[1], angle])
        
        if verbal:
            print(f'#Object {i}')
            print(f'    Object {i} has center {center}')
            print(f'    Object {i} has major axis {major_axis} and minor axis {minor_axis}')
            print(f'    Object {i} is rotated by {angle * 180 / np.pi} degrees')
    
    return poses