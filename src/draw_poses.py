import numpy as np
import cv2

def visualize_pose(image, poses, arrow_length_factor=0.8, save_path=None):
    """
    Visualize the pose estimation results on the original image.
    
    Args:
        image (numpy.ndarray): The original image
        poses (list): A list of pose information returned by pose_estimate function
        arrow_length_factor (float): Factor to determine the length of the arrow relative to the axis length
        save_path (str, optional): Path to save the visualization. If None, the image will be displayed.
    
    Returns:
        numpy.ndarray: The visualization image
    """
    # Make a copy of the original image to draw on
    vis_image = image.copy()
    
    # Define colors for visualization
    center_color = (0, 0, 255)  # Red for center
    major_axis_color = (0, 255, 0)  # Green for major axis
    minor_axis_color = (255, 0, 0)  # Blue for minor axis
    arrow_color = (255, 255, 0)  # Yellow for arrows
    
    for i, pose in enumerate(poses):
        contour, center, major_axis_info, minor_axis_info, angle = pose
        
        # Draw the contour
        cv2.drawContours(vis_image, [contour.astype(np.int32)], 0, (255, 255, 255), 2)
        
        # Extract center coordinates (convert to integers for drawing)
        center_x, center_y = int(center[0]), int(center[1])
        
        # Draw center point
        cv2.circle(vis_image, (center_x, center_y), 5, center_color, -1)
        
        # Unpack axis information
        major_axis, major_length = major_axis_info
        minor_axis, minor_length = minor_axis_info
        
        # Draw major axis line
        major_end_x = int(center_x + major_length * major_axis[0])
        major_end_y = int(center_y + major_length * major_axis[1])
        cv2.line(vis_image, (center_x, center_y), (major_end_x, major_end_y), major_axis_color, 2)
        
        # Draw minor axis line
        minor_end_x = int(center_x + minor_length * minor_axis[0])
        minor_end_y = int(center_y + minor_length * minor_axis[1])
        cv2.line(vis_image, (center_x, center_y), (minor_end_x, minor_end_y), minor_axis_color, 2)
        
        # Draw arrows at specific distances along the axes
        draw_directional_arrow(vis_image, center, major_axis, major_length, arrow_length_factor, arrow_color, thickness=2)
        draw_directional_arrow(vis_image, center, minor_axis, minor_length, arrow_length_factor, arrow_color, thickness=2)
        
        # Draw the gripper
        cv2.line(image,(int(center[0]+minor_axis[0]*(minor_length + 10) -major_axis[0]*30),int(center[1]+minor_axis[1]*(minor_length + 10) -major_axis[1]*30)),(int(center[0]+minor_axis[0]*(minor_length + 10) +major_axis[0]*30),int(center[1]+minor_axis[1]*(minor_length + 10) +major_axis[1]*30)),(255,255,255),2)
        cv2.line(image,(int(center[0]-minor_axis[0]*(minor_length + 10) -major_axis[0]*30),int(center[1]-minor_axis[1]*(minor_length + 10) -major_axis[1]*30)),(int(center[0]-minor_axis[0]*(minor_length + 10) +major_axis[0]*30),int(center[1]-minor_axis[1]*(minor_length + 10) +major_axis[1]*30)),(255,255,255),2)
        
        
        # Add text labels
        cv2.putText(vis_image, f"Object {i}", (center_x + 10, center_y + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_image, f"Angle: {angle * 180 / np.pi:.1f} deg", (center_x + 10, center_y + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Display or save the visualization
    if save_path:
        cv2.imwrite(save_path, vis_image)
    
    return vis_image

def draw_directional_arrow(image, center, axis_direction, axis_length, length_factor, color, thickness=2):
    """
    Draw an arrow pointing in the direction of an axis at a specific distance.
    This arrow represents the direction of the gripper
    
    Args:
        image (numpy.ndarray): The image to draw on
        center (tuple): The center coordinates (x, y)
        axis_direction (numpy.ndarray): The direction vector of the axis
        axis_length (float): The length of the axis
        length_factor (float): The factor to determine where to place the arrow (0-1)
        color (tuple): The color of the arrow (B, G, R)
        thickness (int): The thickness of the arrow lines
    """
    # Calculate the starting point of the arrow (at a specific distance along the axis)
    arrow_start_x = int(center[0] + length_factor * axis_length * axis_direction[0])
    arrow_start_y = int(center[1] + length_factor * axis_length * axis_direction[1])
    
    # Calculate the ending point of the arrow
    arrow_length = 20  # Fixed arrow length
    arrow_end_x = int(arrow_start_x + arrow_length * axis_direction[0])
    arrow_end_y = int(arrow_start_y + arrow_length * axis_direction[1])
    
    # Draw the arrow
    cv2.arrowedLine(image, (arrow_start_x, arrow_start_y), (arrow_end_x, arrow_end_y), 
                   color, thickness, tipLength=0.3)