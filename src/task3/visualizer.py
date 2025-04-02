import cv2
import numpy as np
from src.task3.saver import get_incremented_filename

def draw_directional_arrow(image, center, axis_direction, axis_length, length_factor, color, thickness=2):
    arrow_start_x = int((center[0] + length_factor * axis_length * axis_direction[0]).item())
    arrow_start_y = int((center[1] + length_factor * axis_length * axis_direction[1]).item())

    arrow_length = 20
    arrow_end_x = int((arrow_start_x + arrow_length * axis_direction[0]).item())
    arrow_end_y = int((arrow_start_y + arrow_length * axis_direction[1]).item())
    
    try:
        cv2.arrowedLine(image, (arrow_start_x, arrow_start_y), (arrow_end_x, arrow_end_y), 
                    color, thickness, tipLength=0.3)
    except:
        print("failed to draw arrows")

def normalize_vector(vector):
    magnitude = np.linalg.norm(vector)
    if magnitude == 0:
        return vector  # Avoid division by zero
    return vector / magnitude

def blend(img1, img2, H):
    result = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))
    # result = cv2.addWeighted(img1, 0.5, result, 0.5, 0)
    
    # Blend images by averaging non-zero pixels
    mask = (result == 0)
    result[mask] = img1[mask]

    return result

def visualize_pose(image, pose, obj_id, arrow_length_factor=0.8, save_path=None):
    vis_image = image.copy()

    center_color = (0, 0, 255)
    major_axis_color = (0, 255, 0)
    minor_axis_color = (255, 0, 0)
    arrow_color = (255, 255, 0)

    center, major_axis_info, minor_axis_info, angle = pose

    center_x, center_y, depth = int(center[0].item()), int(center[1].item()), int(center[2].item())

    cv2.circle(vis_image, (center_x, center_y), 5, center_color, -1)

    major_axis, major_length = major_axis_info
    minor_axis, minor_length = minor_axis_info

    major_end_x = int(major_axis[0].item())
    major_end_y = int(major_axis[1].item())
    cv2.line(vis_image, (center_x, center_y), (major_end_x, major_end_y), major_axis_color, 2)

    minor_end_x = int(minor_axis[0].item())
    minor_end_y = int(minor_axis[1].item())
    cv2.line(vis_image, (center_x, center_y), (minor_end_x, minor_end_y), minor_axis_color, 2)

    draw_directional_arrow(vis_image, center, major_axis, major_length, arrow_length_factor, arrow_color, thickness=2)
    draw_directional_arrow(vis_image, center, minor_axis, minor_length, arrow_length_factor, arrow_color, thickness=2)

    
    
    # # Draw the gripper
    # major_axis = normalize_vector(major_axis)
    # minor_axis = normalize_vector(minor_axis)
    # print(major_axis)
    # print(minor_axis)
    # print(minor_length)
    # print(center)
    # cv2.line(vis_image,
    #             (int((center[0]+minor_axis[0]*(minor_length + 10) -major_axis[0]*30).item()),
    #             int((center[1]+minor_axis[1]*(minor_length + 10) -major_axis[1]*30).item())),
    #             (int((center[0]+minor_axis[0]*(minor_length + 10) +major_axis[0]*30).item()),
    #             int((center[1]+minor_axis[1]*(minor_length + 10) +major_axis[1]*30).item())),
    #             (255,255,255),2)
    # cv2.line(vis_image,
    #          (int((center[0]-minor_axis[0]*(minor_length + 10) -major_axis[0]*30).item()),
    #           int((center[1]-minor_axis[1]*(minor_length + 10) -major_axis[1]*30).item())),
    #          (int((center[0]-minor_axis[0]*(minor_length + 10) +major_axis[0]*30).item()),
    #           int((center[1]-minor_axis[1]*(minor_length + 10) +major_axis[1]*30).item())),
    #          (255,255,255),2)

    cv2.putText(vis_image, f"Object {obj_id}, Depth: {depth:.2f}", (center_x + 10, center_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    angle = angle[0].item()
    cv2.putText(vis_image, f"Angle: {angle * 180 / np.pi:.1f} deg", (center_x + 10, center_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return vis_image


def show_images(image1, save_flag= False):    
    import matplotlib.pyplot as plt

    if save_flag:
        save_path = get_incremented_filename("out")
        plt.imsave(save_path, cv2.cvtColor(image1[0], cv2.COLOR_BGR2RGB))
        print(f"Images saved to {save_path}")
    
    # Display images using matplotlib
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(image1[0], cv2.COLOR_BGR2RGB))
    plt.title(image1[1])
    plt.axis('off')
    plt.show()