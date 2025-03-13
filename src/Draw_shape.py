import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def generate_random_shapes(width=800, height=600, background_color=(255, 255, 255), 
                           shape_types=["circle", "rectangle", "triangle", "ellipse"],
                           num_shapes=10, min_size=20, max_size=100):
    """
    Generate an image with random geometric shapes

    Parameters:
    width: Image width
    height: Image height
    background_color: Background color in (B, G, R) format
    shape_types: List of geometric shape types to draw
    num_shapes: Number of geometric shapes to draw
    min_size: Minimum shape size
    max_size: Maximum shape size

    Returns:
    img: The generated image
    """

    # Create a background image
    img = np.ones((height, width, 3), dtype=np.uint8)
    img[:] = background_color

    for _ in range(num_shapes):
        # Randomly select a shape
        shape_type = random.choice(shape_types)

        # Random color (B, G, R)
        color = (0, 255, 255) #yellow

        # Random position
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)

        # Random size
        size = random.randint(min_size, max_size)
        
        # Line thickness
        thickness = random.randint(1, 5)
        
        # Whether to fill
        fill = random.choice([True, False])
        
        if shape_type == "circle":
            # Ensure the circle is fully inside the image
            x = min(max(size, x), width - size)
            y = min(max(size, y), height - size)
            if fill:
                cv2.circle(img, (x, y), size, color, -1)
            else:
                cv2.circle(img, (x, y), size, color, thickness)
        elif shape_type == "rectangle":
            # Ensure the rectangle is inside the image
            x2 = min(x + size, width - 1)
            y2 = min(y + size, height - 1)
            if fill:
                cv2.rectangle(img, (x, y), (x2, y2), color, -1)
            else:
                cv2.rectangle(img, (x, y), (x2, y2), color, thickness)
        elif shape_type == "triangle":
            # Create three vertices for the triangle
            pts = np.array([[x, y],
                            [x + random.randint(-size, size), y + random.randint(-size, size)],
                            [x + random.randint(-size, size), y + random.randint(-size, size)]
                            ], np.int32)

            pts = pts.reshape((-1, 1, 2))

            if fill:
                cv2.fillPoly(img, [pts], color)
            else:
                cv2.polylines(img, [pts], True, color, thickness)
        elif shape_type == "ellipse":
            # Ellipse parameters
            axes = (random.randint(10, size), random.randint(10, size))
            angle = random.randint(0, 360)
            if fill:
                cv2.ellipse(img, (x, y), axes, angle, 0, 360, color, -1)
            else:
                cv2.ellipse(img, (x, y), axes, angle, 0, 360, color, thickness)
    
    return img

# Function to display the image
def show_image(img):
    """Display the image in Jupyter Notebook"""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()
    

epsilon = 10 

def is_overlapping(shape, dimensions, existing_shapes):
    """This function detects overlapping shapes

    Args:
        shape (str): A string that specifies the shape's type
        dimensions (list): A list of the shape's dimensions
        existing_shapes (List): A list of lists that contains existing shapes

    Returns:
        Bool: True: if the shape is overlapping and False if not
    """
    
    if shape == 'circle':
        center, radius = dimensions
        for ex_shape, ex_dimensions in existing_shapes:
            if ex_shape == 'circle':
                ex_center, ex_radius = ex_dimensions
                dist = np.sqrt((center[0] - ex_center[0])**2 + (center[1] - ex_center[1])**2)
                if dist < radius + ex_radius + epsilon:
                    return True
            elif ex_shape == 'rectangle':
                start, end = ex_dimensions
                dist1 = np.sqrt((center[0] - start[0])**2 + (center[1] - start[1])**2)
                dist2 = np.sqrt((center[0] - end[0])**2 + (center[1] - end[1])**2)
                dist3 = np.sqrt((center[0] - start[0])**2 + (center[1] - end[1])**2)
                dist4 = np.sqrt((center[0] - end[0])**2 + (center[1] - start[1])**2)
                if dist1 < radius + epsilon or dist2 < radius + epsilon or dist3 < radius + epsilon or dist4 < radius + epsilon:
                    return True
                elif center[0] > start[0] and center[0] < end[0] and center[1] > start[1] and center[1] < end[1]:
                    return True
                elif (center[0] + radius + epsilon > start[0] and center[1] > start[1] and center[1] < end[1]) \
                    or (center[0] - radius - epsilon < end[0] and center[1] > start[1] and center[1] < end[1])\
                    or (center[1] + radius + epsilon > start[1] and center[0] > start[0] and center[0] < end[0])\
                    or (center[1] - radius - epsilon < end[1] and center[0] > start[0] and center[0] < end[0]):
                    return True
            elif ex_shape == 'ellipse':
                ex_center, ex_axes = ex_dimensions
                dist = np.sqrt((center[0] - ex_center[0])**2 + (center[1] - ex_center[1])**2)
                if dist < radius + max(ex_axes[0], ex_axes[1])+ epsilon:
                    return True
    if shape == 'rectangle':
        start, end = dimensions
        for ex_shape, ex_dimensions in existing_shapes:
            if ex_shape == 'circle':
                ex_center, ex_radius = ex_dimensions
                dist1 = np.sqrt((ex_center[0] - start[0])**2 + (ex_center[1] - start[1])**2)
                dist2 = np.sqrt((ex_center[0] - end[0])**2 + (ex_center[1] - end[1])**2)
                dist3 = np.sqrt((ex_center[0] - start[0])**2 + (ex_center[1] - end[1])**2)
                dist4 = np.sqrt((ex_center[0] - end[0])**2 + (ex_center[1] - start[1])**2)
                if dist1 < ex_radius + epsilon or dist2 < ex_radius + epsilon or dist3 < ex_radius + epsilon or dist4 < ex_radius + epsilon:
                    return True
                elif ex_center[0] > start[0] and ex_center[0] < end[0] and ex_center[1] > start[1] and ex_center[1] < end[1]:
                    return True
                elif (ex_center[0] + ex_radius + epsilon > start[0] and ex_center[1] > start[1] and ex_center[1] < end[1]) \
                    or (ex_center[0] - ex_radius - epsilon < end[0] and ex_center[1] > start[1] and ex_center[1] < end[1])\
                    or (ex_center[1] + ex_radius + epsilon > start[1] and ex_center[0] > start[0] and ex_center[0] < end[0])\
                    or (ex_center[1] - ex_radius - epsilon < end[1] and ex_center[0] > start[0] and ex_center[0] < end[0]):
                    return True
            elif ex_shape == 'rectangle':
                ex_start, ex_end = ex_dimensions
                if start[0] < ex_end[0] and end[0] > ex_start[0] and start[1] < ex_end[1] and end[1] > ex_start[1]:
                    return True
            elif ex_shape == 'ellipse':
                ex_center, ex_axes = ex_dimensions
                if start[0] < ex_center[0] + ex_axes[0] and end[0] > ex_center[0] - ex_axes[0] and start[1] < ex_center[1] + ex_axes[1] and end[1] > ex_center[1] - ex_axes[1]:
                    return True
    if shape == 'ellipse':  
        center, axes = dimensions
        for ex_shape, ex_dimensions in existing_shapes:
            if ex_shape == 'circle':
                ex_center, ex_radius = ex_dimensions
                dist = np.sqrt((center[0] - ex_center[0])**2 + (center[1] - ex_center[1])**2)
                if dist < max(axes[0], axes[1]) + ex_radius + epsilon:
                    return True
            elif ex_shape == 'rectangle':
                ex_start, ex_end = ex_dimensions
                if center[0] < ex_end[0] and center[0] > ex_start[0] and center[1] < ex_end[1] and center[1] > ex_start[1]:
                    return True
            elif ex_shape == 'ellipse':
                ex_center, ex_axes = ex_dimensions
                dist = np.sqrt((center[0] - ex_center[0])**2 + (center[1] - ex_center[1])**2)
                if dist < max(axes[0], axes[1]) + max(ex_axes[0], ex_axes[1]) + epsilon:
                    return True
    return False


def draw_shapes(width=800, height=600, background_color=(255, 255, 255), 
                num_shapes = 10 ,shape_types=['circle', 'rectangle', 'ellipse']):
    """This function draws random shapes on the image
    
    Raises:
        ValueError: Raises an error if the shape is not valid
    """
    
    # Create a background image
    image = np.ones((height, width, 3), dtype=np.uint8)
    image[:] = background_color
    
    # Draw random yellow shapes without overlap
    existing_shapes = []
    
    for _ in range(num_shapes):
        # Randomly select a shape
        shape = random.choice(shape_types)
        
        if shape == 'circle':
            while True:
                center = (random.randint(50, width - 50), random.randint(50, height - 50))
                radius = random.randint(30, 100)
                if not is_overlapping(shape, (center, radius), existing_shapes):
                    cv2.circle(image, center, radius, (0, 255, 255), -1)
                    existing_shapes.append((shape, (center, radius)))
                    break
        elif shape == 'rectangle':
            while True:
                start = (random.randint(50, width - 50), random.randint(50, height - 50))
                end = (random.randint(50, width - 50), random.randint(50, height - 50))
                
                if not is_overlapping(shape, (start, end), existing_shapes) and (start != end or start[0] != end[0] or start[1] != end[1]): 
                    cv2.rectangle(image, start, end, (0, 255, 255), -1)
                    existing_shapes.append((shape, (start, end)))
                    break
        elif shape == 'ellipse':
            while True:
                center = (random.randint(50, width - 50), random.randint(50, height - 50))
                axes = (random.randint(30, 100), random.randint(20, 60))
                angle = random.randint(0, 360)
                if not is_overlapping(shape, (center, axes), existing_shapes):
                    cv2.ellipse(image, center, axes, angle, 0, 360, (0, 255, 255), -1)
                    existing_shapes.append((shape, (center, axes)))
                    break
        else:
            raise ValueError('Invalid shape')    
        
    return image, existing_shapes