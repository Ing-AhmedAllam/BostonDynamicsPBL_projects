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
        color = (255, 255, 0) #yellow

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