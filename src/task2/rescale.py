import os
import cv2

# Load the image
img2_path = os.path.join(os.getcwd(), "src/task_2/t3.jpg")

image = cv2.imread(img2_path)

# Scaling factors (e.g., 50% size reduction)
scale_percent = 50  
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)

# Resize the image
resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

# Display the resized image
cv2.imshow("Resized Image", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite(img2_path, resized_image)
