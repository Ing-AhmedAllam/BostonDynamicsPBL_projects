#!/usr/bin/env python
# coding: utf-8

# In[3]:


pip install opencv-python matplotlib numpy


# In[4]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# ================== 滤波函数 ==================
def mean_filter(image, kernel_size):
    """
    均值滤波
    :param image: 输入图像
    :param kernel_size: 滤波器大小（奇数）
    :return: 滤波后的图像
    """
    return cv2.blur(image, (kernel_size, kernel_size))

def gaussian_filter(image, kernel_size, sigma=0):
    """
    高斯滤波
    :param image: 输入图像
    :param kernel_size: 滤波器大小（奇数）
    :param sigma: 高斯核标准差，0表示自动计算
    :return: 滤波后的图像
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def median_filter(image, kernel_size):
    """
    中值滤波
    :param image: 输入图像
    :param kernel_size: 滤波器大小（奇数）
    :return: 滤波后的图像
    """
    return cv2.medianBlur(image, kernel_size)

# ================== 滤波函数 ==================


# ================== 边缘检测函数 ==================
def image_gradient(image, dx=1, dy=1, ksize=3):
    """
    计算图像梯度（Sobel算子）
    :param image: 输入图像（灰度图）
    :param dx: x方向导数阶数
    :param dy: y方向导数阶数
    :param ksize: Sobel核大小（3/5/7）
    :return: 梯度幅值图像
    """
    # 计算x和y方向的梯度
    grad_x = cv2.Sobel(image, cv2.CV_64F, dx, dy, ksize=ksize)
    grad_y = cv2.Sobel(image, cv2.CV_64F, dy, dx, ksize=ksize)
    
    # 计算梯度幅值并归一化
    grad = np.sqrt(grad_x**2 + grad_y**2)
    grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return grad

def sharpen_image(image, strength=1.5):
    """
    图像锐化（拉普拉斯算子）
    :param image: 输入图像
    :param strength: 锐化强度（建议0.5~2.0）
    :return: 锐化后的图像
    """
    # 定义锐化核
    kernel = np.array([[0, -1, 0],
                       [-1, 5*strength, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def canny_edge_detection(image, low_threshold=50, high_threshold=150, aperture_size=3):
    """
    Canny边缘检测
    :param image: 输入图像（灰度图）
    :param low_threshold: 低阈值
    :param high_threshold: 高阈值
    :param aperture_size: Sobel算子孔径大小
    :return: 边缘二值图像
    """
    return cv2.Canny(image, low_threshold, high_threshold, apertureSize=aperture_size)

# ================== 边缘检测函数 ==================





# ================== 读取滤波图像 ==================
image = cv2.imread('Pictures/sample.jpg')  # 图片路径
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换颜色空间

# 应用滤波
kernel_size = 7  # 可以修改滤波器大小（必须为奇数）

mean_img = mean_filter(image, kernel_size)
gaussian_img = gaussian_filter(image, kernel_size)
median_img = median_filter(image, kernel_size)
# ================== 读取滤波图像 ==================




# ================== 显示滤波图像 ==================

plt.figure(figsize=(16, 10))

plt.subplot(2, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(mean_img)
plt.title(f'Mean Filter (Size: {kernel_size})')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(gaussian_img)
plt.title(f'Gaussian Filter (Size: {kernel_size})')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(median_img)
plt.title(f'Median Filter (Size: {kernel_size})')
plt.axis('off')

plt.tight_layout()
plt.show()
# ================== 显示滤波图像 ==================


# ================== 读取锐化图像 ==================
image = cv2.imread('Pictures/sample.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用边缘检测
gradient_result = image_gradient(gray_image, dx=1, dy=1, ksize=3)
sharpened_result = sharpen_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # 锐化需要彩色图
canny_result = canny_edge_detection(gray_image)

# ================== 读取锐化图像 ==================


# ================== 显示锐化图像 ==================
plt.figure(figsize=(16, 8))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(gradient_result, cmap='gray')
plt.title('Image Gradient (Sobel)')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(sharpened_result)
plt.title('Sharpened Image')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(canny_result, cmap='gray')
plt.title('Canny Edge Detection')
plt.axis('off')

#plt.subplot(2, 2, 2)
#plt.imshow(gray_image, cmap='gray')
#plt.title('gray image')
#plt.axis('off')

plt.tight_layout()
plt.show()
# ================== 显示锐化图像 ==================


# In[ ]:




