#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def generate_random_shapes(width=800, height=600, background_color=(255, 255, 255), 
                           shape_types=["circle", "rectangle", "triangle", "ellipse", "line"],
                           num_shapes=10, min_size=20, max_size=100):
    """
    生成带有随机几何图形的图像
    
    参数:
    width: 图像宽度
    height: 图像高度
    background_color: 背景颜色，格式为(B,G,R)
    shape_types: 要绘制的几何图形类型列表
    num_shapes: 要绘制的几何图形数量
    min_size: 最小图形尺寸
    max_size: 最大图形尺寸
    
    返回:
    img: 生成的图像
    """
    
    # 创建背景图像
    img = np.ones((height, width, 3), dtype=np.uint8)
    img[:] = background_color
    
    for _ in range(num_shapes):
        # 随机选择一种形状
        shape_type = random.choice(shape_types)
        
        # 随机颜色 (B,G,R)
        color = (random.randint(0, 0), random.randint(0, 0), random.randint(0, 0))
        
        # 随机位置
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        
        # 随机尺寸
        size = random.randint(min_size, max_size)
        
        # 线条厚度
        thickness = random.randint(1, 5)
        
        # 是否填充
        fill = random.choice([True, False])
        
        if shape_type == "circle":
            # 确保圆完全在图像内
            x = min(max(size, x), width - size)
            y = min(max(size, y), height - size)
            if fill:
                cv2.circle(img, (x, y), size, color, -1)
            else:
                cv2.circle(img, (x, y), size, color, thickness)
                
        elif shape_type == "rectangle":
            # 确保矩形在图像内
            x2 = min(x + size, width - 1)
            y2 = min(y + size, height - 1)
            if fill:
                cv2.rectangle(img, (x, y), (x2, y2), color, -1)
            else:
                cv2.rectangle(img, (x, y), (x2, y2), color, thickness)
                
        elif shape_type == "triangle":
            # 创建三角形的三个顶点
            pts = np.array([
                [x, y],
                [x + random.randint(-size, size), y + random.randint(-size, size)],
                [x + random.randint(-size, size), y + random.randint(-size, size)]
            ], np.int32)
            pts = pts.reshape((-1, 1, 2))
            if fill:
                cv2.fillPoly(img, [pts], color)
            else:
                cv2.polylines(img, [pts], True, color, thickness)
                
        elif shape_type == "ellipse":
            # 椭圆参数
            axes = (random.randint(10, size), random.randint(10, size))
            angle = random.randint(0, 360)
            if fill:
                cv2.ellipse(img, (x, y), axes, angle, 0, 360, color, -1)
            else:
                cv2.ellipse(img, (x, y), axes, angle, 0, 360, color, thickness)
                
        elif shape_type == "line":
            # 线条终点
            x2 = x + random.randint(-size, size)
            y2 = y + random.randint(-size, size)
            cv2.line(img, (x, y), (x2, y2), color, thickness)
    
    return img

# 展示图像的函数
def show_image(img):
    """在Jupyter Notebook中显示图像"""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()



# In[6]:


# 自定义参数
img3 = generate_random_shapes(
    width=800, 
    height=600, 
    background_color=(240, 240, 240),
    shape_types=["ellipse"],
    num_shapes=5,
    min_size=30,
    max_size=150
)
show_image(img3)

