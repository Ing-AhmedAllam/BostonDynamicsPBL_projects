#!/usr/bin/env python
# coding: utf-8

# In[69]:


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



# In[71]:


# 使用示例
if __name__ == "__main__":
    # 生成只包含圆形和矩形的图像
    img1 = generate_random_shapes(shape_types=["circle", "rectangle"], num_shapes=15)
    show_image(img1)
    


# In[73]:


# 生成所有类型的图形
img2 = generate_random_shapes(num_shapes=20)
show_image(img2)



# In[75]:


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


# In[60]:


def process_shapes_image(img, shapes_info, marker_shape="square", 
                         marker_size=10, marker_color=(0, 0, 255),
                         axesline_color=(255, 0, 0), distance_factor=1.5,
                         target_shape="circle", target_size=15, target_color=(0, 255, 0)):
    """
    处理图像，标记几何图形的中心点和轴线，并在特定方向和距离添加新的几何图形
    
    参数:
    img: 输入图像
    shapes_info: 几何图形信息列表
    marker_shape: 中心点标记形状 ("square", "circle", "cross")
    marker_size: 中心点标记大小
    marker_color: 中心点标记颜色 (B,G,R)
    axesline_color: 轴线颜色 (B,G,R)
    distance_factor: 距离系数，相对于原始图形大小
    target_shape: 目标几何图形形状 ("circle", "square", "triangle")
    target_size: 目标几何图形大小
    target_color: 目标几何图形颜色 (B,G,R)
    
    返回:
    processed_img: 处理后的图像
    """
    # 创建图像副本
    processed_img = img.copy()
    
    for shape_info in shapes_info:
        # 获取中心点
        center = shape_info.get('center')
        if center is None:
            continue
            
        # 绘制中心点标记
        x, y = center
        if marker_shape == "square":
            half_size = marker_size // 2
            cv2.rectangle(processed_img, 
                          (x - half_size, y - half_size), 
                          (x + half_size, y + half_size), 
                          marker_color, -1)
        elif marker_shape == "circle":
            cv2.circle(processed_img, center, marker_size // 2, marker_color, -1)
        elif marker_shape == "cross":
            cv2.line(processed_img, 
                     (x - marker_size//2, y), 
                     (x + marker_size//2, y), 
                     marker_color, 2)
            cv2.line(processed_img, 
                     (x, y - marker_size//2), 
                     (x, y + marker_size//2), 
                     marker_color, 2)
        
        # 绘制轴线
        axes = shape_info.get('axes', [])
        for i, axis in enumerate(axes):
            if len(axis) == 4:  # (x1, y1, x2, y2)
                x1, y1, x2, y2 = axis
                cv2.line(processed_img, (x1, y1), (x2, y2), axesline_color, 2)
                
                # 在轴线方向上的特定距离添加目标几何图形
                if i == 0:  # 只在第一个轴上添加
                    # 计算轴的向量
                    dx = x2 - x1
                    dy = y2 - y1
                    length = math.sqrt(dx*dx + dy*dy)
                    
                    if length > 0:
                        # 计算单位向量并乘以距离系数和原始图形大小
                        size_factor = 0
                        if 'radius' in shape_info:
                            size_factor = shape_info['radius']
                        elif 'width' in shape_info and 'height' in shape_info:
                            size_factor = max(shape_info['width'], shape_info['height'])
                        else:
                            size_factor = marker_size * 2
                            
                        ux = dx / length * size_factor * distance_factor
                        uy = dy / length * size_factor * distance_factor
                        
                        # 计算目标位置
                        target_x = int(x + ux)
                        target_y = int(y + uy)
                        
                        # 绘制目标几何图形
                        if target_shape == "circle":
                            cv2.circle(processed_img, (target_x, target_y), target_size, target_color, -1)
                        elif target_shape == "square":
                            half_size = target_size // 2
                            cv2.rectangle(processed_img, 
                                        (target_x - half_size, target_y - half_size), 
                                        (target_x + half_size, target_y + half_size), 
                                        target_color, -1)
                        elif target_shape == "triangle":
                            # 创建等边三角形
                            triangle_height = int(target_size * math.sqrt(3) / 2)
                            triangle_pts = np.array([
                                [target_x, target_y - triangle_height//2],
                                [target_x - target_size//2, target_y + triangle_height//2],
                                [target_x + target_size//2, target_y + triangle_height//2]
                            ], np.int32)
                            triangle_pts = triangle_pts.reshape((-1, 1, 2))
                            cv2.fillPoly(processed_img, [triangle_pts], target_color)
    
    return processed_img


# In[58]:


# 显示原始图像
show_image(img1)

# 处理图像并添加标记
processed_img = process_shapes_image(
    img1, 
    shapes_info,
    marker_shape="cross",         # 使用十字标记中心点
    marker_color=(0, 0, 255),     # 蓝色标记
    axesline_color=(255, 0, 0),   # 红色轴线
    distance_factor=1.5,          # 在1.5倍原始大小的距离处放置新图形
    target_shape="circle",        # 放置圆形
    target_color=(0, 255, 0)      # 绿色目标图形
)

# 显示处理后的图像
show_image(processed_img)


# In[ ]:




