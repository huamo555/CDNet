from PIL import Image
import numpy as np
import colorsys

# 读取深度图像
depth_path = "/home/data3/gaoyuming/datasets/dataset-data/scenes/scene_0000/realsense/depth/0001.png"
depth = Image.open(depth_path)
depth_array = np.array(depth, dtype=np.uint16)

# 归一化深度值
max_depth = np.max(depth_array)
min_depth = np.min(depth_array)
depth_normalized = (depth_array - min_depth) / (max_depth - min_depth)

# 将归一化的深度值映射到HSV颜色空间
hues = (depth_normalized * 360).astype(int)  # 色调
saturation = 255  # 饱和度
value = 255  # 亮度

# 将HSV转换为RGB
def hsv_to_rgb(h, s, v):
    return colorsys.hsv_to_rgb(h / 360.0, s / 255.0, v / 255.0)

# 创建RGB图像
rgb_image = np.zeros((depth_array.shape[0], depth_array.shape[1], 3), dtype=np.uint8)
for i in range(depth_array.shape[0]):
    for j in range(depth_array.shape[1]):
        r, g, b = hsv_to_rgb(hues[i, j], saturation, value)
        rgb_image[i, j] = [int(r * 255), int(g * 255), int(b * 255)]

# 保存处理后的图像
output_path = "/home/data3/gaoyuming/mutilview_project/graspness_depthguji/view/depth_color.png"
Image.fromarray(rgb_image).save(output_path)
print("已保存")