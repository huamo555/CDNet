from PIL import Image
import numpy as np
import os

root = "/data3/gaoyuming/project/datasets/datasets/dataset-data/"
camera = 'realsense'
sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in range(0, 100)]
range_edges = [100, 200, 300, 400, 500, 600, 700, 800, 850, 900, 1000,2000,3000,4000]  # 定义范围的边界
range_counts = {i: 0 for i in range_edges}  # 初始化计数器字典

for xx in sceneIds:
    for img_num in range(256):
        img_path = os.path.join(root, 'depth_clean', xx, camera, str(img_num).zfill(4) + '.png')
        if os.path.exists(img_path):  # 检查文件是否存在
            depth = np.array(Image.open(img_path))
            max_depth = depth.max()
            print(f"Image {img_num}: Max depth = {max_depth}")
            
            # 统计每个范围内深度值出现的次数
            for start, end in zip(range_edges[:-1], range_edges[1:]):  # 使用zip来配对范围边界
                count = np.sum((depth >= start) & (depth < end))
                range_counts[start] += count
        else:
            print(f"File not found: {img_path}")

# 打印每个范围的计数结果
for start in range_edges[:-1]:  # 遍历除了最后一个边界之外的所有边界
    end = range_edges[range_edges.index(start) + 1]  # 获取对应的结束边界
    print(f"Depth range {start}-{end}: {range_counts[start]} times")