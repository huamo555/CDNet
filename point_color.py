import open3d as o3d
import numpy as np

# 读取点云
pcd = o3d.io.read_point_cloud("/home/data3/gaoyuming/mutilview_project/graspness_depthguji/view/quan0000.pcd")

# 点云分割
# 这里我们使用半径为0.5的球形区域进行分割
radii = [0.5, 1.0, 2.0]
segmented_pcd = pcd.segment半径_search(radii)

# 假设我们有两个物体，我们可以通过分割结果来获取它们
# 通常，分割结果会返回一个列表，其中包含分割后的点云
# 这里我们简单地取前两个分割结果作为示例
object1 = segmented_pcd[0]
object2 = segmented_pcd[1]

# 为不同物体赋予不同的颜色
colors = {
    "object1": [1, 0, 0],  # 红色
    "object2": [0, 1, 0],  # 绿色
}

# 着色
object1.paint_uniform_color(colors["object1"])
object2.paint_uniform_color(colors["object2"])

cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector([object1,object2].astype(np.float32))
o3d.io.write_point_cloud("/home/data3/gaoyuming/mutilview_project/graspness_depthguji/view/quan0000.pcd", cloud) 