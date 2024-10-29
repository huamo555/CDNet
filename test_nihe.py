import open3d as o3d
import numpy as np
from scipy.optimize import least_squares



import numpy as np
from scipy.optimize import least_squares
import open3d as o3d

import open3d as o3d
import numpy as np
from scipy.optimize import least_squares

""" pcd = o3d.io.read_point_cloud("C:/Users/Administrator/Desktop/bunny.pcd")
points = np.asarray(pcd.points)

random_point_idx = np.random.randint(0, len(pcd.points) - 1)
random_point = np.asarray(pcd.points[random_point_idx])  # 圆心坐标

axis_direction = np.array([1, 2, 1])  # 轴线方向

# 定义圆柱误差函数，只优化半径
def cylinder_radius_residuals(radius, points, center, axis_direction):
    # 计算每个点到圆柱轴线的垂直距离
    v = points - center
    projection = np.dot(v, axis_direction)[:, None] * axis_direction
    perpendicular = v - projection
    distances = np.linalg.norm(perpendicular, axis=1) - radius
    return distances

# 初始化圆柱半径
initial_radius = 0.005

# 使用最小二乘法优化圆柱半径
result = least_squares(cylinder_radius_residuals, initial_radius, args=(points, random_point, axis_direction))
optimized_radius = result.x[0]
print("Initial Radius:", initial_radius)
print("Optimized Radius:", optimized_radius)

# 输出拟合结果
print("Fit Result:", result)

# 可视化拟合的圆柱和点云
cylinder_mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=optimized_radius, height=0.02)
cylinder_mesh.compute_vertex_normals()
cylinder_mesh.translate(random_point)

random_point_pcd = o3d.geometry.PointCloud()
random_point_pcd.points = o3d.utility.Vector3dVector(points) 

o3d.visualization.draw_geometries([random_point_pcd, cylinder_mesh])


 """

