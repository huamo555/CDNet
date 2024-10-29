import open3d as o3d
import numpy as np


""" # 中间随机裁剪
def crop_and_concatenate(pcd1, pcd2, crop_percentage=0.3):
    # 获取点云数据
    points1 = np.asarray(pcd1.points)
    points2 = np.asarray(pcd2.points)

    # 计算裁剪的点数
    num_points_to_crop1 = int(len(points1) * crop_percentage)
    num_points_to_crop2 = int(len(points2) * crop_percentage)

    # 从每个点云中选择要裁剪的部分
    cropped_points1 = points1[:num_points_to_crop1, :]
    cropped_points2 = points2[:num_points_to_crop2, :]

    # 将裁剪后的点云拼接在一起
    concatenated_points = np.concatenate((cropped_points1, cropped_points2), axis=0)

    # 创建新的点云
    concatenated_pcd = o3d.geometry.PointCloud()
    concatenated_pcd.points = o3d.utility.Vector3dVector(concatenated_points)

    return concatenated_pcd

# 替换为实际的点云文件路径
path_to_pointcloud1 = "C:/Users/Administrator/Desktop/46.pcd"
path_to_pointcloud2 = "C:/Users/Administrator/Desktop/50.pcd"

# 读取点云文件
pointcloud1 = o3d.io.read_point_cloud(path_to_pointcloud1)
pointcloud2 = o3d.io.read_point_cloud(path_to_pointcloud2)

# 裁剪并拼接点云
concatenated_pointcloud = crop_and_concatenate(pointcloud1, pointcloud2, crop_percentage=0.3)

# 保存结果
o3d.io.write_point_cloud("pC:/Users/Administrator/Desktop/concatenated_pointcloud.pcd", concatenated_pointcloud)

o3d.visualization.draw_geometries([concatenated_pointcloud])
 """










# 处理颜色

""" import open3d as o3d

# 读取.pcd文件
pcd = o3d.io.read_point_cloud("C:/Users/Administrator/Desktop/nontextured_simplified.ply")
pcd_without_color = o3d.geometry.PointCloud()
pcd_without_color.points = pcd.points

o3d.io.write_point_cloud("C:/Users/Administrator/Desktop/56.pcd", pcd_without_color)
# 显示点云
o3d.visualization.draw_geometries([pcd_without_color])  """








""" import open3d as o3d
import numpy as np


# 裁剪中心
def adaptive_crop_region(pointcloud, margin=0.2):
   
    points = np.asarray(pointcloud.points)

    # 计算点云的边界框
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)

    # 计算裁剪区域
    range_x = max_bound[0] - min_bound[0]
    range_y = max_bound[1] - min_bound[1]
    range_z = max_bound[2] - min_bound[2]

    margin_x = margin * range_x
    margin_y = margin * range_y
    margin_z = margin * range_z

    crop_min = np.array([min_bound[0] + margin_x, min_bound[1] + margin_y, min_bound[2] + margin_z])
    crop_max = np.array([max_bound[0] - margin_x, max_bound[1] - margin_y, max_bound[2] - margin_z])

    # 创建 AxisAlignedBoundingBox 对象
    crop_region = o3d.geometry.AxisAlignedBoundingBox(min_bound=crop_min, max_bound=crop_max)

    return crop_region 




# 读取点云文件
pcd1 = o3d.io.read_point_cloud("C:/Users/Administrator/Desktop/50.pcd")
pcd2 = o3d.io.read_point_cloud("C:/Users/Administrator/Desktop/53.pcd")
# 生成自适应的裁剪区域
crop_region1 = adaptive_crop_region(pcd1)
crop_region2 = adaptive_crop_region(pcd2)
# 使用生成的裁剪区域进行裁剪
# ...（接下来的操作和前面的代码一样）

# 将裁剪区域放入列表，然后进行可视化
# 使用 crop_region 进行裁剪
cropped_pcd1 = pcd1.crop(crop_region1)
cropped_pcd2 = pcd2.crop(crop_region2)
# 可视化裁剪后的点云
o3d.visualization.draw_geometries([cropped_pcd1])
o3d.visualization.draw_geometries([cropped_pcd2])

# 将裁剪后的点云转换为 NumPy 数组
points1 = np.asarray(cropped_pcd1.points)
points2 = np.asarray(cropped_pcd2.points)

# 连接两个 NumPy 数组
concatenated_points = np.concatenate((points1, points2), axis=0)

# 创建新的 PointCloud 对象并设置点坐标
concatenated_pcd = o3d.geometry.PointCloud()
concatenated_pcd.points = o3d.utility.Vector3dVector(concatenated_points)


# 可视化合并后的点云
o3d.visualization.draw_geometries([concatenated_pcd])




#读取电脑中的 ply 点云文件
source = cropped_pcd1  #source 为需要配准的点云
target = cropped_pcd2  #target 为目标点云

#降采样（均匀下采样 使用 open3d 中的 uniform_down_sample）
source = source.uniform_down_sample(every_k_points=10)
print(source)
target = target.uniform_down_sample(every_k_points=10)
print(target)
#o3d.visualization.draw_geometries([uni_down_pcd])

#为两个点云上上不同的颜色
source.paint_uniform_color([1, 0.706, 0])    #source 为黄色
target.paint_uniform_color([0, 0.651, 0.929])#target 为蓝色

#为两个点云分别进行outlier removal，即离群点去除
processed_source, outlier_index = source.remove_statistical_outlier(nb_neighbors=16,std_ratio=0.5)

processed_target, outlier_index = target.remove_statistical_outlier(nb_neighbors=16,std_ratio=0.5)

threshold = 1.0  #移动范围的阀值
trans_init = np.asarray([[1,0,0,0],   # 4x4 identity matrix，这是一个转换矩阵，
                         [0,1,0,0],   # 象征着没有任何位移，没有任何旋转，我们输入
                         [0,0,1,0],   # 这个矩阵为初始变换
                         [0,0,0,1]])
#运行icp
print("Apply point-to-point ICP")
reg_p2p = o3d.registration.registration_icp(
        processed_source, processed_target, threshold, trans_init,
        o3d.registration.TransformationEstimationPointToPoint())

#将我们的矩阵依照输出的变换矩阵进行变换
print(reg_p2p)    #输出配准的结果准确度等信息
print("Transformation is:")
print(reg_p2p.transformation)  # 打印旋转矩阵
processed_source.transform(reg_p2p.transformation)

o3d.visualization.draw_geometries([processed_source,processed_target]) """

""" # 计算法线
processed_source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
processed_target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

cropped_pcd1.paint_uniform_color([1, 0.706, 0])
cropped_pcd2.paint_uniform_color([0, 0.651, 0.929])
 """



# 阈值裁剪，有一点效果
""" import open3d as o3d
import numpy as np

def align_and_remove_overlap(source, target):

    # 设置ICP参数
    icp_criteria = o3d.registration.ICPConvergenceCriteria(max_iteration=500)
    icp_transform = np.identity(4)

    # 进行点云配准
    result = o3d.registration.registration_icp(source, target, 0.01, icp_transform,
                                               o3d.registration.TransformationEstimationPointToPoint(),
                                               o3d.registration.ICPConvergenceCriteria())

    # 获取配准后的点云
    aligned_source = source.transform(result.transformation)

    # 找到两个点云之间的重叠区域
    distance_threshold = 0.005  # 设置距离阈值，用于判断点是否在重叠区域内
    distances = np.asarray(aligned_source.compute_point_cloud_distance(target))
    overlap_indices = np.where(distances < distance_threshold)[0]

    # 裁剪重叠部分
    removed_overlap = aligned_source.select_by_index(overlap_indices, invert=True)

    return aligned_source, removed_overlap

# 用法示例
source = o3d.io.read_point_cloud("C:/Users/Administrator/Desktop/50.pcd")
target = o3d.io.read_point_cloud("C:/Users/Administrator/Desktop/53.pcd")

# 对两个点云进行配准并裁剪重叠部分
aligned_source, removed_overlap = align_and_remove_overlap(target, source)

ss =source+ removed_overlap
o3d.io.write_point_cloud("C:/Users/Administrator/Desktop/44444.pcd",ss)
# 可视化结果
o3d.visualization.draw_geometries([target, aligned_source, removed_overlap])
 """











""" source_pcd = o3d.io.read_point_cloud("C:/Users/Administrator/Desktop/50.pcd")
target_pcd = o3d.io.read_point_cloud("C:/Users/Administrator/Desktop/53.pcd")

# 计算目标点云的凸包
convex_hull, _ = target_pcd.compute_convex_hull()

# 通过KD树构建查询结构
kdtree = o3d.geometry.KDTreeFlann(convex_hull)

# 判断每个源点是否在凸包内
is_inside = np.array([len(kdtree.search_knn_vector_3d(point, 1)[1]) > 0 for point in source_pcd.points])

# 提取在凸包外的点
outside_points = source_pcd.select_by_index(np.where(is_inside)[0])

# 可视化结果
o3d.visualization.draw_geometries([outside_points]) """






import open3d as o3d
import numpy as np
import random

def rotate_pointcloud(pointcloud):
    # 生成随机旋转角度
    angle = random.uniform(0, 180)  # 随机生成0到360度之间的角度

    # 创建旋转矩阵
    R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, np.radians(angle)))

    # 应用旋转矩阵
    pointcloud.rotate(R, center=(0, 0, 0))

    return pointcloud

# 读取点云文件
pcd1 = o3d.io.read_point_cloud("C:/Users/Administrator/Desktop/50.pcd")

pcd_rotated1 = rotate_pointcloud(pcd1)


pcd2 = o3d.io.read_point_cloud("C:/Users/Administrator/Desktop/53.pcd")
pcd_rotated2 = rotate_pointcloud(pcd2)
""" pcd3 = o3d.io.read_point_cloud("C:/Users/Administrator/Desktop/56.pcd")
pcd_rotated3 = rotate_pointcloud(pcd3) """

#读取电脑中的 ply 点云文件
source = pcd_rotated1  #source 为需要配准的点云
target = pcd_rotated2  #target 为目标点云

#降采样（均匀下采样 使用 open3d 中的 uniform_down_sample）
source = source.uniform_down_sample(every_k_points=1)
print(source)
target = target.uniform_down_sample(every_k_points=1)
print(target)
#o3d.visualization.draw_geometries([uni_down_pcd])

#为两个点云上上不同的颜色
source.paint_uniform_color([1, 0.706, 0])    #source 为黄色
target.paint_uniform_color([0, 0.651, 0.929])#target 为蓝色

#为两个点云分别进行outlier removal，即离群点去除
processed_source, outlier_index = source.remove_statistical_outlier(nb_neighbors=16,std_ratio=0.5)

processed_target, outlier_index = target.remove_statistical_outlier(nb_neighbors=16,std_ratio=0.5)

threshold = 0.1  #移动范围的阀值
trans_init = np.asarray([[1,0,0,0],   # 4x4 identity matrix，这是一个转换矩阵，
                         [0,1,0,0],   # 象征着没有任何位移，没有任何旋转，我们输入
                         [0,0,1,0],   # 这个矩阵为初始变换
                         [0,0,0,1]])
#运行icp
print("Apply point-to-point ICP")
reg_p2p = o3d.registration.registration_icp(
        processed_source, processed_target, threshold, trans_init,
        o3d.registration.TransformationEstimationPointToPoint())

#将我们的矩阵依照输出的变换矩阵进行变换
print(reg_p2p)    #输出配准的结果准确度等信息
print("Transformation is:")
print(reg_p2p.transformation)  # 打印旋转矩阵
processed_source.transform(reg_p2p.transformation)

o3d.visualization.draw_geometries([processed_source, processed_target])

# 可视化配准后的点云
o3d.visualization.draw_geometries([processed_source, processed_target.transform(reg_p2p.transformation)])



# 计算目标点云的凸包
convex_hull, _ = processed_target.compute_convex_hull()

# 将凸包转换为PointCloud对象
convex_hull_pointcloud = o3d.geometry.PointCloud()
convex_hull_pointcloud.points = o3d.utility.Vector3dVector(np.asarray(convex_hull.vertices))

# 计算每个源点到凸包的距离
distances = np.asarray(processed_source.compute_point_cloud_distance(convex_hull_pointcloud))

# 定义一个阈值，判断是否在凸包内
threshold = 0.0001  # 调整阈值以适应你的数据

# 根据距离判断是否在凸包内
is_inside = distances > threshold

# 提取在凸包外的点
outside_points = processed_source.select_by_index(np.where(is_inside)[0])

# 可视化结果
o3d.visualization.draw_geometries([outside_points])









concatenated_points_j = processed_source + processed_target
# concatenated_points_j = processed_source + processed_target

concatenated_points_j_without_color = o3d.geometry.PointCloud()
concatenated_points_j_without_color.points =concatenated_points_j.points

o3d.visualization.draw_geometries([concatenated_points_j_without_color])

o3d.io.write_point_cloud("C:/Users/Administrator/Desktop/oo.pcd", concatenated_points_j_without_color)

