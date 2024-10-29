import pywavefront
import trimesh

# 读取两个 .obj 文件
mesh1 = pywavefront.Wavefront('C:/Users/Administrator/Desktop/000.obj')
mesh2 = pywavefront.Wavefront('C:/Users/Administrator/Desktop/textured.obj')

# 将两个 mesh 合并为一个
merged_mesh = trimesh.util.concatenate(mesh1.vertices, mesh2.vertices)

# 导出为单独的 .obj 文件
trimesh.exchange.export.export_mesh(merged_mesh, 'C:/Users/Administrator/Desktop/merged.obj')



