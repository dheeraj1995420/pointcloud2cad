# import numpy as np
# import open3d as o3d
# import pandas as pd
# from plyfile import PlyData
#
# pcd = o3d.io.read_point_cloud("full_buro_1m.pcd")
# pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals
# pcd.estimate_normals()
# #o3d.visualization.draw_geometries([pcd], point_show_normal=True)
# print(np.asarray(pcd.points).shape); print(np.asarray(pcd.normals).shape)
# radii = [0.005, 0.01, 0.02, 0.04]
# rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
# # o3d.visualization.draw_geometries([pcd, rec_mesh])
# o3d.io.write_triangle_mesh("p_mesh_c.ply", rec_mesh)w
# plydata = PlyData.read('p_mesh_c.ply')
#

import numpy as np
import open3d as o3d

pcd = o3d.io.read_point_cloud("full_buro_1m.pcd")
pcd_np = np.asarray(pcd.points)
print('run Poisson surface reconstruction')

print("Recompute the normal of the downsampled point cloud")
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.0001, max_nn=5))
# o3d.geometry.estimate_normals(pcd, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
# o3d.visualization.draw_geometries([pcd])

# print("Print a normal vector of the 0th point")
# print(pcd.normals[0])
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10)
print(type(mesh))
print(mesh)
o3d.visualization.draw_geometries([mesh],
                                  zoom=0.664,
                                  front=[-0.4761, -0.4698, -0.7434],
                                  lookat=[1.8900, 3.2596, 0.9284],
                                  up=[0.2304, 1-0.8825, 0.4101])

o3d.io.write_triangle_mesh("p_mesh_c.ply", mesh)
