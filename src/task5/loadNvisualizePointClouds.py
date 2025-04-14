import open3d as o3d

pcd_path = r"C:\Users\ahmed\Downloads\Compressed\apc-vision-toolbox-master\apc-vision-toolbox-master\ros-packages\catkin_ws\src\pose_estimation\src\models\objects\barkely_hide_bones.ply"

pcd = o3d.io.read_point_cloud(pcd_path)
o3d.visualization.draw_geometries([pcd])

