import utils
import numpy as np
import open3d as o3d

image_path = "./resource/Images1/1.Bmp"
save_dir = "./results/Images1"

depth_map = utils.estimate_depth(image_path)

utils.visualize_depth(depth_map, image_path, save_dir=save_dir)

cam = utils.create_intrinsic_matrix(depth_map, 1.0)
points = utils.generate_point_cloud(depth_map, cam)
points = utils.preprocess_point_cloud(points, method="normalize")
points = utils.flatten_to_two_levels(points, threshold_ratio=0.15)

utils.visualize_z_change_scatter(points, save_dir=save_dir)
utils.visualize_point_cloud_with_texture(points, image_path, save_dir=save_dir)