import utils
import numpy as np
import open3d as o3d

image_path = "./resource/Images1/1.Bmp"
depth_map = utils.estimate_depth(image_path)
cam = utils.create_intrinsic_matrix(depth_map, 1.0)
points = utils.generate_point_cloud(depth_map, cam)
points = utils.preprocess_point_cloud(points, method="normalize")
points = utils.flatten_to_two_levels(points, threshold_ratio=0.15)

utils.visualize_point_cloud_with_texture(points, image_path)