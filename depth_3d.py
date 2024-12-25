import utils

image_path = "./resource/struct1.webp"
depth_map = utils.estimate_depth(image_path)
im = utils.create_intrinsic_matrix(depth_map, 1.0)
point_cloud = utils.generate_point_cloud(depth_map, im)

import numpy as np

point_cloud = utils.preprocess_point_cloud(point_cloud, method="normalize")
pcd = utils.visualize_point_cloud_with_texture(point_cloud, image_path)