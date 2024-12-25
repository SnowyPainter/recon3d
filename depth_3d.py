import utils

image_path = "./resource/struct1.webp"
depth_map = utils.estimate_depth(image_path)
im = utils.create_intrinsic_matrix(depth_map, 1.0)
point_cloud = utils.generate_point_cloud(depth_map, im)

import numpy as np

filtered_points = point_cloud[
    (point_cloud[:, 2] > 0) &
    (np.abs(point_cloud) < 1e4).all(axis=1)
]
point_cloud = utils.preprocess_point_cloud(filtered_points, method="normalize")
pcd = utils.visualize_point_cloud_with_texture(point_cloud, image_path)