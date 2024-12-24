import utils

depth_map = utils.estimate_depth("ex1.Bmp")

#utils.visualize_depth_map(depth_map)

point_cloud = utils.generate_point_cloud(depth_map, utils.create_intrinsic_matrix(depth_map, 1.0))

utils.visualize_point_cloud_with_matplotlib(point_cloud)