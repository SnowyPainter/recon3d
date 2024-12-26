import utils
import numpy as np
import open3d as o3d

image_path = "./resource/Images1/1.Bmp"
depth_map = utils.estimate_depth(image_path)
cam = utils.create_intrinsic_matrix(depth_map, 1.0)