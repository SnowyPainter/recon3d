import utils
import os
import glob

image_dir = "./resource/Images2/*.Bmp"
save_base_dir = "./results/Images1"

# 모든 이미지 파일에 대해 반복
for image_path in glob.glob(image_dir):
    # 파일 이름을 기반으로 저장할 디렉터리 생성
    file_name = os.path.basename(image_path).split('.')[0]
    save_dir = os.path.join(save_base_dir, file_name)
    os.makedirs(save_dir, exist_ok=True)

    depth_map = utils.estimate_depth(image_path)

    utils.visualize_depth(depth_map, image_path, save_dir=save_dir)

    cam = utils.create_intrinsic_matrix(depth_map, 1.0)
    points = utils.generate_point_cloud(depth_map, cam)
    points = utils.preprocess_point_cloud(points, method="normalize")
    points = utils.flatten_to_two_levels(points, threshold_ratio=0.15)

    utils.visualize_z_change_scatter(points, save_dir=save_dir)
    utils.visualize_point_cloud_with_texture(points, image_path, save_dir=save_dir)