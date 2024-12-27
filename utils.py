import torch
import urllib.request
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from PIL import Image
import cv2
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from scipy.ndimage import gaussian_gradient_magnitude, gaussian_filter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, time, math

model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.default_transform

def estimate_depth(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map = prediction.cpu().numpy()
    return depth_map

def visualize_depth(depth_map, image_path, save_dir=""):
    if depth_map is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 원본 이미지 표시
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    im = axes[1].imshow(depth_map, cmap='viridis')
    axes[1].set_title("Depth Map")
    axes[1].axis('off')
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Depth', rotation=270, labelpad=15)

    plt.tight_layout()
    if save_dir == "":
        plt.show()
    else:
        plt.savefig(os.path.join(save_dir, "depthmap.png"))
        plt.close()
    
def create_intrinsic_matrix(depth_map, focal_length):
    focal_length = 1.0
    center = (depth_map.shape[1] / 2, depth_map.shape[0] / 2)
    intrinsic_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ])
    return intrinsic_matrix

def preprocess_point_cloud(points, method="normalize"):
    points = points[
        (points[:, 2] > 0) &
        (np.abs(points) < 1e4).all(axis=1)
    ]

    points = np.asarray(points)
    
    if method == "normalize":
        min_vals = points.min(axis=0)
        max_vals = points.max(axis=0)
        return (points - min_vals) / (max_vals - min_vals)
    elif method == "standardize":
        mean_vals = points.mean(axis=0)
        std_vals = points.std(axis=0)
        return (points - mean_vals) / std_vals
    
    else:
        return points

def generate_point_cloud(depth_map, intrinsic_matrix):
    h, w = depth_map.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    z = depth_map.flatten()
    x = (i.flatten() - intrinsic_matrix[0, 2]) * z / intrinsic_matrix[0, 0]
    y = (j.flatten() - intrinsic_matrix[1, 2]) * z / intrinsic_matrix[1, 1]

    points = np.vstack((x, y, z)).T
    return points

def visualize_point_cloud(points):
    points = np.asarray(points)
    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([point_cloud_o3d])

def visualize_point_cloud_with_matplotlib(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.1)
    plt.show()

def create_lines(pcd):
    bbox = pcd.get_axis_aligned_bounding_box()
    min_bound = bbox.get_min_bound()
    max_bound = bbox.get_max_bound()
    center = (min_bound + max_bound) / 2
    corners = [
        min_bound,  # 0
        [max_bound[0], min_bound[1], min_bound[2]],  # 1
        [max_bound[0], max_bound[1], min_bound[2]],  # 2
        [min_bound[0], max_bound[1], min_bound[2]],  # 3
        [min_bound[0], min_bound[1], max_bound[2]],  # 4
        [max_bound[0], min_bound[1], max_bound[2]],  # 5
        max_bound,  # 6
        [min_bound[0], max_bound[1], max_bound[2]]   # 7
    ]

    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    center_lines = [
        [np.array([min_bound[0], center[1], center[2]]), np.array([max_bound[0], center[1], center[2]])],
        [np.array([center[0], min_bound[1], center[2]]), np.array([center[0], max_bound[1], center[2]])],
        [np.array([center[0], center[1], min_bound[2]]), np.array([center[0], center[1], max_bound[2]])],
    ] 
    
    lines = []
    for edge in edges:
        lines.append([corners[edge[0]], corners[edge[1]]])
    lines.extend(center_lines)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array([p for line in lines for p in line]))
    line_set.lines = o3d.utility.Vector2iVector(np.array([[i, i + 1] for i in range(0, len(lines) * 2, 2)], dtype=np.int32))
    return line_set

def visualize_point_cloud_with_texture(points, image_path, save_dir=""):
    image = Image.open(image_path).convert('RGB')
    colors = np.asarray(image) / 255.0
    colors = colors.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    line_set = create_lines(pcd)

    n_points = len(points)
    n_colors = len(colors)
    if n_points > n_colors:
        colors = np.pad(colors, ((0, n_points - n_colors), (0, 0)), mode='edge')
    else:
        colors = colors[:n_points]
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    del colors

    if save_dir == "":
        o3d.visualization.draw_geometries([pcd, line_set],
                                        window_name='Point Cloud Visualization',
                                        width=1024,
                                        height=768,
                                        left=50,
                                        top=50)
    else:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=800, height=600, visible=False)
        time.sleep(0.1)

        vis.add_geometry(pcd)
        vis.add_geometry(line_set)
        ctr = vis.get_view_control()
        ctr.rotate(0, -400)

        vis.update_geometry(pcd)
        vis.update_geometry(line_set)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(1)
        vis.capture_screen_image(os.path.join(save_dir, 'pcd3d.png'), do_render=True)
        
        vis.destroy_window()
    return pcd

def flatten_to_two_levels(points, threshold_ratio=0.1):
    """높이를 두 개의 레벨로 평탄화합니다 (중간 높이 유지, 층 연결)."""
    z_values = points[:, 2]
    z_min = np.min(z_values)
    z_max = np.max(z_values)
    z_range = z_max - z_min

    if z_range == 0:  # 모든 z 값이 동일한 경우 처리
        return points

    # 임계값 계산
    lower_threshold = z_min + z_range * threshold_ratio
    upper_threshold = z_max - z_range * threshold_ratio

    new_points = points.copy()

    # 가장 낮은 층에 속하는 포인트들을 추출
    low_indices = z_values <= lower_threshold
    low_points = points[low_indices]

    # 가장 높은 층에 속하는 포인트들을 추출
    high_indices = z_values >= upper_threshold
    high_points = points[high_indices]

    # 낮은 층이 존재할 경우, 해당 층의 최대 높이로 평탄화
    if len(low_points) > 0:
        max_low_z = np.max(low_points[:, 2])
        new_points[low_indices, 2] = max_low_z

    # 높은 층이 존재할 경우, 해당 층의 최소 높이로 평탄화
    if len(high_points) > 0:
        min_high_z = np.min(high_points[:, 2])
        new_points[high_indices, 2] = min_high_z

    return new_points

def visualize_z_change_scatter(points, save_dir=""):
    x = points[:, 0]
    z = points[:, 2]

    plt.scatter(x, z, s=1)
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('Z Value Change Scatter Plot')
    if save_dir == "":
        plt.show()
    else:
        plt.savefig(os.path.join(save_dir, "pcd_xz_scatter.png"))
        plt.close()