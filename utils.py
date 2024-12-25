import torch
import urllib.request
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from PIL import Image
import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

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

def visualize_depth_map(depth_map):
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_normalized = np.uint8(depth_map_normalized)
    depth_map_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_MAGMA)
    cv2.imshow("Depth Map", depth_map_colored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def create_intrinsic_matrix(depth_map, focal_length):
    focal_length = 1.0  # 예제 값
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
        raise ValueError(f"Unknown method: {method}")

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

def visualize_point_cloud_with_texture(points, image_path):
    image = Image.open(image_path).convert('RGB')
    colors = np.asarray(image) / 255.0
    colors = colors.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    n_points = len(points)
    n_colors = len(colors)
    if n_points > n_colors:
        colors = np.pad(colors, ((0, n_points - n_colors), (0, 0)), mode='edge')
    else:
        colors = colors[:n_points]
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    del colors
    o3d.visualization.draw_geometries([pcd],
                                    window_name='Point Cloud Visualization',
                                    width=1024,
                                    height=768,
                                    left=50,
                                    top=50)
    
    return pcd