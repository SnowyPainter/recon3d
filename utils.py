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

def generate_point_cloud(depth_map, intrinsic_matrix):
    h, w = depth_map.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    z = depth_map.flatten()
    x = (i.flatten() - intrinsic_matrix[0, 2]) * z / intrinsic_matrix[0, 0]
    y = (j.flatten() - intrinsic_matrix[1, 2]) * z / intrinsic_matrix[1, 1]

    points = np.vstack((x, y, z)).T
    return points

def visualize_point_cloud(points):
    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(points)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(point_cloud_o3d)
    view_control = vis.get_view_control()
    view_control.set_zoom(0.8)  # 줌 조정
    view_control.set_lookat([0, 0, 0])  # 카메라가 바라보는 중심 설정
    
    vis.run()
    vis.destroy_window()

def visualize_point_cloud_with_matplotlib(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.1)
    plt.show()