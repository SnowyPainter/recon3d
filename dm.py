import utils
import matplotlib.pyplot as plt

image_path = "./resource/Images1/1.Bmp"
depth_map = utils.estimate_depth(image_path)

processed_depth_map = utils.process_depth_map(depth_map)
# Visualization
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Depth Map")
plt.imshow(depth_map, cmap='viridis')  # Viridis 컬러맵 사용
plt.colorbar()
plt.subplot(1, 2, 2)
plt.title("Processed Depth Map")
plt.imshow(processed_depth_map, cmap='viridis')  # Viridis 컬러맵 사용
plt.colorbar()
plt.show()