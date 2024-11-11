import open3d as o3d
import numpy as np
import os

def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return np.asarray(pcd.points)

def load_dataset(data_dir):
    data = []
    labels = []
    label_map = {"red_cylinder": 0}

    for label_name, label in label_map.items():
        folder_path = os.path.join(data_dir, label_name)
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.ply'):
                file_path = os.path.join(folder_path, file_name)
                points = load_point_cloud(file_path)
                data.append(points)
                labels.append(label)
    
    return np.array(data), np.array(labels)

# Example usage
data, labels = load_dataset("data")
print(f"Loaded {len(data)} point clouds.")
