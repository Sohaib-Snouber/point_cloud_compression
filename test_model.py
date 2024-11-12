import tensorflow as tf
import numpy as np
import open3d as o3d
from data_loader import load_point_cloud

# Load the trained model
model = tf.keras.models.load_model("chess_pointnet_model.h5")

# Path to the test point cloud file
test_file_path = "data/red_cylinder/more/more/full_point_cloud2.ply"  # Adjust this path if needed

# Function to preprocess the point cloud
def preprocess_point_cloud(file_path, max_points=2048):
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    
    # Check the number of points
    num_points = points.shape[0]
    print(f"Number of points in the point cloud: {num_points}")

    # If the point cloud has more than max_points, randomly sample
    if num_points > max_points:
        indices = np.random.choice(num_points, max_points, replace=False)
        points = points[indices]
    # If the point cloud has fewer than max_points, pad with zeros
    elif num_points < max_points:
        padding = max_points - num_points
        points = np.pad(points, ((0, padding), (0, 0)), mode='constant')

    # Add an extra dimension to match the model input shape
    points = np.expand_dims(points, axis=0)
    return points

# Preprocess the test point cloud
test_pcd_processed = preprocess_point_cloud(test_file_path)

# Run prediction
predictions = model.predict(test_pcd_processed)
predicted_class = np.argmax(predictions)
confidence = np.max(predictions)

# Output the results
print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence:.2f}")

# Interpretation of the result
if predicted_class == 0:
    print("Detected object: Red Cylinder")
else:
    print("Detected object: Unknown")
