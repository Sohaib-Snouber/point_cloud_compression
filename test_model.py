import tensorflow as tf
import numpy as np
from data_loader import load_point_cloud

# Load the trained model
model = tf.keras.models.load_model("chess_pointnet_model.h5")

# Load a new point cloud for testing
test_pcd = load_point_cloud("test_piece.ply")
test_pcd_padded = np.pad(test_pcd, ((0, 2048 - len(test_pcd)), (0, 0)), mode='constant')
test_pcd_padded = np.expand_dims(test_pcd_padded, axis=0)

# Run prediction
predictions = model.predict(test_pcd_padded)
predicted_class = np.argmax(predictions)
print(f"Predicted class: {predicted_class}")
