import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from data_loader import load_dataset

# Load the dataset
data, labels = load_dataset("data")

# Preprocess data: pad sequences to the same length
max_points = 2048
data_padded = np.array([np.pad(d, ((0, max_points - len(d)), (0, 0)), mode='constant') for d in data])

# Define the PointNet++ model
def create_pointnet_model(num_classes):
    inputs = layers.Input(shape=(max_points, 3))
    x = layers.Conv1D(64, kernel_size=1, activation='relu')(inputs)
    x = layers.Conv1D(128, kernel_size=1, activation='relu')(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# Create and compile the model
num_classes = 3
model = create_pointnet_model(num_classes)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(data_padded, labels, epochs=20, batch_size=4)
model.save("chess_pointnet_model.h5")
print("Model trained and saved!")
