import numpy as np

# Load the .npy file
label_data_path = "C://Users//gemma//PycharmProjects//pythonProject1//new//voxel_data_with_energy//1E5T_2_force.npy"
labels = np.load(label_data_path)

# Print out some statistics
print(f"Shape of the labels: {labels.shape}")
print(f"Min value in the labels: {labels.min()}")
print(f"Max value in the labels: {labels.max()}")
print(f"Unique values in the labels: {np.unique(labels)}")
