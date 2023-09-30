import numpy as np

# Load the .npy file
label_data_path = "C://Users//gemma//PycharmProjects//pythonProject1//new//voxel_data_with_labels//PED00004e001_label.npy"
labels = np.load(label_data_path)

# Print out some statistics
print(f"Shape of the labels: {labels.shape}")
print(f"Min value in the labels: {labels.min()}")
print(f"Max value in the labels: {labels.max()}")
print(f"Unique values in the labels: {np.unique(labels)}")
