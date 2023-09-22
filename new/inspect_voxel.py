import numpy as np

# Replace this with the actual path to your .npy file
file_path = 'C://Users//gemma//PycharmProjects//pythonProject1//new//voxel_data//2E6Y_2.npy'

# Load the .npy file
voxel_data = np.load(file_path)

# Display various properties of the data
print("Data type:", voxel_data.dtype)
print("Shape:", voxel_data.shape)
print("Dimensions:", voxel_data.ndim)
print("Number of elements:", voxel_data.size)
print("Max value:", np.max(voxel_data))
print("Min value:", np.min(voxel_data))
print("Mean value:", np.mean(voxel_data))
print("Standard deviation:", np.std(voxel_data))

# To show some actual data
print("Sample data (first 5x5x5 voxels):\n", voxel_data[:5, :5, :5])
