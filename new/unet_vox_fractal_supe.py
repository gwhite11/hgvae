import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import StepLR
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from Bio.PDB import PDBParser, PDBIO


def box_counting(voxel_data, epsilon):
    non_zero_coords = np.argwhere(voxel_data == 1)

    if non_zero_coords.size == 0:
        # Return a default value or raise a specific exception
        return 0

    min_coords = np.min(non_zero_coords, axis=0)
    max_coords = np.max(non_zero_coords, axis=0)

    num_boxes = np.ceil((max_coords - min_coords) / epsilon).astype(int)
    translated_coords = non_zero_coords - min_coords
    box_indices = np.floor(translated_coords / epsilon).astype(int)
    unique_boxes = np.unique(box_indices, axis=0)

    return len(unique_boxes)


def compute_fractal_dimension(voxel_data, epsilons=[2, 4, 6, 8]):
    dimensions = [box_counting(voxel_data, epsilon=eps) for eps in epsilons]
    return np.mean(dimensions)


class FractalPooling3D(nn.Module):
    def __init__(self, kernel_size, initial_alpha=0.5, alpha_decay=0.02, alpha_min=0.1):
        super(FractalPooling3D, self).__init__()
        self.pool = nn.MaxPool3d(kernel_size)
        self.initial_alpha = initial_alpha
        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min
        self.fractal_loss = 0  # Initialize fractal loss

    def forward(self, x, current_epoch):
        alpha = max(self.initial_alpha - current_epoch * self.alpha_decay, self.alpha_min)
        fractal_before_pooling = compute_fractal_dimension(x.detach().cpu().numpy())
        out = self.pool(x)
        fractal_after_pooling = compute_fractal_dimension(out.detach().cpu().numpy())

        fractal_loss = np.abs(fractal_before_pooling - fractal_after_pooling).mean()
        self.fractal_loss = alpha * torch.tensor(fractal_loss, device=x.device, dtype=x.dtype)

        return out  # Only return the tensor

    def get_fractal_loss(self):
        return self.fractal_loss


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc1 = nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate=1):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilation_rate, dilation=dilation_rate, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation_rate, dilation=dilation_rate, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.ca = ChannelAttention(out_channels)

    def forward(self, x):
        x = self.conv(x)
        return x * self.ca(x) + x


class UNET3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, features=[64, 128, 256, 512]):
        super(UNET3D, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = FractalPooling3D(kernel_size=2)

        dilation_rates = [1, 2, 4, 8]

        # DOWN Part
        for feature, dilation_rate in zip(features, dilation_rates):
            self.downs.append(DoubleConv(in_channels, feature, dilation_rate=dilation_rate))
            in_channels = feature

        # UP Part
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose3d(
                    feature * 2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv3d(features[0], 3, kernel_size=1)

    def forward(self, x, current_epoch=0):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x, current_epoch)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x), skip_connections


# Directory containing voxel .npy files
input_dir = "C://Users//gemma//PycharmProjects//pythonProject1//new//voxel_data_with_energy"


# Get voxel and label files
voxel_files = sorted([f for f in os.listdir(input_dir) if "voxel" in f])
label_files = sorted([f for f in os.listdir(input_dir) if "force" in f])

# Load all the .npy files into lists
voxel_data = [np.load(os.path.join(input_dir, f)) for f in voxel_files]
label_data = [np.load(os.path.join(input_dir, f)) for f in label_files]

# Convert lists to numpy arrays and add channel dimension
voxel_data_np = np.array(voxel_data)
label_data_np = np.array(label_data)

# Normalize label data
label_mean = label_data_np.mean()
label_std = label_data_np.std()
label_data_np = (label_data_np - label_mean) / label_std

# Save the normalization parameters for future use (if needed)
# np.save("label_mean.npy", label_mean)
# np.save("label_std.npy", label_std)

voxel_data_np = np.expand_dims(voxel_data_np, axis=1)
label_data_np = np.expand_dims(label_data_np, axis=1)

# Split the data
train_voxels, test_voxels, train_labels, test_labels = train_test_split(voxel_data_np, label_data_np, test_size=0.2, random_state=42)
val_voxels, test_voxels, val_labels, test_labels = train_test_split(test_voxels, test_labels, test_size=0.5, random_state=42)

# Convert numpy arrays to PyTorch tensors
train_voxel_tensor = torch.Tensor(train_voxels)
train_label_tensor = torch.Tensor(train_labels)
val_voxel_tensor = torch.Tensor(val_voxels)
val_label_tensor = torch.Tensor(val_labels)
test_voxel_tensor = torch.Tensor(test_voxels)
test_label_tensor = torch.Tensor(test_labels)

# Create TensorDatasets
train_dataset = TensorDataset(train_voxel_tensor, train_label_tensor)
val_dataset = TensorDataset(val_voxel_tensor, val_label_tensor)
test_dataset = TensorDataset(test_voxel_tensor, test_label_tensor)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)
test_loader = DataLoader(test_dataset, batch_size=4)

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32
num_epochs = 10
num_workers = 2
image_height = 32
image_width = 32
image_length = 32


def train_fn(loader, model, optimizer, criterion, epoch):
    model.train()

    for batch_idx, (data, labels) in enumerate(loader):
        data, labels = data.to(device), labels.to(device)

        # Forward pass
        outputs, _ = model(data, epoch)
        labels = labels.squeeze(1)  # This will remove the singleton dimension at position 1
        labels = labels.permute(0, 4, 2, 3, 1)  # This moves the channel dimension to the correct position

        # Compute the primary loss
        loss = criterion(outputs, labels)

        # Add the fractal loss from the pooling layer
        loss += model.pool.get_fractal_loss()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Batch {batch_idx + 1}/{len(loader)}, Loss: {loss.item()}")


def val_fn(loader, model, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(loader):
            data, labels = data.to(device), labels.to(device)

            outputs, _ = model(data, num_epochs)
            labels = labels.squeeze(1)  # This will remove the singleton dimension at position 1
            labels = labels.permute(0, 4, 2, 3, 1)  # This moves the channel dimension to the correct position

            loss = criterion(outputs, labels)
            val_loss += loss.item()

    average_val_loss = val_loss / len(loader)
    print(f"Validation Loss: {average_val_loss}")


def test_fn(loader, model, criterion):
    model.eval()
    test_loss = 0.0
    total_pred = 0.0
    total_real = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(loader):
            data, labels = data.to(device), labels.to(device)
            outputs, _ = model(data, num_epochs)

            labels = labels.squeeze(1)  # This will remove the singleton dimension at position 1
            labels = labels.permute(0, 4, 2, 3, 1)  # This moves the channel dimension to the correct position

            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Sum the predicted and real values for calculating averages later
            total_pred += torch.sum(outputs).item()
            total_real += torch.sum(labels).item()
            total_samples += labels.numel()

    average_test_loss = test_loss / len(loader)
    rmsd = torch.sqrt(torch.tensor(average_test_loss))

    avg_predicted_value = total_pred / total_samples
    avg_real_value = total_real / total_samples

    print(f"Test Loss (MSE): {average_test_loss}")
    print(f"RMSD: {rmsd.item()}")
    print(f"Average Predicted Value: {avg_predicted_value}")
    print(f"Average Ground Truth Value: {avg_real_value}")


def load_new_file(file_path):
    data = np.load(file_path)
    data = np.expand_dims(data, axis=(0, 1))  # Adding batch and channel dimensions
    tensor = torch.Tensor(data)
    return tensor


def inference(model, new_data, device):
    model.eval()
    with torch.no_grad():
        new_data = new_data.to(device)
        outputs, skip_connections = model(new_data)
    return skip_connections


def main():
    # Initialize the U-Net model
    model = UNET3D(in_channels=1, out_channels=3)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=5,
                       gamma=0.95)  # This will decay the learning rate by a factor of 0.95 every 5 epochs

    # Initialize the loss function
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        train_fn(train_loader, model, optimizer, criterion, epoch)
        val_fn(val_loader, model, criterion)
        scheduler.step()

    # Test the model on the test set after all epochs
    test_fn(test_loader, model, criterion)

    # Save the model parameters
    torch.save(model.state_dict(), 'model_new_vox_2.pth')

    original_pdb_file = "C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//input_chig//chig.pdb"
    new_data_path = "C://Users//gemma//PycharmProjects//pythonProject1//new//voxel_data//chig//chig.npy"
    new_data_tensor = load_new_file(new_data_path)

    def feature_based_clustering(model, new_data_tensor, device, n_clusters=30):
        # Extract the skip connections for a sample
        skip_connections = inference(model, new_data_tensor, device)

        # Use one of the skip connections
        feature_maps = skip_connections[-1]  # using the last skip connection

        # Reshape the feature maps for clustering
        # Treat each voxel as a data point
        B, C, H, W, D = feature_maps.shape
        feature_maps_reshaped = feature_maps.permute(0, 2, 3, 4, 1).reshape(-1, C).cpu().numpy()

        # Apply clustering
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(feature_maps_reshaped)

        # Reshape clusters back to 3D format to get per-voxel labels
        cluster_map = clusters.reshape(H, W, D)

        return cluster_map


    new_data_path = "C://Users//gemma//PycharmProjects//pythonProject1//new//voxel_data//chig//chig.npy"
    new_data_tensor = load_new_file(new_data_path)
    clusters = feature_based_clustering(model, new_data_tensor, device)

    def assign_clusters_to_pdb(original_pdb_file, cluster_map, voxel_size):
        # Parse the PDB file
        parser = PDBParser()
        structure = parser.get_structure('structure', original_pdb_file)

        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        # Determine the voxel this atom belongs to
                        x, y, z = atom.get_coord()
                        voxel_coord = (int(x / voxel_size), int(y / voxel_size), int(z / voxel_size))

                        # Assign the cluster ID to the atom's b-factor
                        if 0 <= voxel_coord[0] < cluster_map.shape[0] and \
                                0 <= voxel_coord[1] < cluster_map.shape[1] and \
                                0 <= voxel_coord[2] < cluster_map.shape[2]:
                            atom.bfactor = cluster_map[voxel_coord]

        # Save the modified PDB
        io = PDBIO()
        io.set_structure(structure)
        io.save('clustered.pdb')

    # Adjust voxel_size to your voxel grid dimensions
    assign_clusters_to_pdb(original_pdb_file, clusters, voxel_size=1.0)

# Call the main function to start training
if __name__ == "__main__":
    main()