import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNET3D(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET3D, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # DOWN Part
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
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
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                # Interpolation logic might be needed here if the shapes don't match
                pass

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


# Directory containing voxel .npy files
input_dir = "C://Users//gemma//PycharmProjects//pythonProject1//new//voxel_data"

# List all .npy files in the directory
all_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]

# Load all the .npy files into a list
all_data = [np.load(os.path.join(input_dir, f)) for f in all_files]

# Convert list to numpy array
all_data_np = np.array(all_data)

# Add a channel dimension, so shape becomes (N, 1, 32, 32, 32)
all_data_np = np.expand_dims(all_data_np, axis=1)

# Split the data into training, validation, and test sets
train_data, test_data = train_test_split(all_data_np, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

print(f"Number of samples in training set: {len(train_data)}")
print(f"Number of samples in validation set: {len(val_data)}")
print(f"Number of samples in test set: {len(test_data)}")


# Convert numpy arrays to PyTorch tensors
train_tensor = torch.Tensor(train_data)
val_tensor = torch.Tensor(val_data)
test_tensor = torch.Tensor(test_data)

# Create TensorDataset
train_dataset = TensorDataset(train_tensor)
val_dataset = TensorDataset(val_tensor)
test_dataset = TensorDataset(test_tensor)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)
test_loader = DataLoader(test_dataset, batch_size=4)

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16
num_epochs = 10
num_workers = 2
image_height = 32
image_width = 32
image_length = 32


def train_fn(loader, model, optimizer, criterion):
    model.train()

    for batch_idx, (data,) in enumerate(loader):
        data = data.to(device)

        # Forward pass
        outputs = model(data)

        # Compute the loss
        loss = criterion(outputs, data.squeeze(1).long())  # Squeezing out the channel dimension

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Batch {batch_idx + 1}/{len(loader)}, Loss: {loss.item()}")


def val_fn(loader, model, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (data,) in enumerate(loader):
            data = data.to(device)

            # Forward pass
            outputs = model(data)

            # Compute the loss
            loss = criterion(outputs, data.squeeze(1).long())
            val_loss += loss.item()

    average_val_loss = val_loss / len(loader)
    print(f"Validation Loss: {average_val_loss}")


def load_new_file(file_path):
    data = np.load(file_path)
    data = np.expand_dims(data, axis=(0, 1))  # Adding batch and channel dimensions
    tensor = torch.Tensor(data)
    return tensor


def inference(model, new_data, device):
    model.eval()
    with torch.no_grad():
        new_data = new_data.to(device)
        outputs = model(new_data)
        _, predicted = torch.max(outputs, 1)
        predicted_np = predicted.cpu().numpy()
    return predicted_np





def main():
    # Initialize the U-Net model
    model = UNET3D(in_channels=1, out_channels=10)  # this will give 10 clusters
    model.to(device)

    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Initialize the loss function
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Training
        train_fn(train_loader, model, optimizer, criterion)

        # Validation
        val_fn(val_loader, model, criterion)

    # Save the model parameters
    torch.save(model.state_dict(), 'model_new_vox.pth')





# Call the main function to start training
if __name__ == "__main__":
    main()