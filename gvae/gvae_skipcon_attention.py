import os
import torch
import torch_geometric
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Batch
from torch.nn import Linear
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import distance_matrix
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from collections import defaultdict
from Bio.PDB import *
from torch_geometric.nn import SAGEConv, SAGPooling, GATConv


# Pooling code
class PoolGraph(torch.nn.Module):
    def __init__(self, num_node_features):
        super().__init__()
        self.pool1 = SAGPooling(num_node_features)

    def forward(self, x, edge_index, batch):
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, batch=batch)
        return torch_geometric.data.Data(x=x, edge_index=edge_index, batch=batch)


# Unpooling code
class PoolUnpoolGraph(torch.nn.Module):
    def __init__(self, num_node_features):
        super().__init__()
        self.pool1 = SAGPooling(num_node_features)

    def forward(self, x, edge_index, batch):
        x_cg, edge_index_cg, _, _, perm, _ = self.pool1(x, edge_index, batch=batch)

        # Create a new batch assignment for the unpooled nodes
        new_batch = batch[perm]

        # Use the 'perm' tensor directly for unpooling. This perm tensor gives
        # the ordering of nodes after pooling, so use it to obtain the mapping
        x_unpooled = x[perm]

        # Unpool the edge indices using the perm tensor
        edge_index_unpooled = torch.stack([perm[edge_index_cg[0]],
                                           perm[edge_index_cg[1]]], dim=0)

        return x_unpooled, x_cg, edge_index_unpooled, new_batch


class VAE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_samples=10, heads=1):
        super(VAE, self).__init__()
        self.dropout_prob = 0.49
        self.n_samples = n_samples

        # First-level Encoder
        self.encoder1 = torch.nn.Sequential(
            SAGEConv(in_channels, hidden_channels),
            torch.nn.Dropout(self.dropout_prob),
            torch.nn.BatchNorm1d(hidden_channels)
        )

        # Second-level Encoder with GAT
        self.encoder2 = torch.nn.Sequential(
            GATConv(hidden_channels, hidden_channels, heads=heads, dropout=self.dropout_prob),
            torch.nn.BatchNorm1d(hidden_channels * heads)  # Adjust for multi-head attention
        )

        # Third-level Encoder
        self.encoder3 = torch.nn.Sequential(
            SAGEConv(hidden_channels, hidden_channels),
            torch.nn.Dropout(self.dropout_prob),
            torch.nn.BatchNorm1d(hidden_channels),
            SAGEConv(hidden_channels, 2 * out_channels)  # Outputting mean and log_std
        )

        # First-level Decoder with skip connection
        self.decoder1 = torch.nn.Sequential(
            Linear(out_channels, hidden_channels),
            torch.nn.Dropout(self.dropout_prob),
            torch.nn.BatchNorm1d(hidden_channels)
        )

        # Second-level Decoder with skip connection
        self.decoder2 = torch.nn.Sequential(
            Linear(hidden_channels * 2, hidden_channels),  # Adjust for concatenated skip connection
            torch.nn.Dropout(self.dropout_prob),
            torch.nn.BatchNorm1d(hidden_channels)
        )

        # Third-level Decoder
        self.decoder3 = torch.nn.Sequential(
            Linear(hidden_channels * 2, hidden_channels),  # Adjust for concatenated skip connection
            torch.nn.Dropout(self.dropout_prob),
            torch.nn.BatchNorm1d(hidden_channels),
            Linear(hidden_channels, in_channels)
        )

    def encode(self, x, edge_index):
        # First-level encoding
        x1 = F.relu(self.encoder1[0](x, edge_index))
        x1 = self.encoder1[1](x1)
        x1 = self.encoder1[2](x1)

        # Second-level encoding
        x2 = F.relu(self.encoder2[0](x1, edge_index))
        x2 = self.encoder2[1](x2)

        # Third-level encoding
        x3 = F.relu(self.encoder3[0](x2, edge_index))
        x3 = self.encoder3[1](x3)
        x3 = self.encoder3[2](x3)
        mean, log_std = self.encoder3[3](x3, edge_index).chunk(2, dim=-1)

        return x1, x2, mean, log_std

    def decode(self, z, skip1, skip2):
        # First-level decoding
        h1 = self.decoder1(z)

        # Concatenate the first skip connection
        h1 = torch.cat([h1, skip1], dim=1)

        # Second-level decoding
        h2 = self.decoder2(h1)

        # Concatenate the second skip connection
        h2 = torch.cat([h2, skip2], dim=1)

        # Third-level decoding
        return self.decoder3(h2)

    def reparameterize(self, mean, log_std):
        std = log_std.exp()
        z_samples = [mean + std * torch.randn_like(std) for _ in range(self.n_samples)]
        z = torch.mean(torch.stack(z_samples), dim=0)
        return z

    def forward(self, x, edge_index, batch):
        skip1, skip2, mean, log_std = self.encode(x, edge_index)
        z = self.reparameterize(mean, log_std)
        reconstruction = self.decode(z, skip1, skip2)
        return reconstruction, mean, log_std

    def recon_loss(self, x, reconstruction):
        return F.mse_loss(reconstruction, x, reduction='mean')

    def kl_divergence(self, mean, log_std):
        std = log_std.exp()
        kl_loss = -0.5 * torch.sum(1 + 2 * log_std - mean.pow(2) - std.pow(2), dim=-1)
        return kl_loss.mean()


def train_vae(model, loader, optimizer, lambda_kl=1.0):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()

        # Forward pass
        reconstruction, _, _ = model(data.x, data.edge_index, data.batch)

        # Get the mean and log_std from the encoding
        # Updated call
        skip1, skip2, mean, log_std = model.encode(data.x, data.edge_index)

        # Compute reconstruction loss
        recon_loss = F.mse_loss(reconstruction, data.x, reduction='mean')

        # Compute KL divergence
        kl_loss = model.kl_divergence(mean, log_std)

        # Combined loss
        loss = recon_loss + lambda_kl * kl_loss

        # Backpropagate
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def validate_vae(model, loader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data in loader:
            reconstruction, _, _ = model(data.x, data.edge_index, data.batch)
            # Updated call
            skip1, skip2, mean, log_std = model.encode(data.x, data.edge_index)
            recon_loss = F.mse_loss(reconstruction, data.x, reduction='mean')
            kl_loss = model.kl_divergence(mean, log_std)
            loss = recon_loss + kl_loss
            total_loss += loss.item()

    return total_loss / len(loader)


def compute_rmsd(coords1, coords2):
    """Compute RMSD between two sets of coordinates"""
    assert coords1.size() == coords2.size(), "Coordinate tensors must have the same shape"

    diff = coords1 - coords2
    squared_diff = diff * diff
    mean_squared_diff = squared_diff.mean(dim=-1)
    rmsd = torch.sqrt(mean_squared_diff).mean()
    return rmsd


def test_vae_rmsd(model, loader):
    model.eval()
    total_rmsd = 0

    with torch.no_grad():
        for data in loader:
            reconstruction, _, _ = model(data.x, data.edge_index, data.batch)

            # Assuming the coordinates are stored in the first 3 columns of data.x
            original_coords = data.x[:, :3]
            reconstructed_coords = reconstruction[:, :3]

            rmsd = compute_rmsd(original_coords, reconstructed_coords)
            total_rmsd += rmsd.item()

    return total_rmsd / len(loader)


class CustomGraphDataset(Dataset):
    def __init__(self, data_folder, numerical_indices):
        self.data_file_list = [os.path.join(data_folder, filename) for filename in os.listdir(data_folder) if
                               filename.endswith('.pt')]
        self.numerical_indices = numerical_indices

        # Compute mean and std on the training set
        self.mean, self.std = self._compute_mean_std()

    def __len__(self):
        return len(self.data_file_list)

    def __getitem__(self, index):
        graph_data = torch.load(self.data_file_list[index])

        # Normalize only numerical features
        graph_data.x[:, self.numerical_indices] = (graph_data.x[:, self.numerical_indices] - self.mean) / self.std

        return graph_data

    def _compute_mean_std(self):
        all_data = [torch.load(file) for file in self.data_file_list]
        all_features = torch.cat([data.x[:, self.numerical_indices] for data in all_data], dim=0)
        return torch.mean(all_features, dim=0), torch.std(all_features, dim=0)


def collate_fn(batch):
    return Batch.from_data_list(batch)


if __name__ == '__main__':
    data_folder = "C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//graphs_energy_no_y"
    numerical_indicies = [102, 103, 104]
    dataset = CustomGraphDataset(data_folder, numerical_indicies)

    # Define the split size
    train_size = int(0.8 * len(dataset))  # Use 80% of the data for training
    valid_size = len(dataset) - train_size  # Use the rest for validation

    # Perform the split
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    batch_size = 1  # Adjust the batch size if memory crashes

    # Create data loaders for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0,
                              collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0,
                              collate_fn=collate_fn)

    # Define the model
    in_channels = train_dataset[0].num_node_features
    hidden_channels = 256
    out_channels = 28
    model = VAE(in_channels, hidden_channels, out_channels)
    num_epochs = 20

    # Set up the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_vae(model, train_loader, optimizer)
        valid_loss = validate_vae(model, valid_loader)

        print(f"Epoch: {epoch}, Train Loss: {train_loss}, Validation Loss: {valid_loss}")

    # Save the model parameters
    torch.save(model.state_dict(), '/autoencoder/new_vae_2.pth')

    # testing code:

    # Step 1: Load the model's saved state
    model_path = '/autoencoder/new_vae_2.pth'
    model = VAE(in_channels, hidden_channels, out_channels)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Step 2: Initialize the DataLoader for the test set
    test_dataset = CustomGraphDataset(data_folder, numerical_indicies)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0,
                             collate_fn=collate_fn)

    # Step 3: Evaluate the model on the test set using RMSD
    test_rmsd = test_vae_rmsd(model, test_loader)
    print(f"Test RMSD: {test_rmsd}")

    # Define the model
    model = VAE(in_channels, hidden_channels, out_channels)

    # Load the model parameters
    model.load_state_dict(torch.load('/autoencoder/new_vae_2.pth'))

    new_graph = "C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//chi_energy//chig_1.pdb.pt"
    original_pdb = 'C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//input_chig_with_forces//chig_1.pdb'

    # Compute embeddings for all nodes in the new graph
    model.eval()
    with torch.no_grad():
        new_graph_data = torch.load(new_graph)
        new_graph_data.x[:, numerical_indicies] = (new_graph_data.x[:, numerical_indicies] - dataset.mean) / dataset.std
        x1, x2, mean, log_std = model.encode(new_graph_data.x, new_graph_data.edge_index)
        new_embeddings = model.reparameterize(mean, log_std).cpu().numpy()

    # Extract atom information from the PDB file
    parser = PDBParser()
    structure = parser.get_structure("original", original_pdb)
    atom_info = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom_info[atom.serial_number] = {
                        'coord': atom.coord.tolist(),
                        'residue_name': residue.resname,
                        'atom_name': atom.name
                    }

    # Compute the distance matrix from spatial coordinates
    all_coords = [atom_info[atom]['coord'] for atom in atom_info]
    dist_matrix = distance_matrix(all_coords, all_coords)

    # Optional: Reduce the dimensionality of the distance matrix using PCA
    pca = PCA(n_components=10)  # Choose a reasonable number of components
    reduced_dist_matrix = pca.fit_transform(dist_matrix)

    # Combine the embeddings with the reduced distance matrix
    combined_embeddings = np.hstack((new_embeddings, reduced_dist_matrix))

    # Use Agglomerative Clustering
    hierarchical = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5)  # You can adjust parameters
    hierarchical_clusters = hierarchical.fit_predict(combined_embeddings)

    linked = linkage(combined_embeddings, 'ward')  # 'ward' is one method, you can explore 'single', 'complete', etc.

    # Plot the dendrogram to visualize the structure
    plt.figure(figsize=(19, 10))
    dendrogram(linked)
    plt.title('Dendrogram')
    plt.ylabel('Euclidean distances')
    plt.show()

    # Decide on a height to cut the dendrogram
    cut_height = float(input("Enter the height at which to cut the dendrogram to form clusters: "))

    # Perform hierarchical Clustering
    cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=cut_height, linkage='ward')
    labels = cluster.fit_predict(combined_embeddings)


def list_atoms_per_cluster(labels, atom_info, output_file="clusters_info_16.txt"):
    # Create a dictionary to store atom info for each cluster
    clusters_dict = defaultdict(list)

    # Assign each atom to its respective cluster
    for atom_serial, label in enumerate(labels, start=1):  # assuming atom serial numbers start from 1
        clusters_dict[label].append(atom_info[atom_serial])

    # Save the atoms for each cluster into a file
    with open(output_file, "w") as f:
        for cluster_label, atoms in clusters_dict.items():
            f.write(f"Cluster {cluster_label}:\n")
            for atom in atoms:
                atom_name = atom['atom_name']
                residue_name = atom['residue_name']
                coord = atom['coord']
                f.write(f"    Atom: {atom_name} (Residue: {residue_name}, Coordinates: {coord})\n")
            f.write("\n")  # Separate clusters by a newline

    print(f"Clusters info saved to {output_file}.")

    return clusters_dict


# Call the function
clusters_dict = list_atoms_per_cluster(labels, atom_info)


def generate_colored_pdb(labels, original_pdb_path, output_pdb_path):
    # Load the structure from the original PDB file
    parser = PDBParser()
    structure = parser.get_structure("original", original_pdb_path)

    # Modify the B-factor of each atom based on its cluster label
    for atom, label in zip(structure.get_atoms(), labels):
        atom.set_bfactor(label)

    # Save the modified structure to a new PDB file
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb_path)

    print(f"Colored PDB file saved to {output_pdb_path}.")


# Specify the path for the new PDB file
output_pdb_path = "../autoencoder/colored_clusters_16.pdb"
original_pdb = 'C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//input_chig_with_forces//chig_1.pdb'

# Generate the colored PDB file
generate_colored_pdb(labels, original_pdb, output_pdb_path)

