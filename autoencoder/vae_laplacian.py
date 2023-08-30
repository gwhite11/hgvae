import os
import torch
import torch_geometric
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Batch
from torch.nn import Linear, Dropout, BatchNorm1d
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import distance_matrix
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from collections import defaultdict
from Bio.PDB import *
from torch_geometric.nn import SAGEConv, SAGPooling
import pytorch_lightning as pl



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def normalized_laplacian(edge_index, num_nodes=None):
    if edge_index.shape[0] != 2:
        raise ValueError(f"Expected edge_index to have shape (2, E), but got {edge_index.shape}.")

    # Ensure that num_nodes is provided and is greater than the maximum node index in edge_index
    if num_nodes is None:
        raise ValueError("num_nodes must be provided.")
    max_index = edge_index.max()
    if max_index >= num_nodes:
        raise ValueError(f"Invalid edge_index, max index is {max_index}, but num_nodes is {num_nodes}.")

    deg = compute_degree(edge_index, num_nodes=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    # Convert edge_index to a dense adjacency matrix
    adj = torch_geometric.utils.to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
    identity = torch.eye(num_nodes, device=adj.device)

    # Compute the normalized Laplacian
    L = identity - torch.mm(torch.mm(deg_inv_sqrt.diag(), adj), deg_inv_sqrt.diag())
    print("Shape of edge_index:", edge_index.shape)
    return L


def compute_degree(edge_index, num_nodes=None):
    row, col = edge_index[0], edge_index[1]
    out = torch.zeros((num_nodes), dtype=torch.float, device=edge_index.device)
    return out.scatter_add_(dim=0, index=row, src=torch.ones_like(row).float())


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

        # Update the number of nodes after unpooling
        num_nodes_unpooled = x_cg.size(0)

        # Return the updated num_nodes_unpooled
        return x_cg, edge_index_cg, num_nodes_unpooled


class VAE(pl.LightningModule):
    def __init__(self, in_channels, hidden_channels, out_channels, n_samples=10):
        super().__init__()
        self.dropout_prob = 0.5
        self.n_samples = n_samples

        # First-level Encoder
        self.encoder1 = torch.nn.ModuleList([
            SAGEConv(in_channels, hidden_channels),
            Dropout(self.dropout_prob),
            BatchNorm1d(hidden_channels)
        ])

        # Second-level Encoder
        self.encoder2 = torch.nn.ModuleList([
            SAGEConv(hidden_channels, 2 * hidden_channels),
            Dropout(self.dropout_prob),
            BatchNorm1d(2 * hidden_channels),
            SAGEConv(2 * hidden_channels, 2 * out_channels)
        ])

        # First-level Decoder
        self.decoder1 = torch.nn.ModuleList([
            Linear(out_channels, hidden_channels),
            Dropout(self.dropout_prob),
            BatchNorm1d(hidden_channels),
        ])

        # Second-level Decoder
        self.decoder2 = torch.nn.ModuleList([
            Linear(hidden_channels, in_channels)
        ])

        # Pooling
        self.pool_graph = PoolGraph(hidden_channels)
        self.pool_unpool_graph = PoolUnpoolGraph(hidden_channels)

        self.out_channels = out_channels

    def encode(self, x, edge_index):
        # First-level encoding
        x1 = F.relu(self.encoder1[0](x, edge_index))
        x1 = self.encoder1[1](x1)
        x1 = self.encoder1[2](x1)

        # Second-level encoding
        x2 = F.relu(self.encoder2[0](x1, edge_index))
        x2 = self.encoder2[1](x2)
        x2 = self.encoder2[2](x2)
        mean, log_std = self.encoder2[3](x2, edge_index).chunk(2, dim=-1)
        return x1, mean, log_std

    def decode(self, z, x1):
        # First-level decoding
        h = F.relu(self.decoder1[0](z))
        h = self.decoder1[1](h)
        h = self.decoder1[2](h)

        # Add coarse information from x1
        h = h + x1

        # Second-level decoding
        return self.decoder2[0](h)

    def reparameterize(self, mean, log_std):
        std = log_std.exp()
        z_samples = [mean + std * torch.randn_like(std) for _ in range(self.n_samples)]
        z = torch.mean(torch.stack(z_samples), dim=0)
        return z

    def forward(self, x, edge_index, batch):
        x1, mean, log_std = self.encode(x, edge_index)
        z = self.reparameterize(mean, log_std)
        x_cg, _, num_nodes_unpooled = self.pool_unpool_graph(x1, edge_index, batch)
        reconstruction = self.decode(z, x1)
        return reconstruction, x_cg, num_nodes_unpooled

    def recon_loss(self, x, mean, log_std, edge_index, x1):
        z = self.reparameterize(mean, log_std)
        x_recon = self.decode(z, x1)
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        return recon_loss

    def kl_divergence(self, mean, log_std):
        std = log_std.exp()
        kl_loss = -0.5 * torch.sum(1 + 2 * log_std - mean.pow(2) - std.pow(2), dim=-1)
        return kl_loss.mean()

    def l_frequency(self, edge_index_orig, edge_index_pooled, num_nodes_orig, num_nodes_pooled):
        # Ensure the number of nodes is provided
        if num_nodes_orig is None or num_nodes_pooled is None:
            raise ValueError("The number of nodes for both original and pooled graphs must be provided.")

        # Compute the graph Laplacian for the original and pooled graphs
        L_orig = normalized_laplacian(edge_index_orig, num_nodes=num_nodes_orig)
        L_pooled = normalized_laplacian(edge_index_pooled, num_nodes=num_nodes_pooled)

        # Compute eigenvalues (only a few largest ones for computational efficiency)
        orig_eigvals, _ = torch.linalg.eigh(L_orig)
        pooled_eigvals, _ = torch.linalg.eigh(L_pooled)
        orig_eigvals = orig_eigvals[:1]
        pooled_eigvals = pooled_eigvals[:1]

        # Normalize the eigenvalues
        orig_eigvals = (orig_eigvals - torch.min(orig_eigvals)) / (torch.max(orig_eigvals) - torch.min(orig_eigvals))
        pooled_eigvals = (pooled_eigvals - torch.min(pooled_eigvals)) / (
                    torch.max(pooled_eigvals) - torch.min(pooled_eigvals))

        # Compare the spectral content
        return F.mse_loss(orig_eigvals, pooled_eigvals)


def train_vae(model, loader, optimizer):
    model.to(device)  # Move the model to GPU

    total_loss = 0
    for data in loader:
        data = data.to(device)  # Move data to GPU

        optimizer.zero_grad()

        reconstruction, x_cg, num_nodes_unpooled = model(data.x, data.edge_index, data.batch)

        x1, mean, log_std = model.encode(data.x, data.edge_index)  # Encoding to get the pooled representation (x1)

        recon_loss = F.mse_loss(reconstruction, data.x, reduction='mean')
        kl_loss = model.kl_divergence(mean, log_std)

        # Calculate edge_index_cg
        _, edge_index_cg, _ = model.pool_unpool_graph(x1, data.edge_index, data.batch)

        # Pass edge_index_cg to l_frequency function
        l_frequency_loss = model.l_frequency(data.edge_index, edge_index_cg, data.x.size(0), num_nodes_unpooled)
        print(data.edge_index.shape, edge_index_cg.shape)
        loss = recon_loss + kl_loss + l_frequency_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


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
    data_folder = "C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//graphs_2"
    numerical_indicies = [0, 1, 2, 3, 4, 5]
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
    hidden_channels = 128
    out_channels = 28
    model = VAE(in_channels, hidden_channels, out_channels)
    num_epochs = 100

    # Set up the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        loss = train_vae(model, train_loader, optimizer)
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {loss}")

    # Save the model parameters
    torch.save(model.state_dict(), 'model_new_7.pth')

    # Define the model
    model = VAE(in_channels, hidden_channels, out_channels)

    # Load the model parameters
    model.load_state_dict(torch.load('C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//model_new_7.pth'))

    new_graph = "C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//chi_graph//chig.pdb.pt"
    original_pdb = 'C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//input_chig//chig.pdb'

    # Compute embeddings for all nodes in the new graph
    model.eval()
    with torch.no_grad():
        new_graph_data = torch.load(new_graph)
        new_graph_data.x[:, numerical_indicies] = (new_graph_data.x[:, numerical_indicies] - dataset.mean) / dataset.std
        x1, mean, log_std = model.encode(new_graph_data.x, new_graph_data.edge_index)
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


def list_atoms_per_cluster(labels, atom_info, output_file="clusters_info_6.txt"):
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
output_pdb_path = "colored_clusters_6.pdb"
original_pdb = 'C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//input_chig//chig.pdb'

# Generate the colored PDB file
generate_colored_pdb(labels, original_pdb, output_pdb_path)