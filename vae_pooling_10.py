import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Batch
from torch_geometric.nn import SAGEConv
from torch.nn import Linear, Dropout, BatchNorm1d
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from Bio.PDB import PDBIO, Chain, Residue, Atom, Model, Structure
import numpy as np
from Bio.PDB import PDBParser
from collections import Counter
from sklearn.decomposition import PCA
from scipy.spatial import distance_matrix
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from collections import defaultdict
from Bio.PDB import *

class VAE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(VAE, self).__init__()
        self.dropout_prob = 0.5  # Adjust the dropout probability if needed

        # Encoder layers
        self.encoder = torch.nn.ModuleList([
            SAGEConv(in_channels, hidden_channels),
            Dropout(self.dropout_prob),
            BatchNorm1d(hidden_channels),
            SAGEConv(hidden_channels, 2 * hidden_channels),
            Dropout(self.dropout_prob),
            BatchNorm1d(2 * hidden_channels),
            SAGEConv(2 * hidden_channels, 2 * out_channels)  # latent space with mean and log_std
        ])

        # Decoder layers
        self.decoder = torch.nn.ModuleList([
            Linear(out_channels, 2 * hidden_channels),
            Dropout(self.dropout_prob),
            BatchNorm1d(2 * hidden_channels),
            Linear(2 * hidden_channels, hidden_channels),
            Dropout(self.dropout_prob),
            BatchNorm1d(hidden_channels),
            Linear(hidden_channels, in_channels)
        ])
        self.out_channels = out_channels

    def encode(self, x, edge_index):
        x = F.relu(self.encoder[0](x, edge_index))
        x = self.encoder[1](x)  # Dropout
        x = self.encoder[2](x)  # BatchNorm
        x = F.relu(self.encoder[3](x, edge_index))
        x = self.encoder[4](x)  # Dropout
        x = self.encoder[5](x)  # BatchNorm
        x = self.encoder[6](x, edge_index)
        mean, log_std = x.chunk(2, dim=-1)  # Split into mean and log_std
        return mean, log_std

    def decode(self, z):
        h = F.relu(self.decoder[0](z))
        h = self.decoder[1](h)  # Dropout
        h = self.decoder[2](h)  # BatchNorm
        h = F.relu(self.decoder[3](h))
        h = self.decoder[4](h)  # Dropout
        h = self.decoder[5](h)  # BatchNorm
        return self.decoder[6](h)

    def reparameterize(self, mean, log_std):
        std = log_std.exp()
        epsilon = torch.randn_like(std)
        z = mean + std * epsilon
        return z

    def forward(self, x, edge_index):
        mean, log_std = self.encode(x, edge_index)
        z = self.reparameterize(mean, log_std)
        return z

    def global_mean_pool(self, z):
        return torch.mean(z, dim=0)

    def recon_loss(self, x, mean, log_std, edge_index):
        z = self.reparameterize(mean, log_std)
        x_recon = self.decode(z)
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        return recon_loss

    def kl_divergence(self, mean, log_std):
        std = log_std.exp()
        kl_loss = -0.5 * torch.sum(1 + 2 * log_std - mean.pow(2) - std.pow(2), dim=-1)
        return kl_loss.mean()


def train_vae(model, loader, optimizer):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        mean, log_std = model.encode(data.x, data.edge_index)
        recon_loss = model.recon_loss(data.x, mean, log_std, data.edge_index)
        kl_loss = model.kl_divergence(mean, log_std)
        loss = recon_loss + kl_loss
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

    batch_size = 4  # Adjust the batch size if memory crashes

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
    #
    # # Training loop
    # for epoch in range(num_epochs):
    #     loss = train_vae(model, train_loader, optimizer)
    #     if epoch % 10 == 0:
    #         print(f"Epoch: {epoch}, Loss: {loss}")
    #
    # # Save the model parameters
    # torch.save(model.state_dict(), 'model_new_4.pth')

    # Define the model
    model = VAE(in_channels, hidden_channels, out_channels)

    # Load the model parameters
    model.load_state_dict(torch.load('C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//model_new_4.pth'))

    new_graph = "C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//chi_graph//chig.pdb.pt"
    original_pdb = 'C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//input_chig//chig.pdb'

    # Compute embeddings for all nodes in the new graph
    model.eval()
    with torch.no_grad():
        new_graph_data = torch.load(new_graph)
        new_graph_data.x[:, numerical_indicies] = (new_graph_data.x[:, numerical_indicies] - dataset.mean) / dataset.std
        mean, log_std = model.encode(new_graph_data.x, new_graph_data.edge_index)
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
    labels = cluster.fit_predict(new_embeddings)


def list_atoms_per_cluster(labels, atom_info, output_file="clusters_info.txt"):
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
output_pdb_path = "colored_clusters.pdb"
original_pdb = 'C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//input_chig//chig.pdb'

# Generate the colored PDB file
generate_colored_pdb(labels, original_pdb, output_pdb_path)