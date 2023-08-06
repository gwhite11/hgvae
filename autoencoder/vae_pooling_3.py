import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Batch
from torch_geometric.nn import SAGEConv
from torch.nn import Linear
from scipy.spatial.distance import cdist
import math
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from Bio.PDB import PDBIO, Chain, Residue, Atom, Model, Structure
import numpy as np
from Bio.PDB import PDBParser
from collections import Counter


class VAE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(VAE, self).__init__()
        self.encoder = torch.nn.ModuleList([
            SAGEConv(in_channels, hidden_channels),
            SAGEConv(hidden_channels, 2 * out_channels)  # The latent space has twice the size for mean and log_std
        ])
        self.decoder = torch.nn.ModuleList([
            Linear(out_channels, hidden_channels),
            Linear(hidden_channels, in_channels)
        ])
        self.out_channels = out_channels

    def encode(self, x, edge_index):
        x = F.relu(self.encoder[0](x, edge_index))
        x = self.encoder[1](x, edge_index)
        mean, log_std = x.chunk(2, dim=-1)  # Split into mean and log_std
        return mean, log_std

    def reparameterize(self, mean, log_std):
        std = log_std.exp()
        epsilon = torch.randn_like(std)
        z = mean + std * epsilon
        return z

    def decode(self, z):
        h = F.relu(self.decoder[0](z))
        return self.decoder[1](h)

    def forward(self, x, edge_index):
        mean, log_std = self.encode(x, edge_index)
        z = self.reparameterize(mean, log_std)

        # Apply global mean pooling on the latent space
        z = self.global_mean_pool(z)

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


def visualize_original_and_reconstructed(original_data, reconstructed_data):
    G_original = to_networkx(original_data)
    G_reconstructed = to_networkx(reconstructed_data)

    if original_data.is_directed():
        G_original = G_original.to_directed()

    if reconstructed_data.is_directed():
        G_reconstructed = G_reconstructed.to_directed()

    plt.figure(figsize=(12, 6))

    # Visualize the original graph on the left
    plt.subplot(1, 2, 1)
    pos_original = nx.spring_layout(G_original, seed=42)  # You can change the layout algorithm if needed
    nx.draw(G_original, pos_original, with_labels=True)
    plt.title("Original Graph")

    # Visualize the reconstructed graph on the right
    plt.subplot(1, 2, 2)
    pos_reconstructed = nx.spring_layout(G_reconstructed, seed=42)  # You can change the layout algorithm if needed
    nx.draw(G_reconstructed, pos_reconstructed, with_labels=True)
    plt.title("Reconstructed Graph")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    data_folder = "C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//graphs"
    numerical_indicies = [0, 1, 2]
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

    # Set up the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    #
    # # Training loop
    # for epoch in range(50):  # Change this to the number of epochs you want
    #     loss = train_vae(model, train_loader, optimizer)
    #     if epoch % 10 == 0:
    #         print(f"Epoch: {epoch}, Loss: {loss}")
    #
    # # Save the model parameters
    # torch.save(model.state_dict(), 'model_3.pth')

    # Define the model
    model = VAE(in_channels, hidden_channels, out_channels)

    # Load the model parameters
    model.load_state_dict(torch.load('model_3.pth'))

    new_graph = "C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//test_graph_1//7E29_20.pdb.pt"
    original_pdb = 'C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//test_pdb_1//7E29_20.pdb'

    # Compute embeddings for all nodes in the new graph
    model.eval()
    with torch.no_grad():
        # new_data is your new graph data
        new_graph_data = torch.load(new_graph)
        new_graph_data.x[:, numerical_indicies] = (new_graph_data.x[:, numerical_indicies] - dataset.mean) / dataset.std
        mean, log_std = model.encode(new_graph_data.x, new_graph_data.edge_index)
        new_embeddings = model.reparameterize(mean, log_std).cpu().numpy()

        # Reconstruct the graph using the decoder
        reconstructed_graph_data = new_graph_data.clone()
        reconstructed_graph_data.x = model.decode(torch.tensor(new_embeddings))

    # Compute the mean of all embeddings
    mean_embedding = np.mean(new_embeddings, axis=0)

    # Compute the distance of each node to the mean
    distances_to_mean = np.linalg.norm(new_embeddings - mean_embedding, axis=1)

    # Sort nodes by their distance to the mean
    sorted_indices = np.argsort(distances_to_mean)

    # Get the top n_principal_nodes
    n_principal_nodes = math.ceil(new_embeddings.shape[0] / 4)
    principal_nodes_indices = sorted_indices[:n_principal_nodes]

    print("Principal nodes:", principal_nodes_indices)

    distances = cdist(new_embeddings, new_embeddings[principal_nodes_indices])
    labels = np.argmin(distances, axis=1)


    def visualize_clusters(data, labels, original_pdb_path):
        # Parse the original PDB file
        parser = PDBParser()
        structure = parser.get_structure("original", original_pdb_path)

        # Extract the coordinates and identities of each atom
        atom_info = {}
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        atom_info[atom.serial_number] = {'coord': atom.coord.tolist(),
                                                         'residue_name': residue.resname,
                                                         'atom_name': atom.name}

        # Convert the PyTorch Geometric graph data to a NetworkX graph
        G = to_networkx(data)

        # Check if the graph is directed
        if data.is_directed():
            G = G.to_directed()

        # Create a color map from cluster labels
        cmap = [labels[node] for node in G]

        plt.figure(figsize=(8, 8))
        nx.draw(G, node_color=cmap, with_labels=True, cmap=plt.cm.tab10)
        plt.show()

        # Save the coarse grained graph to a PDB file
        coarse_grained_graph = Structure.Structure("H")
        model = Model.Model(0)  # add a model with model_id = 0
        chain = Chain.Chain("A")
        model.add(chain)
        coarse_grained_graph.add(model)

        # Create clusters of atoms (or 'beads')
        clusters = {}
        for node, label in enumerate(labels):
            if label in clusters:
                clusters[label].append(node)
            else:
                clusters[label] = [node]

        # Assign atom coordinates in each cluster by their centroid
        for label, nodes in clusters.items():
            # Use the coordinates from the original PDB file
            try:
                centroid = np.mean([atom_info[node + 1]['coord'] for node in nodes], axis=0)
            except KeyError as e:
                print(f'KeyError: {e} not found in atom_info')

            # Determine the most common atom and residue type for the atoms in the cluster
            atom_types = [atom_info[node + 1]['atom_name'] for node in nodes]
            residue_types = [atom_info[node + 1]['residue_name'] for node in nodes]

            most_common_atom_type = Counter(atom_types).most_common(1)[0][0]
            most_common_residue_type = Counter(residue_types).most_common(1)[0][0]
            most_common_element = most_common_atom_type[
                0]  # Assumes the first character of atom name is the element type

            residue = Residue.Residue((" ", label, " "), most_common_residue_type, " ")
            atom = Atom.Atom(most_common_atom_type, centroid.tolist(), 1, 0, " ",
                             most_common_atom_type, label, most_common_element)
            residue.add(atom)
            chain.add(residue)

        io = PDBIO()
        io.set_structure(coarse_grained_graph)
        io.save("coarse_grained_graph.pdb_4")


    visualize_clusters(new_graph_data, labels, original_pdb)

    # Convert the PyTorch Geometric graph data to a NetworkX graph
    G = to_networkx(new_graph_data)

    # Check if the graph is directed
    if new_graph_data.is_directed():
        G = G.to_directed()

    # Step 1: Group nodes based on cluster labels
    clusters = {}
    for node, label in zip(G.nodes(), labels):
        if label not in clusters:
            clusters[label] = [node]
        else:
            clusters[label].append(node)

    # Step 2: Plot all nodes in the same plot with different colors for each cluster
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G)  # Adjust the layout algorithm if needed

    for i, cluster_nodes in enumerate(clusters.values()):
        nx.draw_networkx_nodes(G, pos, nodelist=cluster_nodes, node_color='C{}'.format(i), cmap=plt.cm.tab10,
                               node_size=200)

    nx.draw_networkx_edges(G, pos)
    plt.title(f"Clusters")
    plt.show()

    # Visualize the original and reconstructed graphs
    visualize_original_and_reconstructed(new_graph_data, reconstructed_graph_data)
