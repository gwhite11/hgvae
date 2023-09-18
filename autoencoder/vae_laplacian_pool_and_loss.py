import os
import torch
import torch_geometric
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Batch
from torch.nn import Linear, Dropout, BatchNorm1d
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from collections import defaultdict
from collections import Counter
import csv
import networkx as nx
import numpy as np
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain, Residue, Atom
from torch_geometric.nn import SAGEConv, SAGPooling, GCNConv
import pytorch_lightning as pl
from torch.nn.functional import mse_loss

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


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


def compute_fiedler_value(edge_index, num_nodes=None):
    L = normalized_laplacian(edge_index, num_nodes=num_nodes)
    eigvals = torch.linalg.eigvalsh(L)
    return eigvals[1]


class SpectralRegularizedSAGPooling(SAGPooling):
    def __init__(self, in_channels, alpha=0.5, ratio=0.5, GNN=GCNConv, **kwargs):
        super(SpectralRegularizedSAGPooling, self).__init__(in_channels, ratio=ratio, GNN=GNN, **kwargs)

        self.alpha = alpha

    def forward(self, x, edge_index, **kwargs):
        fiedler_before_pooling = compute_fiedler_value(edge_index, x.size(0))
        out = super(SpectralRegularizedSAGPooling, self).forward(x, edge_index, **kwargs)
        fiedler_after_pooling = compute_fiedler_value(out[1], out[0].size(0))

        spectral_loss = (fiedler_before_pooling - fiedler_after_pooling).abs()

        # Convert the tuple to a list for modification
        out_list = list(out)

        if out_list[2] is None:
            out_list[2] = self.alpha * spectral_loss
        else:
            out_list[2] += self.alpha * spectral_loss

        # Convert the list back to a tuple
        out = tuple(out_list)

        return out


class PoolGraph(torch.nn.Module):
    def __init__(self, num_node_features):
        super().__init__()
        self.pool1 = SpectralRegularizedSAGPooling(num_node_features)

    def forward(self, x, edge_index, batch):
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, batch=batch)
        return torch_geometric.data.Data(x=x, edge_index=edge_index, batch=batch)


class PoolUnpoolGraph(torch.nn.Module):
    def __init__(self, num_node_features):
        super().__init__()
        self.pool1 = SpectralRegularizedSAGPooling(num_node_features)

    def forward(self, x, edge_index, batch):
        x_cg, edge_index_cg, _, _, perm, _ = self.pool1(x, edge_index, batch=batch)

        # Create an empty tensor for the unpooled feature matrix
        x_unpooled = torch.zeros(x.size(0), x_cg.size(1), device=x.device)

        # Copy the features of the pooled nodes back to their original positions
        x_unpooled[perm] = x_cg

        edge_index_unpooled = edge_index_cg

        return x_unpooled, edge_index_unpooled, batch


class VAE(pl.LightningModule):
    def __init__(self, in_channels, hidden_channels, out_channels, n_samples=10, beta=0.5):
        super(VAE, self).__init__()
        self.dropout_prob = 0.5
        self.n_samples = n_samples
        self.beta = beta

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
            Linear(out_channels + in_channels, hidden_channels),
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

        # Adding the transformation for x1 here
        self.x1_transform = torch.nn.Linear(hidden_channels, in_channels)

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
        # Transform x1
        x1_transformed = self.x1_transform(x1)

        # Combine z and transformed x1
        combined_input = torch.cat([z, x1_transformed], dim=-1)

        # First-level decoding with combined input
        h = F.relu(self.decoder1[0](combined_input))
        h = self.decoder1[1](h)
        h = self.decoder1[2](h)

        # Second-level decoding to match in_channels
        return self.decoder2[0](h)

    def reparameterize(self, mean, log_std):
        std = log_std.exp()
        z_samples = [mean + std * torch.randn_like(std) for _ in range(self.n_samples)]
        z = torch.mean(torch.stack(z_samples), dim=0)
        return z

    def forward(self, x, edge_index, batch):
        x1, mean, log_std = self.encode(x, edge_index)
        z = self.reparameterize(mean, log_std)
        reconstruction = self.decode(z, x1)

        x_unpooled, edge_index_unpooled, new_batch = self.pool_unpool_graph(x1, edge_index, batch)

        return reconstruction, x_unpooled, edge_index_unpooled, mean, log_std, x1  # Added x1 to the return values

    # Add a function to compute the MSE difference between pooled and unpooled eigenvalues
    def mse_diff_eigen(self, edge_index_pooled, edge_index_unpooled, num_nodes):
        eigvals_pooled = torch.linalg.eigvalsh(normalized_laplacian(edge_index_pooled, num_nodes=num_nodes))
        eigvals_unpooled = torch.linalg.eigvalsh(normalized_laplacian(edge_index_unpooled, num_nodes=num_nodes))
        mse_diff = mse_loss(eigvals_pooled, eigvals_unpooled)
        return mse_diff

    def recon_loss(self, x_recon, x_original, mean, log_std, edge_index, x1, edge_index_unpooled):
        z = self.reparameterize(mean, log_std)
        x_decoded = self.decode(z, x1)
        recon_loss = F.mse_loss(x_decoded, x_original, reduction='mean')

        # Add the eigenvalue MSE difference to the reconstruction loss
        num_nodes = edge_index.max().item() + 1
        eigen_mse_diff = self.mse_diff_eigen(edge_index, edge_index_unpooled, num_nodes)

        # Add a scaling factor (e.g., 0.1) to control the contribution of the eigenvalue loss term
        return recon_loss + 0.1 * eigen_mse_diff

    def kl_divergence(self, mean, log_std):
        std = log_std.exp()
        kl_loss = -0.5 * torch.sum(1 + 2 * log_std - mean.pow(2) - std.pow(2), dim=-1)
        return kl_loss.mean()


def train_vae(model, loader, optimizer, clip_value=None, device=None):
    model.to(device)  # Move the model to GPU

    model.train()
    total_loss = 0
    for data in loader:
        # Move the data to the GPU
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        data.batch = data.batch.to(device)

        optimizer.zero_grad()
        reconstruction, x_unpooled, edge_index_unpooled, mean, log_std, x1 = model(data.x, data.edge_index, data.batch)  # Note the addition of edge_index_unpooled
        recon_loss = model.recon_loss(reconstruction, data.x, mean, log_std, data.edge_index, x1, edge_index_unpooled)  # Pass edge_index_unpooled as well
        kl_loss = model.kl_divergence(mean, log_std)

        # If the pooling layer returns the spectral preservation loss, add it to the total loss
        total_vae_loss = recon_loss + model.beta * kl_loss

        total_vae_loss.backward()
        if clip_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        total_loss += total_vae_loss.item()
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
    data_folder = ".//pdb_files//graphs_2"
    numerical_indicies = [0, 1, 2, 3, 4, 5]
    dataset = CustomGraphDataset(data_folder, numerical_indicies)

    # Define the split size
    train_size = int(0.8 * len(dataset))  # Use 80% of the data for training
    valid_size = len(dataset) - train_size  # Use the rest for validation

    # Perform the split
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    batch_size = 2  # Adjust the batch size if memory crashes

    # Create data loaders for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0,
                              collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0,
                              collate_fn=collate_fn)

    # Define the model
    in_channels = train_dataset[0].num_node_features
    hidden_channels = 168  # try using fewer layers
    out_channels = 42  # try smaller bottleneck
    model = VAE(in_channels, hidden_channels, out_channels, beta=0.5)
    num_epochs = 30
    clip_value = 1

    # Set up the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # try different optimizers?

    # Training loop
    for epoch in range(num_epochs):
        loss = train_vae(model, train_loader, optimizer, clip_value)
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {loss}")

    # Save the model parameters
    torch.save(model.state_dict(), 'model_new_8.pth')

    # Define the model
    model = VAE(in_channels, hidden_channels, out_channels)

    # Load the model parameters
    model.load_state_dict(torch.load('.//model_new_8.pth'))

    new_graph = ".//pdb_files//chi_graph//chig.pdb.pt"
    original_pdb = './/pdb_files//input_chig//chig.pdb'

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

    # Use Agglomerative Clustering
    linked = linkage(new_embeddings, 'ward')

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
original_pdb = './/pdb_files//input_chig//chig.pdb'

# Generate the colored PDB file
generate_colored_pdb(labels, original_pdb, output_pdb_path)


def visualize_clusters(data, labels, original_pdb):
    # Atomic masses mapping
    atomic_masses = {
        'H': 1.008,
        'C': 12.01,
        'N': 14.01,
        'O': 16.00
    }

    # Parse the original PDB file
    parser = PDBParser()
    structure = parser.get_structure("original", original_pdb)

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

    # Save the coarse-grained graph to a PDB file
    coarse_grained_graph = Structure.Structure("H")
    model = Model.Model(0)
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

    # For writing to CSV
    with open('outputs/beads_to_atoms_12.csv', 'w', newline='') as csvfile:
        beadwriter = csv.writer(csvfile)
        beadwriter.writerow(["BeadID", "ImportantAtom", "AllAtomsInCluster", "Mass"])

        # Unique Bead ID and bead types
        unique_bead_id = 1
        bead_types = {}

        # Assign atom coordinates in each cluster by their centroid
        for label, nodes in clusters.items():
            # Use the coordinates from the original PDB file
            centroid_atoms = []
            bead_mass = 0  # initialize the bead's mass

            try:
                centroid = np.mean([atom_info[node + 1]['coord'] for node in nodes], axis=0)
                for node in nodes:
                    atom_name = atom_info[node + 1]['atom_name']
                    centroid_atoms.append(
                        f"{atom_name} ({atom_info[node + 1]['residue_name']})")
                    bead_mass += atomic_masses.get(atom_name[0], 0)  # accumulate mass based on atom type
            except KeyError as e:
                print(f'KeyError: {e} not found in atom_info')

            atom_types = [atom_info[node + 1]['atom_name'] for node in nodes]
            residue_types = [atom_info[node + 1]['residue_name'] for node in nodes]

            most_common_atom_type = Counter(atom_types).most_common(1)[0][0]
            most_common_residue_type = Counter(residue_types).most_common(1)[0][0]
            most_common_element = most_common_atom_type[0]

            # Determine bead type
            atom_set = tuple(sorted(set(atom_types)))
            if atom_set not in bead_types:
                bead_types[atom_set] = unique_bead_id
                unique_bead_id += 1

            # Write to CSV
            beadwriter.writerow([f"B{bead_types[atom_set]:03d}", most_common_atom_type, "; ".join(centroid_atoms), bead_mass])

            # Continue with the PDB writing
            residue = Residue.Residue((" ", label, " "), most_common_residue_type, " ")
            atom = Atom.Atom(most_common_atom_type, centroid.tolist(), 1, 0, " ",
                             most_common_atom_type, label, most_common_element)
            residue.add(atom)
            chain.add(residue)

        io = PDBIO()
        io.set_structure(coarse_grained_graph)
        io.save("coarse_grained_graph_20.pdb")
