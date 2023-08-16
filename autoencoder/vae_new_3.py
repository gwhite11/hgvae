import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, TopKPooling
from torch.nn import Linear, Dropout, BatchNorm1d
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import distance_matrix
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from collections import defaultdict
from Bio.PDB import *


class VAE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_samples=10, pooling_ratio=0.5):
        super(VAE, self).__init__()
        self.dropout_prob = 0.5
        self.n_samples = n_samples

        self.pool1 = TopKPooling(hidden_channels, ratio=pooling_ratio)
        self.pool2 = TopKPooling(2 * hidden_channels, ratio=pooling_ratio)

        self.encoder1 = torch.nn.ModuleList([
            SAGEConv(in_channels, hidden_channels),
            Dropout(self.dropout_prob),
            BatchNorm1d(hidden_channels)
        ])

        self.encoder2 = torch.nn.ModuleList([
            SAGEConv(hidden_channels, 2 * hidden_channels),
            Dropout(self.dropout_prob),
            BatchNorm1d(2 * hidden_channels),
            SAGEConv(2 * hidden_channels, 2 * out_channels)
        ])

        self.decoder1 = torch.nn.ModuleList([
            Linear(out_channels, hidden_channels),
            Dropout(self.dropout_prob),
            BatchNorm1d(hidden_channels),
        ])

        self.decoder2 = torch.nn.ModuleList([
            Linear(hidden_channels, in_channels)
        ])

        self.out_channels = out_channels

    def encode(self, x, edge_index):
        # First encoder layer
        x1 = F.relu(self.encoder1[0](x, edge_index))
        x1 = self.encoder1[1](x1)
        x1 = self.encoder1[2](x1)

        # Pooling after first encoder layer
        x1_pooled, edge_index_pooled, _, batch, perm1, _ = self.pool1(x1, edge_index)

        # Print shapes for debugging
        print("x1_pooled shape:", x1_pooled.shape)
        print("perm1 shape:", perm1.shape)

        # Second encoder layer
        x2 = F.relu(self.encoder2[0](x1_pooled, edge_index_pooled))
        x2 = self.encoder2[1](x2)
        x2 = self.encoder2[2](x2)
        mean, log_std = self.encoder2[3](x2, edge_index_pooled).chunk(2, dim=-1)

        # Pooling after second encoder layer
        x2_pooled, edge_index2_pooled, _, _, perm2, _ = self.pool2(x2, edge_index_pooled)

        # Print shapes for debugging
        print("x2_pooled shape:", x2_pooled.shape)
        print("perm2 shape:", perm2.shape)

        return x1, x1_pooled, x2, x2_pooled, mean, log_std, edge_index_pooled, edge_index2_pooled, perm1, perm2

    def decode(self, z, x1, x2, perm1, perm2):
        h = F.relu(self.decoder1[0](z))
        h = self.decoder1[1](h)
        h = self.decoder1[2](h)

        # Print the size of perm2 for debugging
        print("perm2 size:", perm2.size())

        # First unpooling
        h_unpooled_1 = self.unpool(h, perm2, x2.size(0))

        # Print shapes for debugging
        print("h_unpooled_1 shape:", h_unpooled_1.shape)

        # Second unpooling and adding coarse information
        h_unpooled_2 = self.unpool(h_unpooled_1, perm1, x1.size(0)) + x1

        # Print shapes for debugging
        print("h_unpooled_2 shape:", h_unpooled_2.shape)

        return self.decoder2[0](h_unpooled_2)

    def unpool(self, x_pooled, perm, num_nodes):
        num_features = x_pooled.size(1)
        x_unpooled = torch.zeros(num_nodes, num_features, device=x_pooled.device)

        # This will ensure only valid indices in perm will be filled
        x_unpooled[perm[:x_pooled.size(0)]] = x_pooled

        print("x_unpooled shape:", x_unpooled.shape)

        return x_unpooled
    def reparameterize(self, mean, log_std):
        std = log_std.exp()
        z_samples = [mean + std * torch.randn_like(std) for _ in range(self.n_samples)]
        z = torch.mean(torch.stack(z_samples), dim=0)
        return z

    def graph_laplacian_regularization(self, z, edge_index):
        source_nodes, target_nodes = edge_index[0], edge_index[1]
        edge_diffs = z[source_nodes] - z[target_nodes]
        return torch.sum(edge_diffs.pow(2))

    def forward(self, x, edge_index):
        x1, x1_pooled, x2, x2_pooled, mean, log_std, edge_index_pooled, edge_index2_pooled, perm1, perm2 = self.encode(x, edge_index)
        z = self.reparameterize(mean, log_std)
        laplacian_reg = self.graph_laplacian_regularization(z, edge_index_pooled)
        reconstruction = self.decode(z, x1, x2, perm1, perm2)

        pooled_graph = Data(x=x1_pooled, edge_index=edge_index_pooled)
        double_pooled_graph = Data(x=x2_pooled, edge_index=edge_index2_pooled)

        return reconstruction, laplacian_reg, mean, log_std, pooled_graph, double_pooled_graph

    def l_fp(self, pooled_graph, double_pooled_graph):
        return F.mse_loss(pooled_graph.x, double_pooled_graph.x)

    def l_similarity(self, original_graph, pooled_graph):
        # Assuming node features are the 0-th attribute in the data
        return F.mse_loss(pooled_graph.x, torch.mean(original_graph.x, dim=0))

    def l_regularization(self, original_graph, pooled_graph):
        # Ensuring the pooled graph is not too dissimilar from the original
        return F.mse_loss(original_graph.x, pooled_graph.x)

    def global_mean_pool(self, z):
        return torch.mean(z, dim=0)

    def recon_loss(self, x, mean, log_std, edge_index):
        x1, _, _, _, _, _, _, perm2 = self.encode(x, edge_index)  # Extract the necessary tensor for reconstruction
        z = self.reparameterize(mean, log_std)
        x_recon = self.decode(z, x1, perm2)  # Decode using the perm2 tensor
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        return recon_loss

    def kl_divergence(self, mean, log_std):
        std = log_std.exp()
        kl_loss = -0.5 * torch.sum(1 + 2 * log_std - mean.pow(2) - std.pow(2), dim=-1)
        return kl_loss.mean()

    def compute_loss(self, data):
        reconstruction, laplacian_reg, mean, log_std, pooled_graph, double_pooled_graph, perm1, perm2 = self(data.x,
                                                                                                             data.edge_index)
        x1 = self.encode(data.x, data.edge_index)[0]
        recon_loss = self.recon_loss(data.x, mean, log_std, data.edge_index, x1)
        kl_loss = self.kl_divergence(mean, log_std)
        lambda_reg = adjust_lambda_reg(kl_loss)
        loss_vae = recon_loss + kl_loss + lambda_reg * laplacian_reg

        L_FP = self.l_fp(pooled_graph, double_pooled_graph)
        L_similarity = self.l_similarity(data, pooled_graph)
        L_regularization = self.l_regularization(data, pooled_graph)

        alpha, beta, gamma = 1.0, 1.0, 1.0
        total_loss = loss_vae + alpha * L_FP + beta * L_similarity + gamma * L_regularization

        return total_loss


def train_vae(model, loader, optimizer):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        loss = model.compute_loss(data)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# Dynamic Lambda Regulation
def adjust_lambda_reg(kl_loss, threshold=0.1):
    if kl_loss > threshold:
        return 0.01
    else:
        return 0.001


# Validation function
def validate_vae(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            loss = model.compute_loss(data)
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
    num_epochs = 50

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    best_validation_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss = train_vae(model, train_loader, optimizer)
        val_loss = validate_vae(model, valid_loader)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    best_validation_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss = train_vae(model, train_loader, optimizer)
        val_loss = validate_vae(model, valid_loader)

        # Early stopping based on validation loss (for simplicity, we stop if the validation loss doesn't improve for 10 epochs)
        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            patience = 10
            torch.save(model.state_dict(), 'model_best.pth')
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping!")
                break

        scheduler.step()
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}")

    # Save the model parameters
    torch.save(model.state_dict(), 'model_new_10.pth')

    # Define the model
    model = VAE(in_channels, hidden_channels, out_channels)

    # Load the model parameters
    model.load_state_dict(torch.load('C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//model_new_10.pth'))

    new_graph = "C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//chi_graph//chig.pdb.pt"
    original_pdb = 'C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//input_chig//chig.pdb'

    # Compute embeddings for all nodes in the new graph
    model.eval()
    model.eval()
    with torch.no_grad():
        new_graph_data = torch.load(new_graph)

        # Normalize only numerical features using the mean and std computed from the training dataset
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


def list_atoms_per_cluster(labels, atom_info, output_file="clusters_info_10.txt"):
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
output_pdb_path = "colored_clusters_10.pdb"
original_pdb = 'C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//input_chig//chig.pdb'

# Generate the colored PDB file
generate_colored_pdb(labels, original_pdb, output_pdb_path)