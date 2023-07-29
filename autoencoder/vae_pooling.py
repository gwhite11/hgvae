import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Batch
from torch_geometric.nn import SAGEConv
from torch.nn import Linear
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx


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

    print(f"Total number of graphs: {len(dataset)}")
    print(f"Number of training graphs: {len(train_dataset)}")
    print(f"Number of validation graphs: {len(valid_dataset)}")

    # Check the data
    print("First training graph: ", train_dataset[0])
    print("Node features of the first training graph: ", train_dataset[0].x)
    print("Node features of the first training graph (first 5 nodes): ", train_dataset[0].x[:5])

    # Print first batch from training and validation data
    print("First batch of training graphs: ", next(iter(train_loader)))
    print("First batch of validation graphs: ", next(iter(valid_loader)))

    # Define the model
    in_channels = train_dataset[0].num_node_features
    hidden_channels = 64
    out_channels = 32
    model = VAE(in_channels, hidden_channels, out_channels)

    # Set up the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(100):  # Change this to the number of epochs you want
        loss = train_vae(model, train_loader, optimizer)
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {loss}")

    # Save the model parameters
    torch.save(model.state_dict(), 'model_3.pth')

    # Define the model
    model = VAE(in_channels, hidden_channels, out_channels)

    # Load the model parameters
    model.load_state_dict(torch.load('model_3.pth'))

    # Compute embeddings for all nodes in the new graph
    model.eval()
    with torch.no_grad():
        # new_data is your new graph data
        new_graph_data = torch.load(
            'C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//chi_graph//chig.pdb.pt')
        new_graph_data.x[:, numerical_indicies] = (new_graph_data.x[:, numerical_indicies] - dataset.mean) / dataset.std
        mean, log_std = model.encode(new_graph_data.x, new_graph_data.edge_index)
        new_embeddings = model.reparameterize(mean, log_std).cpu().numpy()

        # Reconstruct the graph using the decoder
        reconstructed_graph_data = new_graph_data.clone()
        reconstructed_graph_data.x = model.decode(torch.tensor(new_embeddings))

    # Define a range of possible cluster numbers
    min_clusters = 20
    max_clusters = min(50, new_embeddings.shape[0] - 1)

    best_score = -1
    best_n_clusters = 2

    for n_clusters in range(min_clusters, max_clusters + 1):
        spectral = SpectralClustering(n_clusters=n_clusters)
        labels = spectral.fit_predict(new_embeddings)
        score = silhouette_score(new_embeddings, labels)

        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters

    print("Best number of clusters:", best_n_clusters)

    spectral = SpectralClustering(n_clusters=best_n_clusters)
    labels = spectral.fit_predict(new_embeddings)

    print(labels)


    def visualize_clusters(data, labels):
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

    visualize_clusters(new_graph_data, labels)

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



