import os
import torch
from torch.nn import ModuleList
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from sklearn.cluster import KMeans
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx


class GAE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GAE, self).__init__()
        self.encoder = ModuleList([
            GCNConv(in_channels, hidden_channels),
            GCNConv(hidden_channels, out_channels)
        ])

    def encode(self, x, edge_index):
        for conv in self.encoder:
            x = conv(x, edge_index)
            x = torch.relu(x)
        return x

    def decode(self, z, edge_index):
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    def recon_loss(self, z, pos_edge_index, neg_edge_index):
        pos_loss = -torch.log(self.decode(z, pos_edge_index) + 1e-15).mean()
        neg_loss = -torch.log(1 - self.decode(z, neg_edge_index) + 1e-15).mean()
        return pos_loss + neg_loss

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return z

def train(model, loader, optimizer):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        z = model(data.x, data.edge_index)
        pos_edge_index = data.edge_index
        neg_edge_index = negative_sampling(pos_edge_index, num_nodes=data.num_nodes)
        loss = model.recon_loss(z, pos_edge_index, neg_edge_index)
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

    # If you want to check the data, you can still do so:
    print("First training graph: ", train_dataset[0])
    print("Node features of the first training graph: ", train_dataset[0].x)
    print("Node features of the first training graph (first 5 nodes): ", train_dataset[0].x[:5])

    # Print first batch from training and validation data
    print("First batch of training graphs: ", next(iter(train_loader)))
    print("First batch of validation graphs: ", next(iter(valid_loader)))


    # Define the model
    in_channels = train_dataset[0].num_node_features
    hidden_channels = 32
    out_channels = 16
    model = GAE(in_channels, hidden_channels, out_channels)

    # Set up the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(500):  # Change this to the number of epochs you want
        loss = train(model, train_loader, optimizer)
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {loss}")

    # Save the model parameters
    torch.save(model.state_dict(), 'model.pth')

    # Define the model
    model = GAE(in_channels, hidden_channels, out_channels)

    # Load the model parameters
    model.load_state_dict(torch.load('model.pth'))

    # Compute embeddings for all nodes in the new graph
    model.eval()
    with torch.no_grad():
        # new_data is your new graph data
        new_graph_data = torch.load(
            'C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//chi_graph//chig.pdb.pt')
        new_graph_data.x[:, numerical_indicies] = (new_graph_data.x[:, numerical_indicies] - dataset.mean) / dataset.std
        new_embeddings = model(new_graph_data.x, new_graph_data.edge_index).cpu().numpy()

    # Apply KMeans to the embeddings
    kmeans = KMeans(n_clusters=10)  # change to the number of clusters you want
    labels = kmeans.fit_predict(new_embeddings)

    print(labels)  # These are your node clusters in the new graph!


    def visualize_clusters(data, labels):
        # Convert the PyTorch Geometric graph data to a NetworkX graph
        G = to_networkx(data)

        # Check if the graph is directed
        if data.is_directed():
            G = G.to_directed()

        # Create a color map from cluster labels
        cmap = []
        for node in G:
            cmap.append(labels[node])

        plt.figure(figsize=(8,8))
        nx.draw(G, node_color=cmap, with_labels=True)
        plt.show()


    visualize_clusters(new_graph_data, labels)
