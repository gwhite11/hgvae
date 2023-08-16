import os
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Batch
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GraphUNet, diff_pool, diff_unpool
import torch.optim as optim
from torch.nn import Linear
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from Bio.PDB import PDBIO, Chain, Residue, Atom, Model, Structure
import numpy as np
from torch_geometric.utils import add_self_loops


def reconstruction_loss(reconstructed, original):
    # Compare reconstructed node features with original node features
    return F.mse_loss(reconstructed, original)


def fixed_point_loss(pooled_features, double_pooled_features):
    return F.mse_loss(pooled_features, double_pooled_features)


class GraphUNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, depth, pool_ratio):
        super(GraphUNet, self).__init__()
        self.depth = depth
        self.pools = torch.nn.ModuleList()
        self.unpools = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        # Down-sampling (RG Coarse-Graining)
        for i in range(depth):
            self.pools.append(diff_pool)
            self.convs.append(SAGEConv(in_channels if i == 0 else hidden_channels, hidden_channels))

        # Up-sampling (RG Fine-Graining)
        for i in range(depth - 1):
            self.unpools.append(diff_unpool)
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.final_conv = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch=None, compute_loss=True):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        edge_indices = [edge_index]
        xs = [x]
        clusters = []
        losses = {"L_FP": [], "L_similarity": [], "L_regularization": []}

        # Down-sampling steps
        for i in range(self.depth):
            x, edge_index, _, batch, perm, score = self.pools[i](xs[-1], edge_indices[-1], None, batch)
            clusters.append(perm)
            x = F.relu(self.convs[i](x, edge_index))

            if compute_loss:
                # Compute L_FP (use current and next depth if possible)
                if i < self.depth - 1:
                    next_x, _, _, _, _, _ = self.pools[i + 1](x, edge_index, None, batch)
                    losses["L_FP"].append(fixed_point_loss(x, next_x))
                # Compute L_similarity (compare current x to original x)
                losses["L_similarity"].append(F.mse_loss(x, xs[0][perm]))
                # Compute L_regularization (simple difference between pooled and original)
                losses["L_regularization"].append(F.l1_loss(x, xs[-1]))

            xs.append(x)
            edge_indices.append(edge_index)

        # Up-sampling steps
        for i in range(self.depth - 2, -1, -1):
            x = self.unpools[i](xs[-1], clusters[i], xs[i])
            edge_index = edge_indices[i]
            x = F.relu(self.convs[self.depth + i](x, edge_index))

        if compute_loss:
            return self.final_conv(x, edge_indices[0]), losses
        else:
            return self.final_conv(x, edge_indices[0])


class CustomGraphDataset(Dataset):
    def __init__(self, data_folder, numerical_indices):
        self.data_file_list = [os.path.join(data_folder, filename) for filename in os.listdir(data_folder) if
                               filename.endswith('.pt')]
        self.numerical_indices = numerical_indices
        self.mean, self.std = self._compute_mean_std()

    def __len__(self):
        return len(self.data_file_list)

    def __getitem__(self, index):
        graph_data = torch.load(self.data_file_list[index])
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

    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    batch_size = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0,
                              collate_fn=collate_fn)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0,
                              collate_fn=collate_fn)

    # Initialize the model, optimizer, and other settings
    model = GraphUNet(in_channels=dataset.num_node_features, hidden_channels=16, out_channels=dataset.num_classes,
                         depth=3, pool_ratio=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()


    def train():
        model.train()
        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            # Assuming node classification task, you should modify as needed
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)


    def validate():
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for data in val_loader:
                out = model(data.x, data.edge_index, data.batch)
                loss = criterion(out, data.y)
                total_loss += loss.item()
        return total_loss / len(val_loader)


    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        output, losses = model(data.x, data.edge_index, compute_loss=True)

        # Assuming `data.adj` is your adjacency matrix
        primary_loss = reconstruction_loss(output, data.adj)

        alpha, beta, gamma = 1.0, 1.0, 1.0  # adjust these based on your needs
        total_loss = primary_loss
        total_loss += alpha * sum(losses["L_FP"])
        total_loss += beta * sum(losses["L_similarity"])
        total_loss += gamma * sum(losses["L_regularization"])
        print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
