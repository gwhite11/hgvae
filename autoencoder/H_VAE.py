import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
import glob
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data.dataset import random_split
from torch_geometric.data import InMemoryDataset


def load_data(graph_files):
    data_list = []
    for graph_file in graph_files:
        # load graph data
        data = torch.load(graph_file)
        # build graph
        data.edge_index = build_graph(data.x)
        data_list.append(data)
    return data_list


# Load the data
graph_files = glob.glob('path_to_your_graph_files/*.pt')
data_list = load_data(graph_files)


def build_graph(x, distance_threshold=5.0):
    edge_index = []
    for i, xi in enumerate(x):
        for j, xj in enumerate(x):
            if i != j and torch.norm(xi - xj) < distance_threshold:
                edge_index.append([i, j])
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()


def estimate_latent_dim_from_graphs(graphs, var_threshold=0.9):
    """
    A simple function to estimate the number of principal components needed to explain var_threshold variance.

    Parameters:
    graphs (list[torch_geometric.data.Data]): a list of graph data
    var_threshold (float): minimum explained variance ratio.

    Returns:
    latent_dim (int): the estimated number of principal components.
    """
    # Concatenate all node features
    x = torch.cat([g.x for g in graphs], dim=0)

    # Center the data
    x = x - x.mean(dim=0)

    # Compute the covariance matrix
    cov_matrix = (x.t() @ x) / (x.size(0) - 1)

    # Perform SVD (Singular Value Decomposition)
    U, S, V = torch.svd(cov_matrix)

    # Compute the explained variance and find the smallest number of components
    # needed to explain at least var_threshold total variance.
    explained_variance = S / S.sum()
    cumulative_explained_variance = torch.cumsum(explained_variance, dim=0)
    latent_dim = torch.searchsorted(cumulative_explained_variance, var_threshold, right=True) + 1

    return latent_dim.item()


# Define your dimensions
input_dim = 3
hidden_dim = 64

# Estimate the latent dim
latent_dim = estimate_latent_dim_from_graphs(data_list)

print('Estimated latent dimension: ', latent_dim)


class VAE(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        self.encoder = torch.nn.Sequential(
            GCNConv(input_dim, hidden_dim),
            torch.nn.ReLU(),
            GCNConv(hidden_dim, latent_dim * 2)
        )

        self.decoder = torch.nn.Sequential(
            GCNConv(latent_dim, hidden_dim),
            torch.nn.ReLU(),
            GCNConv(hidden_dim, input_dim)
        )

    def encode(self, x, edge_index):
        x = self.encoder(x, edge_index)
        return x[:, :x.size(1) // 2], x[:, x.size(1) // 2:]

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(logvar) * torch.exp(logvar)
        else:
            return mu

    def decode(self, z, edge_index):
        return self.decoder(z, edge_index)

    def forward(self, x, edge_index):
        mu, logvar = self.encode(x, edge_index)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, edge_index), mu, logvar

    def training_step(self, batch, batch_idx):
        x, edge_index = batch.x, batch.edge_index
        recon_x, mu, logvar = self(x, edge_index)
        loss = F.mse_loss(recon_x, x) - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, edge_index = batch.x, batch.edge_index
        recon_x, mu, logvar = self(x, edge_index)
        loss = F.mse_loss(recon_x, x) - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


# Initialize the list of VAEs
vae_models = [VAE(input_dim, hidden_dim, latent_dim) for _ in range(2)]


class ProteinGraphDataset(InMemoryDataset):
    def __init__(self, root, data_list, transform=None, pre_transform=None):
        super(ProteinGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = self.collate(data_list)

    def get(self, idx):
        return self.data.__getitem__(idx)


# Create the InMemoryDataset from the list
dataset = ProteinGraphDataset(None, data_list)

# Define the split sizes for train, validation and test
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

# Split the data
train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

# Create data loaders
train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
val_loader = DataLoader(val_data, batch_size=1, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

# Create the trainer with the model checkpoint callback
checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min')
trainer = pl.Trainer(max_epochs=100, callbacks=[checkpoint_callback])

for i, model in enumerate(vae_models):
    # Train the model
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min')
    trainer = pl.Trainer(max_epochs=100, callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader, val_loader)

    # Test the model
    trainer.test(model, dataloaders=test_loader)

    # Use the encoder part of the VAE to reduce the dimensionality of the graphs
    model.eval()
    with torch.no_grad():
        for j, data in enumerate(data_list):
            mu, logvar = model.encode(data.x, data.edge_index)
            z = model.reparameterize(mu, logvar)
            coarse_grained_x = mu
            coarse_grained_edge_index = build_graph(coarse_grained_x)

            # Update the graph with the coarse-grained version
            data_list[j] = Data(x=coarse_grained_x, edge_index=coarse_grained_edge_index)

            # Save the coarse-grained graph after each level
            torch.save(data_list[j], f'path_to_save_your_graphs/level_{i + 1}/' + data.name)

            # Generate and save reconstructed graph only for the first model
            if i == 0:
                # Generate reconstructed graph
                reconstructed_x = model.decode(z, data.edge_index)
                reconstructed_edge_index = build_graph(reconstructed_x)

                # Save the reconstructed graph
                reconstructed_data = Data(x=reconstructed_x, edge_index=reconstructed_edge_index)
                torch.save(reconstructed_data, f'path_to_save_your_reconstructed_graphs/level_{i + 1}/' + data.name)
