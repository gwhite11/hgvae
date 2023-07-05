import torch
import torch.nn as nn
import random
import os
from torch_geometric.nn import GraphConv, GCNConv, TopKPooling, GraphSAGE
from torch_geometric.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pytorch_lightning as pl


random_seed = 42
random.seed(random_seed)
torch.manual_seed(random_seed)


class VariationalAutoencoder(pl.LightningModule):
    def __init__(self, input_dim, latent_dim, num_layers, num_levels, coarse_grain_dims, dropout_rate=0.5):
        super(VariationalAutoencoder, self).__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pooling = nn.ModuleList()
        self.dropout = nn.ModuleList()

        for level in range(num_levels):
            input_dim_level = input_dim if level == 0 else coarse_grain_dims[level - 1]
            output_dim_level = coarse_grain_dims[level]

            encoder = GraphSAGE(input_dim_level, latent_dim, num_layers, aggr='mean')

            self.encoder.append(encoder)

            decoder = GCNConv(latent_dim, output_dim_level)
            self.decoder.append(decoder)

            dropout = nn.Dropout(p=dropout_rate)
            self.dropout.append(dropout)

            pool = TopKPooling(latent_dim)
            self.pooling.append(pool)

        self.clustering = AgglomerativeClustering(n_clusters=coarse_grain_dims[-1])

        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)

        self.learning_rate = None  # Define learning_rate attribute

    def encode(self, xs, edge_indices):
        for encoder, dropout, pool, x, edge_index in zip(self.encoder, self.dropout, self.pooling, xs, edge_indices):
            x = dropout(x)
            print(edge_index.shape)
            x = encoder(x, edge_index)  # Directly pass edge_index
            x, edge_index, _, batch, _, _ = pool(x, edge_index)
        x = x.mean(dim=0)  # Global pooling
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z, edge_index):
        for decoder in self.decoder:
            z = decoder(z, edge_index)
        return z

    def forward(self, xs, edge_indices):
        print(edge_indices.shape)
        mu, logvar = self.encode(xs, edge_indices)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, edge_indices[-1]), mu, logvar  # Directly pass edge_index

    def training_step(self, batch, batch_idx):
        x = batch.x
        edge_index = batch.edge_index

        outputs, mu, logvar = self(x, edge_index)
        reconstruction_loss = 0.0
        for level_output in outputs:
            reconstruction_loss += nn.MSELoss()(level_output, x[:level_output.size(0)])

        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = reconstruction_loss + kl_loss

        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True)  # Use on_step=True
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch.x
        edge_index = batch.edge_index

        outputs, mu, logvar = self(x, edge_index)
        reconstruction_loss = 0.0
        for level_output in outputs:
            reconstruction_loss += nn.MSELoss()(level_output, x[:level_output.size(0)])

        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = reconstruction_loss + kl_loss

        self.log("val_loss", loss, prog_bar=True, logger=True, on_step=True)  # Use on_step=True
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def create_coarse_graph(self, cluster_labels, original_edge_index, x):
        unique_labels = np.unique(cluster_labels)
        num_clusters = len(unique_labels)
        coarse_grained_reps = []

        for label in unique_labels:
            cluster_indices = np.where(cluster_labels == label)[0]
            cluster_x = x[cluster_indices]
            cluster_mean = torch.mean(cluster_x, dim=0)
            coarse_grained_reps.append(cluster_mean)

        return torch.stack(coarse_grained_reps), original_edge_index

    def coarse_grain_representation(self, x, labels):
        num_clusters = torch.max(labels) + 1
        coarse_grained_reps = []

        for i in range(num_clusters):
            cluster_indices = torch.where(labels == i)[0]
            cluster_x = x[cluster_indices]
            cluster_mean = torch.mean(cluster_x, dim=0)
            coarse_grained_reps.append(cluster_mean)

        return torch.stack(coarse_grained_reps, dim=0)

    def hierarchical_clustering(self, x, edge_index, num_layers, coarse_grain_dims):
        self.labels = np.zeros((x.shape[0], num_layers))
        edge_indices = []

        for level in range(num_layers):
            if level == 0:
                clustering = self.clustering
            else:
                clustering = AgglomerativeClustering(n_clusters=coarse_grain_dims[level])

            clustering.fit(x)
            self.labels[:, level] = clustering.labels_

            coarse_edge_index, edge_index = self.create_coarse_graph(torch.tensor(self.labels[:, level], dtype=torch.long), edge_index, x)
            edge_indices.append(coarse_edge_index.unsqueeze(0).t())  # Reshape edge_index

            x = self.coarse_grain_representation(x, self.labels[:, level])

        return self.labels, edge_indices


def estimate_latent_dim(data_list, variance_threshold=0.95):
    node_features = torch.cat([data.x for data in data_list], dim=0)
    pca = PCA()
    pca.fit(node_features)
    explained_variance_ratio = torch.tensor(pca.explained_variance_ratio_, dtype=torch.float32)
    cumulative_explained_variance = torch.cumsum(explained_variance_ratio, dim=0)
    variance_threshold_tensor = torch.tensor(variance_threshold, dtype=torch.float32)
    latent_dim = torch.nonzero(cumulative_explained_variance > variance_threshold_tensor.item()).min().item()
    return latent_dim


def random_translate(data, translate_range):
    translation = torch.FloatTensor(data.pos.size()).uniform_(-translate_range, translate_range)
    data.pos += translation
    return data


input_dim = 3
variance_threshold = 0.95
hidden_dim = 128
num_layers = 2  # Fill in the number of layers
num_levels = 3
coarse_grain_dims = [3, 3, 3]
batch_size = 32
dropout_rate = 0.5
num_epochs = 500
learning_rate = 0.001
num_folds = 5
translate_range = 0.1

input_files_directory = 'C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//graph_data_test'
data_list = []
for filename in os.listdir(input_files_directory):
    if filename.endswith(".pt"):
        file_path = os.path.join(input_files_directory, filename)
        data = torch.load(file_path)
        data_list.append(data)

latent_dim = estimate_latent_dim(data_list, variance_threshold)

train_data = []
val_data = []
for i, data in enumerate(data_list):
    if i % num_folds == 0:
        val_data.append(data)
    else:
        train_data.append(data)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)

model = VariationalAutoencoder(input_dim, latent_dim, num_layers, num_levels, coarse_grain_dims, dropout_rate)

model.learning_rate = learning_rate

trainer = pl.Trainer(max_epochs=num_epochs)
trainer.fit(model, train_loader, val_loader)

# Save the trained model
torch.save(model.state_dict(), 'best_model.pt')

# Load the trained model
model = VariationalAutoencoder(input_dim, latent_dim, num_layers, num_levels, coarse_grain_dims, dropout_rate)
model.load_state_dict(torch.load('best_model.pt'))
model.eval()

# Load the new protein data
new_data = torch.load('C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//chi_graph//chig.pdb.pt')

# Apply the model to the new data
new_data_loader = DataLoader([new_data], batch_size=1)

new_coarse_grained_reps = []
new_reconstructed_reps = []

for data in new_data_loader:
    x, edge_index, batch = data.x, data.edge_index, data.batch
    outputs, mu, logvar = model(x, edge_index)

    # Coarse-grained representations at different levels
    x_level = model.encoder[0](x, edge_index)
    labels, edge_indices = model.hierarchical_clustering(x_level, edge_index, num_levels, coarse_grain_dims)
    coarse_grained_reps = []
    for level in range(num_levels):
        level_labels = labels[:, level]
        coarse_grained_rep = model.coarse_grain_representation(x_level, level_labels)
        coarse_grained_reps.append(coarse_grained_rep)
    new_coarse_grained_reps.append(coarse_grained_reps)

    # Reconstructed representations
    reconstructed_reps = model.decoder[-1](outputs, edge_index)
    new_reconstructed_reps.append(reconstructed_reps)

# Save the coarse-grained representations
torch.save(new_coarse_grained_reps, 'new_protein_coarse_grained_reps.pt')

# Save the reconstructed representations
torch.save(new_reconstructed_reps, 'new_protein_reconstructed_reps.pt')
