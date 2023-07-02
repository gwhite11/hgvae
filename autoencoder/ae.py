import torch
import torch.nn as nn
import random
import os
from torch_geometric.nn import GCNConv, TopKPooling, GATConv
from torch_geometric.data import DataLoader, Data
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pytorch_lightning as pl

random_seed = 42
random.seed(random_seed)
torch.manual_seed(random_seed)


class HierarchicalCoarseGraining(pl.LightningModule):
    def __init__(self, input_dim, latent_dim, num_levels, coarse_grain_dims, dropout_rate=0.5):
        super(HierarchicalCoarseGraining, self).__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pooling = nn.ModuleList()
        self.dropout = nn.ModuleList()

        for level in range(num_levels):
            input_dim_level = input_dim if level == 0 else coarse_grain_dims[level - 1]
            output_dim_level = coarse_grain_dims[level]

            if level == 0:
                encoder = GCNConv(input_dim_level, latent_dim)
            else:
                encoder = GATConv(input_dim_level, latent_dim)

            self.encoder.append(encoder)

            decoder = GCNConv(latent_dim, output_dim_level)
            self.decoder.append(decoder)

            dropout = nn.Dropout(p=dropout_rate)
            self.dropout.append(dropout)

            pool = TopKPooling(latent_dim)
            self.pooling.append(pool)

        self.clustering = AgglomerativeClustering(n_clusters=coarse_grain_dims[-1])

    def forward(self, x, edge_index):
        outputs = []
        batch_indices = []

        for encoder, decoder, dropout, pool in zip(self.encoder, self.decoder, self.dropout, self.pooling):
            x = dropout(x)
            x = encoder(x, edge_index)
            x, edge_index, _, batch, _, _ = pool(x, edge_index)
            x = decoder(x, edge_index)

            outputs.append(x)
            batch_indices.append(batch)

        return outputs, batch_indices

    def training_step(self, batch, batch_idx):
        x = batch.x
        edge_index = batch.edge_index

        outputs, _ = self(x, edge_index)

        reconstruction_loss = 0.0
        for level_output in outputs:
            reconstruction_loss += self.reconstruction_loss_function(level_output, x[:level_output.size(0)])

        self.log("train_loss", reconstruction_loss, prog_bar=True, logger=True, batch_size=batch.x.size(0))
        return reconstruction_loss

    def validation_step(self, batch, batch_idx):
        x = batch.x
        edge_index = batch.edge_index

        outputs, _ = self(x, edge_index)

        reconstruction_loss = 0.0
        for level_output in outputs:
            reconstruction_loss += self.reconstruction_loss_function(level_output, x[:level_output.size(0)])

        self.log("val_loss", reconstruction_loss, prog_bar=True, logger=True, batch_size=batch.x.size(0))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def coarse_grain_representation(self, x, labels):
        unique_labels = np.unique(labels)
        num_clusters = len(unique_labels)
        coarse_grained_reps = []

        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            cluster_x = x[cluster_indices]
            cluster_mean = np.mean(cluster_x, axis=0)
            coarse_grained_reps.append(cluster_mean)

        return torch.tensor(coarse_grained_reps)

    def hierarchical_clustering(self, x, num_levels):
        cluster_labels = np.zeros((x.shape[0], num_levels))

        for level in range(num_levels):
            if level == 0:
                clustering = self.clustering
            else:
                clustering = AgglomerativeClustering(n_clusters=coarse_grain_dims[level])

            clustering.fit(x)
            labels = clustering.labels_
            cluster_labels[:, level] = labels
            x = self.coarse_grain_representation(x, labels)

        return cluster_labels


def estimate_latent_dim(data_list, variance_threshold=0.95):
    node_features = torch.cat([data.x for data in data_list], dim=0).detach().numpy()
    pca = PCA()
    pca.fit(node_features)
    cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
    latent_dim = np.argmax(cumulative_explained_variance > variance_threshold)
    return latent_dim


def random_translate(data, translate_range):
    translation = torch.FloatTensor(data.pos.size()).uniform_(-translate_range, translate_range)
    data.pos += translation
    return data


input_dim = 3
variance_threshold = 0.95
hidden_dim = 128
num_levels = 3
coarse_grain_dims = [3, 3, 3]
batch_size = 32
dropout_rate = 0.5
num_epochs = 1000
learning_rate = 0.001
num_folds = 5
translate_range = 0.1

input_files_directory = 'C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//graph_data'
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

model = HierarchicalCoarseGraining(input_dim, latent_dim, num_levels, coarse_grain_dims, dropout_rate)

model.reconstruction_loss_function = nn.MSELoss()
model.learning_rate = learning_rate

trainer = pl.Trainer(max_epochs=num_epochs)
trainer.fit(model, train_loader, val_loader)

# Save the trained model
torch.save(model.state_dict(), 'best_model.pt')

# Load the trained model
model = HierarchicalCoarseGraining(input_dim, latent_dim, num_levels, coarse_grain_dims, dropout_rate)
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
    outputs, _ = model(x, edge_index)

    # Coarse-grained representations at different levels
    x_level = model.encoder[0](outputs[0], edge_index)
    labels = model.hierarchical_clustering(x_level, num_levels)
    coarse_grained_reps = []
    for level in range(num_levels):
        level_labels = labels[:, level]
        coarse_grained_rep = model.coarse_grain_representation(x_level, level_labels)
        coarse_grained_reps.append(coarse_grained_rep)
    new_coarse_grained_reps.append(coarse_grained_reps)

    # Reconstructed representations
    reconstructed_reps = model.decoder[-1](outputs[-1], edge_index)
    new_reconstructed_reps.append(reconstructed_reps)

# Save the coarse-grained representations
torch.save(new_coarse_grained_reps, 'new_protein_coarse_grained_reps.pt')

# Save the reconstructed representations
torch.save(new_reconstructed_reps, 'new_protein_reconstructed_reps.pt')
