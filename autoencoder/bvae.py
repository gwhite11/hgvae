import torch
import torch.nn as nn
import random
import os
from torch_geometric.nn import GCNConv, TopKPooling, GATConv, global_mean_pool
from torch_geometric.data import DataLoader, Data
from sklearn.decomposition import PCA
import numpy as np
import pytorch_lightning as pl

random_seed = 42
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)


class HierarchicalCoarseGraining(pl.LightningModule):
    def __init__(self, input_dim, latent_dim, num_levels, coarse_grain_dims, dropout_rate=0.5):
        super(HierarchicalCoarseGraining, self).__init__()

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pooling = nn.ModuleList()
        self.dropout = nn.ModuleList()

        # new addition for B-VAE
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_var = nn.Linear(latent_dim, latent_dim)

        for level in range(num_levels):
            input_dim_level = input_dim if level == 0 else coarse_grain_dims[level - 1]
            output_dim_level = coarse_grain_dims[level]

            if level == 0:
                encoder = GCNConv(input_dim_level, latent_dim)
            else:
                encoder = GATConv(input_dim_level, latent_dim)

            self.encoders.append(encoder)

            decoder = GCNConv(latent_dim, output_dim_level)
            self.decoders.append(decoder)

            dropout = nn.Dropout(p=dropout_rate)
            self.dropout.append(dropout)

            pool = TopKPooling(latent_dim)
            self.pooling.append(pool)

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, edge_index):
        outputs = []
        batch_indices = []

        for encoder, decoder, dropout, pool in zip(self.encoders, self.decoders, self.dropout, self.pooling):
            x = dropout(x)
            z = encoder(x, edge_index)
            mu = self.fc_mu(z)
            log_var = self.fc_var(z)
            z = self.reparameterize(mu, log_var)
            x, edge_index, _, batch, _, _ = pool(z, edge_index)
            x = decoder(x, edge_index)

            outputs.append(x)
            batch_indices.append(batch)

        return outputs, batch_indices, mu, log_var

    def loss_function(self, outputs, x, mu, log_var):
        reconstruction_loss = 0.0
        for level_output in outputs:
            reconstruction_loss += self.reconstruction_loss_function(level_output, x[:level_output.size(0)])
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return reconstruction_loss + KLD

    def training_step(self, batch, batch_idx):
        x = batch.x
        edge_index = batch.edge_index

        outputs, _, mu, log_var = self(x, edge_index)

        loss = self.loss_function(outputs, x, mu, log_var)

        self.log("train_loss", loss, prog_bar=True, logger=True, batch_size=batch.x.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch.x
        edge_index = batch.edge_index

        outputs, _, mu, log_var = self(x, edge_index)

        loss = self.loss_function(outputs, x, mu, log_var)

        self.log("val_loss", loss, prog_bar=True, logger=True, batch_size=batch.x.size(0))
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def reconstruction_loss_function(self, recon_x, x):
        MSE = nn.MSELoss()
        return MSE(recon_x, x)


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
new_data = torch.load('C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//chi_graph')

# Apply the model to the new data
new_data_loader = DataLoader([new_data], batch_size=1)

new_coarse_grained_reps = []
new_reconstructed_reps = []

for data in new_data_loader:
    x, edge_index, batch = data.x, data.edge_index, data.batch
    outputs, batch_indices, _, _ = model(x, edge_index)

    # Coarse-grained representations
    coarse_grained_reps = []
    for level_output, batch_index in zip(outputs, batch_indices):
        graph = Data(x=level_output, edge_index=edge_index, batch=batch_index)
        coarse_grained_reps.append(graph)
    new_coarse_grained_reps.append(coarse_grained_reps)

    # Reconstructed representations
    reconstructed_reps = model.decoders[-1](outputs[-1], edge_index)
    new_reconstructed_reps.append(reconstructed_reps)

# Save the coarse-grained representations
torch.save(new_coarse_grained_reps, 'new_protein_coarse_grained_reps.pt')

# Save the reconstructed representations
torch.save(new_reconstructed_reps, 'new_protein_reconstructed_reps.pt')
