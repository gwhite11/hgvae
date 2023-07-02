import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import GCNConv, TopKPooling, GATConv
from sklearn.decomposition import PCA
import pytorch_lightning as pl
from torch.distributions import Normal


class HierarchicalCoarseGraining(pl.LightningModule):
    def __init__(self, input_dim, latent_dim, num_levels, coarse_grain_dims, dropout_rate, beta=1.0):

        super(HierarchicalCoarseGraining, self).__init__()

        # add beta as a class attribute
        self.beta = beta

        self.latent_dim = latent_dim
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pooling = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for level in range(num_levels):
            input_dim_level = input_dim if level == 0 else coarse_grain_dims[level - 1]
            output_dim_level = coarse_grain_dims[level]

            if level == 0:
                encoder = GCNConv(input_dim_level, latent_dim*2) # the encoder output size is doubled to account for mu and logvar
            else:
                encoder = GATConv(input_dim_level, latent_dim*2) # the encoder output size is doubled to account for mu and logvar

            self.encoders.append(encoder)

            decoder = GCNConv(latent_dim, output_dim_level) # The decoder input is latent_dim now
            self.decoders.append(decoder)

            dropout = nn.Dropout(p=dropout_rate)
            self.dropout.append(dropout)

            batch_norm = nn.BatchNorm1d(latent_dim*2)
            self.batch_norms.append(batch_norm)

            pool = TopKPooling(latent_dim)
            self.pooling.append(pool)

    def forward(self, x, edge_index):
        outputs = []
        batch_indices = []
        kl_divergence = 0.0

        for encoder, decoder, dropout, batch_norm, pool in zip(
            self.encoders, self.decoders, self.dropout, self.batch_norms, self.pooling
        ):
            x = dropout(x)
            x = batch_norm(encoder(x, edge_index))
            mu, logvar = x.chunk(2, dim=-1) # splitting the tensor along the last dimension
            q_z = Normal(mu, logvar.mul(0.5).exp())

            z = q_z.rsample()
            kl_divergence += self.kl_divergence(mu, logvar)

            x, edge_index, _, batch, _, _ = pool(z, edge_index) # use the sampled z for pooling
            x = decoder(x, edge_index)

            outputs.append(x)
            batch_indices.append(batch)

        return outputs, batch_indices, kl_divergence

    def kl_divergence(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)


    def compute_loss(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        outputs, _, kl_divergence = self(x, edge_index)

        reconstruction_loss = sum(
            self.reconstruction_loss_function(level_output, x[:level_output.size(0)])
            for level_output in outputs
        )

        # multiply KL divergence with beta
        return reconstruction_loss + self.beta * kl_divergence

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("val_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)


def estimate_latent_dim(data_list, variance_threshold=0.95):
    """Estimates the latent dimension for the data."""
    node_features = torch.cat([data.x for data in data_list], dim=0).detach().numpy()
    pca = PCA()
    pca.fit(node_features)
    cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
    latent_dim = np.argmax(cumulative_explained_variance > variance_threshold)
    return max(latent_dim, 1)  # Ensure the latent dimension is at least 1


def load_data(data_dir):
    """Loads .pt data files from a directory."""
    data_list = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".pt"):
            file_path = os.path.join(data_dir, filename)
            data = torch.load(file_path)
            data_list.append(data)

    return data_list


random_seed = 42
random.seed(random_seed)
if torch.cuda.is_available():
    torch.manual_seed(random_seed)

input_dim = 3
variance_threshold = 0.95
num_levels = 3
coarse_grain_dims = [3, 3, 3]
batch_size = 32
dropout_rate = 0.5
num_epochs = 1000
learning_rate = 0.001
num_folds = 5

input_files_directory = 'C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//graph_data'

try:
    data_list = load_data(input_files_directory)
except Exception as e:
    print(f"Error: {e}")
    exit()

latent_dim = estimate_latent_dim(data_list, variance_threshold)

train_data = [data for i, data in enumerate(data_list) if i % num_folds != 0]
val_data = [data for i, data in enumerate(data_list) if i % num_folds == 0]

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)

model = HierarchicalCoarseGraining(input_dim, latent_dim, num_levels, coarse_grain_dims, dropout_rate, beta=5.0)


model.reconstruction_loss_function = nn.MSELoss()
model.learning_rate = learning_rate
model.weight_decay = 1e-5

trainer = pl.Trainer(max_epochs=num_epochs)
trainer.fit(model, train_loader, val_loader)

# Save the trained model
model_path = 'best_model.pt'
torch.save(model.state_dict(), model_path)

# Load the trained model
model = HierarchicalCoarseGraining(input_dim, latent_dim, num_levels, coarse_grain_dims, dropout_rate)
model.load_state_dict(torch.load(model_path))
model.eval()

# Load new protein data
new_data_path = 'C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//chi_graph'
new_data = torch.load(new_data_path)

# Apply the model to the new data
new_data_loader = DataLoader([new_data], batch_size=1)

new_coarse_grained_reps = []
new_reconstructed_reps = []

for data in new_data_loader:
    x, edge_index, batch = data.x, data.edge_index, data.batch
    outputs, batch_indices, _ = model(x, edge_index) # unpack all three returned values

    # Coarse-grained representations
    coarse_grained_reps = [
        Data(x=level_output, edge_index=edge_index, batch=batch_index)
        for level_output, batch_index in zip(outputs, batch_indices)
    ]
    new_coarse_grained_reps.append(coarse_grained_reps)

    # Reconstructed representations
    new_reconstructed_reps.append(model.decoders[-1](outputs[-1], edge_index))

# Save the coarse-grained representations and reconstructed representations
torch.save(new_coarse_grained_reps, 'new_protein_coarse_grained_reps.pt')
torch.save(new_reconstructed_reps, 'new_protein_reconstructed_reps.pt')
