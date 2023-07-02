#cannot get the maths to add up on this one - it does not work. Only the AE models will work so far

import torch
import torch.nn as nn
import random
import os
import numpy as np
from torch_geometric.nn import GCNConv, TopKPooling, GATConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected
from sklearn.decomposition import PCA

# Set the seeds
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def estimate_latent_dim(data_list, variance_threshold=0.95):
    """
    Estimates the innermost dimension of VAE using PCA.
    """
    # Convert the data to a suitable format
    node_features = torch.cat([data.x for data in data_list], dim=0).numpy()

    # Use PCA on data
    pca = PCA()
    pca.fit(node_features)

    # Calculate cumulative explained variance and find the number of components for a certain threshold
    cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
    latent_dim = np.argmax(cumulative_explained_variance > variance_threshold)

    return latent_dim


def normalize_data(x):
    mean = torch.mean(x, dim=0)
    std = torch.std(x, dim=0)
    x = (x - mean) / std
    return x


def clip_edge_indices(edge_index, max_index):
    edge_index = torch.clamp(edge_index, max=max_index)
    return edge_index


input_files_directory = 'C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//graph_data'
data_list = []

for filename in os.listdir(input_files_directory):
    if filename.endswith(".pt"):
        file_path = os.path.join(input_files_directory, filename)
        data = torch.load(file_path)
        data.x = normalize_data(data.x)  # Normalize the node features
        undirected_edge_index = to_undirected(data.edge_index)
        data_list.append(data)

# Shuffle and split the data
random.shuffle(data_list)
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

train_cutoff = int(train_ratio * len(data_list))
val_cutoff = int(val_ratio * len(data_list))
test_cutoff = len(data_list) - train_cutoff - val_cutoff

train_data, val_data, test_data = data_list[:train_cutoff], data_list[train_cutoff:train_cutoff + val_cutoff], data_list[-test_cutoff:]

batch_size = 32

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

# Determine the maximum node index
max_node_index = 0
for data in data_list:
    max_node_index = max(max_node_index, torch.max(data.edge_index[0]), torch.max(data.edge_index[1]))
print("Max node index:", max_node_index)

input_dim = max_node_index + 1

# Define the model
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_levels, coarse_grain_dims, dropout_rate=0.5):
        super(VAE, self).__init__()

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pooling = nn.ModuleList()
        self.dropout = nn.ModuleList()

        self.latent_dim = latent_dim

        for level in range(num_levels):
            input_dim_level = input_dim if level == 0 else coarse_grain_dims[level - 1]
            output_dim_level = coarse_grain_dims[level]

            if level == 0:
                encoder = GCNConv(input_dim_level, hidden_dim)
            else:
                encoder = GATConv(input_dim_level, hidden_dim)

            self.encoders.append(encoder)

            decoder = GCNConv(hidden_dim, output_dim_level)
            self.decoders.append(decoder)

            dropout = nn.Dropout(p=dropout_rate)
            self.dropout.append(dropout)

            pool = TopKPooling(hidden_dim)
            self.pooling.append(pool)

        self.mu_layer = nn.Linear(coarse_grain_dims[-1], latent_dim)
        self.logvar_layer = nn.Linear(coarse_grain_dims[-1], latent_dim)
        self.latent_decoder = nn.Linear(latent_dim, coarse_grain_dims[-1])

    def encode(self, x, edge_index, undirected_edge_index, batch):
        edge_attr = None  # Initialize edge attributes as None
        for encoder, dropout, pool in zip(self.encoders, self.dropout, self.pooling):
            x = dropout(x)
            x = encoder(x, edge_index, edge_attr)  # Pass edge_attr to the encoder
            x, undirected_edge_index, edge_attr, batch, _, _ = pool(x, undirected_edge_index, edge_attr, batch)
            if undirected_edge_index.numel() == 0:
                return x, undirected_edge_index, batch
        return x, undirected_edge_index, batch

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, edge_index):
        if edge_index.numel() == 0:
            return z
        edge_index = torch.cat([edge_index, torch.arange(z.size(0)).unsqueeze(0).repeat(2, 1)], dim=1)
        for decoder in self.decoders[::-1]:
            z = decoder(z, edge_index)
            if z is None:
                return z
        return z

    def forward(self, x, edge_index, undirected_edge_index, batch):
        # Filter out-of-range indices
        valid_indices = (edge_index < x.size(0)).all(dim=0)
        edge_index = edge_index[:, valid_indices]

        z, undirected_edge_index, _ = self.encode(x, edge_index, undirected_edge_index, batch)
        if undirected_edge_index.numel() == 0:
            return z, None, None  # Return appropriate values when undirected_edge_index is empty

        mu = self.mu_layer(z)
        logvar = self.logvar_layer(z)
        z = self.reparameterize(mu, logvar)
        z = self.latent_decoder(z)

        return self.decode(z, edge_index), mu, logvar


input_dim = 3
hidden_dim = 128
latent_dim = estimate_latent_dim(data_list, variance_threshold=0.95)
num_levels = 3
# coarse_grain_dims = [hidden_dim, hidden_dim, input_dim]  # the last element should match input_dim
dropout_rate = 0.5

# Define the output dimensions for each level
coarse_grain_dims = [128, 64, 32]  # adjust according to your needs

# Create the VAE model
model = VAE(input_dim, hidden_dim, latent_dim, num_levels, coarse_grain_dims, dropout_rate)

# Move the model to the appropriate device
model = model.to(device)

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()

    for data in train_loader:
        data = data.to(device)
        if data.edge_index.numel() == 0:
            continue  # Skip empty edge_index

        # Clip edge indices to avoid out-of-bounds error
        edge_index = clip_edge_indices(data.edge_index, max_node_index)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data.x, edge_index, edge_index, data.batch)  # Pass 'batch' argument here

        # Skip if recon_batch is None
        if recon_batch is None:
            continue

        loss = model.loss_function(recon_batch, data.x, mu, logvar)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for data in val_loader:
            data = data.to(device)
            if data.edge_index.numel() == 0:
                continue  # Skip empty edge_index

            # Clip edge indices to avoid out-of-bounds error
            edge_index = clip_edge_indices(data.edge_index, max_node_index)

            recon_batch, mu, logvar = model(data.x, edge_index, edge_index, data.batch)  # Pass 'batch' argument here

            # Skip if recon_batch is None
            if recon_batch is None:
                continue

            val_loss += model.loss_function(recon_batch, data.x, mu, logvar)

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs} | Validation Loss: {avg_val_loss}")

# Save the trained model
torch.save(model.state_dict(), 'coarse_graining_model.pt')

# Load the new protein graph data
new_protein_data = torch.load('new_protein.pt').to(device)

new_coarse_grained_reps = []
model.eval()
with torch.no_grad():
    x = new_protein_data.x
    edge_index = new_protein_data.edge_index

    output, _, _ = model(x, edge_index)
    last_level_output = output

    if last_level_output.numel() != 0:
        batch_size = last_level_output.size(0)
        new_batch = torch.arange(batch_size).to(x.device)
        rep = global_mean_pool(last_level_output, new_batch)
        new_coarse_grained_reps.append(rep)

# Save the new coarse-grained representations
torch.save(new_coarse_grained_reps, 'new_protein_coarse_grained_reps.pt')
