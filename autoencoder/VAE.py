import torch
import torch.nn as nn
import random
import os
import numpy as np
from torch_geometric.nn import GCNConv, TopKPooling, GATConv, global_mean_pool
from torch_geometric.data import DataLoader

# Set the seeds
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Split the data into training, validation, and test sets
train_data = []
val_data = []
test_data = []

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
batch_size = 32

input_files_directory = 'C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//graph_data'
data_list = []
for filename in os.listdir(input_files_directory):
    if filename.endswith(".pt"):
        file_path = os.path.join(input_files_directory, filename)
        data = torch.load(file_path)
        data_list.append(data)

# Shuffle and split the data
random.shuffle(data_list)
num_examples = len(data_list)
train_cutoff = int(train_ratio * num_examples)
val_cutoff = train_cutoff + int(val_ratio * num_examples)

train_data = data_list[:train_cutoff]
val_data = data_list[train_cutoff:val_cutoff]
test_data = data_list[val_cutoff:]

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

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

    def encode(self, x, edge_index, batch):
        for encoder, dropout, pool in zip(self.encoders, self.dropout, self.pooling):
            x = dropout(x)
            x = encoder(x, edge_index)
            x, edge_index, _, batch, _, _ = pool(x, edge_index)
        return global_mean_pool(x, batch)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        for decoder in self.decoders[::-1]:
            z = decoder(z)
        return z

    def forward(self, x, edge_index, batch):
        z = self.encode(x, edge_index, batch)
        mu = self.mu_layer(z)
        logvar = self.logvar_layer(z)
        z = self.reparameterize(mu, logvar)
        z = self.latent_decoder(z)
        return self.decode(z), mu, logvar


def random_translate(data, translate_range):
    translation = torch.FloatTensor(data.pos.size()).uniform_(-translate_range, translate_range)
    data.pos += translation
    return data


input_dim = 3
hidden_dim = 128
latent_dim = 32  # Set the desired dimensionality of the latent space
num_levels = 3
coarse_grain_dims = [3, 3, 3]
dropout_rate = 0.5
num_epochs = 100
learning_rate = 0.001

# Create three VAE models for different levels of coarse-graining
model_coarse = VAE(input_dim, hidden_dim, latent_dim, 1, [coarse_grain_dims[0]], dropout_rate)
model_coarser = VAE(input_dim, hidden_dim, latent_dim, 2, coarse_grain_dims[:2], dropout_rate)
model_coarsest = VAE(input_dim, hidden_dim, latent_dim, 3, coarse_grain_dims, dropout_rate)

models = [model_coarse, model_coarser, model_coarsest]

reconstruction_loss_function = nn.MSELoss()

optimizers = [torch.optim.Adam(model.parameters(), lr=learning_rate) for model in models]

sample_data = data_list[0]  # Choose a sample data element from data_list
print(f"x shape: {sample_data.x.shape}")
print(f"edge_index shape: {sample_data.edge_index.shape}")


def loss_function(recon_x, x, mu, logvar):
    reconstruction_loss = reconstruction_loss_function(recon_x, x)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + kl_divergence


for epoch in range(num_epochs):
    for model, optimizer in zip(models, optimizers):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            x = batch.x
            edge_index = batch.edge_index
            batch_ids = batch.batch

            output, mu, logvar = model(x, edge_index, batch_ids)
            reconstruction_loss = loss_function(output, x, mu, logvar)

            optimizer.zero_grad()
            reconstruction_loss.backward()
            optimizer.step()

            total_loss += reconstruction_loss.item()

        average_loss = total_loss / len(train_loader)
        print(f"Epoch: {epoch + 1}/{num_epochs}, Average Loss: {average_loss}")


        with torch.no_grad():
            total_val_loss = 0.0
            for batch in val_loader:
                x = batch.x
                edge_index = batch.edge_index
                batch_ids = batch.batch

                output, mu, logvar = model(x, edge_index, batch_ids)
                val_loss = loss_function(output, x, mu, logvar)

                total_val_loss += val_loss.item() * x.size(0)

            avg_val_loss = total_val_loss / len(val_loader.dataset)

        print(f"Validation Loss: {avg_val_loss}")

# Define the testing function
def test():
    # not sure if I need a test here since there is not a 'ground truth' to test the CG reps against
    pass

# Call the test function after training
test()

# Generate CG reps for each model
coarse_grained_reps = []
for model in models:
    model.eval()
    with torch.no_grad():
        current_coarse_grained_rep = []

        for batch in train_loader:
            x = batch.x
            edge_index = batch.edge_index

            output, _, _ = model(x, edge_index)
            last_level_output = output

            if last_level_output.numel() == 0:
                continue

            batch_size = last_level_output.size(0)
            new_batch = torch.arange(batch_size).to(x.device)

            rep = global_mean_pool(last_level_output, new_batch)
            current_coarse_grained_rep.append(rep)

        coarse_grained_reps.append(current_coarse_grained_rep)

# Save the CG reps for each level
for level, reps in enumerate(coarse_grained_reps):
    torch.save(reps, f'coarse_grained_reps_level_{level+1}.pt')

# Save the trained models
model_paths = ['model_coarse.pt', 'model_coarser.pt', 'model_coarsest.pt']
for i, model in enumerate(models):
    torch.save(model.state_dict(), model_paths[i])

# Load the trained models
loaded_models = []
for path in model_paths:
    model = VAE(input_dim, hidden_dim, latent_dim, num_levels, coarse_grain_dims, dropout_rate)
    model.load_state_dict(torch.load(path))
    model.eval()
    loaded_models.append(model)

# Load the new protein graph data
new_protein_data = torch.load('new_protein.pt')

new_coarse_grained_reps = []
for model in loaded_models:
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


