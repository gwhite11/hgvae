import torch
import torch.nn as nn
import random
import os
import numpy as np
from torch_geometric.nn import GCNConv, TopKPooling, GATConv, global_mean_pool
from torch_geometric.data import DataLoader
from torch.utils.data.dataset import random_split

# I have tried to add in code that implements the training loop with early stopping - I am not sure if this is a good
# plan or not. I have also added in a bit that hopefully will normalise the data but please be aware I do not know if
# this is the correct way to do this for graph data - it is pretty basic. (I will sort it out)

# Set the seeds
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def normalize_data(x):
    mean = torch.mean(x, dim=0)
    std = torch.std(x, dim=0)
    x = (x - mean) / std
    return x


input_files_directory = 'C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//graph_data'
data_list = []

for filename in os.listdir(input_files_directory):
    if filename.endswith(".pt"):
        file_path = os.path.join(input_files_directory, filename)
        data = torch.load(file_path)
        data.x = normalize_data(data.x)  # Normalize the node features
        data_list.append(data)

# Shuffle and split the data
random.shuffle(data_list)
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

train_cutoff = int(train_ratio * len(data_list))
val_cutoff = int(val_ratio * len(data_list))
test_cutoff = len(data_list) - train_cutoff - val_cutoff

train_data, val_data, test_data = random_split(data_list, [train_cutoff, val_cutoff, test_cutoff])

batch_size = 32

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


input_dim = 3
hidden_dim = 128
latent_dim = 32
num_levels = 3
coarse_grain_dims = [3, 3, 3]
dropout_rate = 0.5
num_epochs = 100
learning_rate = 0.001
patience = 10

# Create three VAE models for different levels of coarse-graining
model_coarse = VAE(input_dim, hidden_dim, latent_dim, 1, [coarse_grain_dims[0]], dropout_rate)
model_coarser = VAE(input_dim, hidden_dim, latent_dim, 2, coarse_grain_dims[:2], dropout_rate)
model_coarsest = VAE(input_dim, hidden_dim, latent_dim, 3, coarse_grain_dims, dropout_rate)

models = {'model_coarse': model_coarse, 'model_coarser': model_coarser, 'model_coarsest': model_coarsest}

reconstruction_loss_function = nn.MSELoss()


sample_data = data_list[0]  # Choose a sample data element from data_list
print(f"x shape: {sample_data.x.shape}")
print(f"edge_index shape: {sample_data.edge_index.shape}")


def loss_function(recon_x, x, mu, logvar):
    reconstruction_loss = reconstruction_loss_function(recon_x, x)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + kl_divergence


# Early stopping addition
def early_stopping(val_loss_history, patience=patience):
    if len(val_loss_history) < patience + 1:
        return False
    return min(val_loss_history[-patience:]) <= val_loss_history[-1]


sample_data = data_list[0]  # Choose a sample data element from data_list
print(f"x shape: {sample_data.x.shape}")
print(f"edge_index shape: {sample_data.edge_index.shape}")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for name, model in models.items():  # Assign models to device
    model.to(device)

# Assign optimizers
optimizers = {name: torch.optim.Adam(model.parameters(), lr=learning_rate) for name, model in models.items()}

best_val_loss = {name: float('inf') for name in models}
val_loss_history = {name: [] for name in models}


# Learning rate scheduler addition (I am not sure if I really need this I just thought it might be a nice idea)
def adjust_learning_rate(optimizer, epoch, initial_lr, factor=0.1, patience=patience):
    if epoch % patience == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= factor


# Training
for epoch in range(num_epochs):
    for name, model in models.items():
        # Adjust learning rate for each optimizer
        adjust_learning_rate(optimizers[name], epoch, learning_rate)

        optimizer = optimizers[name]
        model.train()
        train_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data.x, data.edge_index, data.batch)
            loss = loss_function(recon_batch, data.x, mu, logvar)
            loss.backward()
            train_loss += loss
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader.dataset)
        print(f"====> Model: {name}, Epoch: {epoch}, Average training loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                recon, mu, logvar = model(data.x, data.edge_index, data.batch)
                val_loss += loss_function(recon, data.x, mu, logvar)

            avg_val_loss = val_loss / len(val_loader.dataset)
            print(f"====> Model: {name}, Epoch: {epoch}, Average validation loss: {avg_val_loss:.4f}")

            val_loss_history[name].append(avg_val_loss)

            if early_stopping(val_loss_history[name], patience):
                print(f"Early stopping triggered for model {name}.")
                break

            if avg_val_loss < best_val_loss[name]:
                best_val_loss[name] = avg_val_loss
                torch.save(model.state_dict(), f'{name}_best_model.pt')

# Loading the models
for name, model in models.items():
    if os.path.exists(f'{name}_best_model.pt'):
        model.load_state_dict(torch.load(f'{name}_best_model.pt'))
        model.eval()

# Generate coarse-grained representations for each model
coarse_grained_reps = []
for name, model in models.items():
    model.eval()
    with torch.no_grad():
        current_coarse_grained_rep = []

        for batch in train_loader:
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)

            output, _, _ = model(x, edge_index)
            last_level_output = output

            if last_level_output.numel() == 0:
                continue

            batch_size = last_level_output.size(0)
            new_batch = torch.arange(batch_size).to(x.device)

            rep = global_mean_pool(last_level_output, new_batch)
            current_coarse_grained_rep.append(rep)

        coarse_grained_reps.append(current_coarse_grained_rep)

# Save the coarse-grained representations for each level
for level, reps in enumerate(coarse_grained_reps):
    torch.save(reps, f'coarse_grained_reps_level_{level+1}.pt')

# Save the trained models
model_paths = ['model_coarse.pt', 'model_coarser.pt', 'model_coarsest.pt']
for i, (name, model) in enumerate(models.items()):
    torch.save(model.state_dict(), model_paths[i])

# Load the new protein graph data
new_protein_data = torch.load('new_protein.pt').to(device)

new_coarse_grained_reps = []
for name, model in models.items():
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
