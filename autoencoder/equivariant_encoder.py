import torch
import torch.nn as nn
import random
import os
from torch_geometric.nn import GATConv, DeepGCNLayer, global_mean_pool
from torch_geometric.data import DataLoader

random_seed = 42
random.seed(random_seed)
torch.manual_seed(random_seed)


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_levels, coarse_grain_dims, dropout_rate=0.5):
        super(VAE, self).__init__()

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        self.latent_dim = latent_dim

        for level in range(num_levels):
            input_dim_level = input_dim if level == 0 else coarse_grain_dims[level - 1]
            output_dim_level = coarse_grain_dims[level]

            if level == 0:
                encoder = GATConv(input_dim_level, hidden_dim)
            else:
                encoder = DeepGCNLayer(input_dim_level, hidden_dim)

            self.encoders.append(encoder)

            decoder = GATConv(hidden_dim, output_dim_level)
            self.decoders.append(decoder)

        self.mu_layer = nn.Linear(coarse_grain_dims[-1], latent_dim)
        self.logvar_layer = nn.Linear(coarse_grain_dims[-1], latent_dim)
        self.latent_decoder = nn.Linear(latent_dim, coarse_grain_dims[-1])

    def encode(self, x, edge_index):
        for encoder in self.encoders:
            x = encoder(x, edge_index)
        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        for decoder in self.decoders[::-1]:
            z = decoder(z, None)
        return z

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        mu = self.mu_layer(z)
        logvar = self.logvar_layer(z)
        z = self.reparameterize(mu, logvar)
        z = self.latent_decoder(z)
        out = self.decode(z)
        return out, mu, logvar


def random_translate(data, translate_range):
    translation = torch.FloatTensor(data.pos.size()).uniform_(-translate_range, translate_range)
    data.pos += translation
    return data


input_dim = 3
hidden_dim = 128
latent_dim = 32  # Set the desired dimensionality of the latent space
num_levels = 3
coarse_grain_dims = [3, 3, 3]
batch_size = 32
dropout_rate = 0.5
num_epochs = 100
learning_rate = 0.001
num_folds = 5
translate_range = 0.1  # Adjust the translation range as per your requirements

input_files_directory = 'C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//graph_data'
data_list = []
for filename in os.listdir(input_files_directory):
    if filename.endswith(".pt"):
        file_path = os.path.join(input_files_directory, filename)
        data = torch.load(file_path)
        data_list.append(data)

train_data = []
val_data = []
for i, data in enumerate(data_list):
    if i % num_folds == 0:
        val_data.append(data)
    else:
        train_data.append(data)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)

model = VAE(input_dim, hidden_dim, latent_dim, num_levels, coarse_grain_dims, dropout_rate)

reconstruction_loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
sample_data = data_list[0]  # Choose a sample data element from data_list
print(f"x shape: {sample_data.x.shape}")
print(f"edge_index shape: {sample_data.edge_index.shape}")


def loss_function(recon_x, x, mu, logvar):
    reconstruction_loss = reconstruction_loss_function(recon_x, x)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + kl_divergence


for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        x = batch.x
        edge_index = batch.edge_index

        outputs, mu, logvar = model(x, edge_index)
        reconstruction_loss = loss_function(outputs[-1], x, mu, logvar)

        optimizer.zero_grad()
        reconstruction_loss.backward()
        optimizer.step()

        total_loss += reconstruction_loss.item() * x.size(0)

    average_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch: {epoch + 1}/{num_epochs}, Average Loss: {average_loss}")

    model.eval()
    with torch.no_grad():
        total_val_loss = 0.0
        for batch in val_loader:
            x = batch.x
            edge_index = batch.edge_index

            outputs, mu, logvar = model(x, edge_index)
            val_loss = loss_function(outputs[-1], x, mu, logvar)

            total_val_loss += val_loss.item() * x.size(0)

        avg_val_loss = total_val_loss / len(val_loader.dataset)

    print(f"Validation Loss: {avg_val_loss}")

# Generate multiple levels of coarse-grained representations
coarse_grained_reps = []
for level in range(num_levels):
    current_coarse_grain_dims = coarse_grain_dims[:level+1]

    model = VAE(input_dim, hidden_dim, latent_dim, level+1, current_coarse_grain_dims, dropout_rate)
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()

    current_loader = DataLoader(data_list, batch_size=batch_size)
    current_coarse_grained_rep = []

    with torch.no_grad():
        for batch in current_loader:
            x = batch.x
            edge_index = batch.edge_index

            outputs, _, _ = model(x, edge_index)
            last_level_output = outputs[-1]

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
