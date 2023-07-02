import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.utils import to_dense_adj
import os
import glob


class GraphVariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(GraphVariationalAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            GCNConv(input_dim, hidden_dim),
            nn.ReLU(),
            GCNConv(hidden_dim, hidden_dim),
            nn.ReLU(),
            GCNConv(hidden_dim, latent_dim * 2)
        )

        self.decoder = nn.Sequential(
            GCNConv(latent_dim, hidden_dim),
            nn.ReLU(),
            GCNConv(hidden_dim, hidden_dim),
            nn.ReLU(),
            GCNConv(hidden_dim, input_dim)
        )

        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)

    def encode(self, x, edge_index):
        z = self.encoder(x, edge_index)
        mu, logvar = self.fc_mu(z), self.fc_logvar(z)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z, edge_index):
        return self.decoder(z, edge_index)

    def forward(self, x, edge_index):
        mu, logvar = self.encode(x, edge_index)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, edge_index), mu, logvar


def train_vae(model, train_loader, optimizer, device):
    model.train()
    loss_train = 0.0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data.x, data.edge_index)
        loss = loss_function(recon_batch, data.x, mu, logvar)
        loss.backward()
        loss_train += loss.item() * data.num_nodes
        optimizer.step()

    return loss_train / len(train_loader.dataset)


def loss_function(recon_x, x, mu, logvar):
    mse_loss = nn.MSELoss(reduction='mean')
    mse = mse_loss(recon_x, x)

    kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return mse + kld_loss


def generate_coarse_grained_representation(model, data, device):
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        _, mu, _ = model(data.x, data.edge_index)
        return mu


# Specify the path to your directory with graph data
graph_directory = 'C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//chi_graph'

# Get all graph data files in the directory
graph_files = glob.glob(os.path.join(graph_directory, '*.pt'))

# Hyperparameters
input_dim = 3  # Dimension of input features (e.g., atom coordinates)
hidden_dim = 64  # Dimension of hidden layers in the VAE
latent_dim = 16  # Dimension of the latent space
batch_size = 32
num_epochs = 100
learning_rate = 0.001

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create VAE model
model = GraphVariationalAutoencoder(input_dim, hidden_dim, latent_dim).to(device)

# Create optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=10, gamma=0.9)

# Create DataLoader
train_loader = DataLoader(graph_files, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    loss_train = train_vae(model, train_loader, optimizer, device)
    scheduler.step()
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss_train:.4f}')

# Generate coarse-grained representations for a specific graph data file
data = torch.load(graph_files[0])  # Replace with the graph data file you want to generate representations for
data = data.to(device)
coarse_grained_reps = generate_coarse_grained_representation(model, data, device)
print(f'Coarse-Grained Representations: {coarse_grained_reps}')
