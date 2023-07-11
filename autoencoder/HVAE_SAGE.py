import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data, DataLoader
import glob
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data.dataset import random_split
from torch_geometric.data import InMemoryDataset
from torch_geometric.nn import radius_graph


# Function to build a radius graph based on an input feature tensor and a distance threshold
def build_graph(x, distance_threshold=5.0):
    """
    Build a radius graph based on an input feature tensor and a distance threshold.
    Create an edge between two atoms if the distance between them is below the threshold.

    Args:
        x (torch.Tensor): Input feature tensor
        distance_threshold (float): Distance threshold for edge creation

    Returns:
        torch.Tensor: Edge index tensor
    """
    # Use the radius_graph function to create the edge_index tensor.
    edge_index = radius_graph(x, r=distance_threshold, batch=None, loop=False)

    return edge_index


# Function to standardize the feature data in the graph
def standardize_data(data):
    """
    Standardize the feature data in the graph.

    Args:
        data (torch_geometric.data.Data): Input graph data

    Returns:
        torch_geometric.data.Data: Standardized graph data
    """
    new_data = Data(x=(data.x - data.x.mean(dim=0)) / data.x.std(dim=0), edge_index=data.edge_index)
    return new_data

def load_data(graph_files):
    """
    Load the graph data from files, standardize the data, and build the graph.

    Args:
        graph_files (list): List of file paths to the graph data

    Returns:
        list: List of preprocessed graph data
    """
    data_list = []
    for graph_file in graph_files:
        # Load graph data
        data = torch.load(graph_file)
        # Standardize the data
        data = standardize_data(data)
        # Build graph
        new_data = Data(x=data.x, edge_index=build_graph(data.x))
        data_list.append(new_data)
    return data_list


# Load the graph data from files
graph_files = glob.glob(
    'C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//graph_data_test/*.pt')
data_list = load_data(graph_files)


# Function to estimate the number of principal components needed to explain a certain variance threshold
def estimate_latent_dim_from_graphs(graphs, var_threshold=0.95):
    """
    Estimate the number of principal components needed to explain var_threshold variance.

    Args:
        graphs (list[torch_geometric.data.Data]): List of graph data
        var_threshold (float): Minimum explained variance ratio

    Returns:
        int: Estimated number of principal components
    """
    # Concatenate all node features
    x = torch.cat([g.x for g in graphs], dim=0)

    # Center the data
    x = x - x.mean(dim=0)

    # Compute the covariance matrix
    cov_matrix = (x.t() @ x) / (x.size(0) - 1)

    # Perform Singular Value Decomposition (SVD)
    U, S, V = torch.svd(cov_matrix)

    # Compute the explained variance and find the smallest number of components
    # needed to explain at least var_threshold total variance.
    explained_variance = S / S.sum()
    cumulative_explained_variance = torch.cumsum(explained_variance, dim=0)
    latent_dim = torch.searchsorted(cumulative_explained_variance, var_threshold, right=True) + 1

    return latent_dim.item()


# Set hyperparameters
num_epochs = 100
learning_rate = 0.001

# Define input and hidden dimensions
input_dim = 3
hidden_dim = 64

# Estimate the latent dimension
latent_dim = estimate_latent_dim_from_graphs(data_list)

print('Estimated latent dimension:', latent_dim)


# Define the Variational Autoencoder (VAE) model
class VAE(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        self.encoder = torch.nn.Sequential(
            SAGEConv(input_dim, hidden_dim),
            torch.nn.ReLU(),
            SAGEConv(hidden_dim, latent_dim * 2)
        )

        self.decoder = torch.nn.Sequential(
            SAGEConv(latent_dim, hidden_dim),
            torch.nn.ReLU(),
            SAGEConv(hidden_dim, input_dim)
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
        return torch.optim.Adam(self.parameters(), lr=learning_rate)


# Initialize a list of VAE models
vae_models = [VAE(input_dim, hidden_dim, latent_dim) for _ in range(3)]


# Define a custom dataset class for protein graph data
class ProteinGraphDataset(InMemoryDataset):
    def __init__(self, root, data_list, transform=None, pre_transform=None):
        super(ProteinGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = self.collate(data_list)

    def get(self, idx):
        return self.data.__getitem__(idx)


# Create an InMemoryDataset from the list of graph data
dataset = ProteinGraphDataset(None, data_list)

# Define the split sizes for train, validation, and test sets
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

# Split the data into train, validation, and test sets
train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

batch_size = 16  # Choose your desired batch size

# Create data loaders for train, validation, and test sets
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# Create the trainer with the model checkpoint callback
checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min')
trainer = pl.Trainer(max_epochs=num_epochs, callbacks=[checkpoint_callback])

for i, model in enumerate(vae_models):
    # Train the model
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

            # If there's another level, update the loaders for the next level
            if i < len(vae_models) - 1:
                dataset = ProteinGraphDataset(None, data_list)
                train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])
                train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
                test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# Load the last trained model
trained_model = vae_models[-1]


# Function to predict and reconstruct a graph from a file using the trained VAE models
def predict_and_reconstruct_graph(filepath, vae_models, output_file):
    """
    Predict and reconstruct a graph from a file using the trained VAE models and save the results to an output file.

    Args:
        filepath (str): File path to the graph data
        vae_models (list): List of trained VAE models
        output_file (str): Output file path
    """
    data = torch.load(filepath)

    for i, model in enumerate(vae_models):
        model.eval()
        with torch.no_grad():
            mu, logvar = model.encode(data.x, data.edge_index)
            z = model.reparameterize(mu, logvar)
            coarse_grained_x = mu
            coarse_grained_edge_index = build_graph(coarse_grained_x)

            # Update the graph with the coarse-grained version
            data = Data(x=coarse_grained_x, edge_index=coarse_grained_edge_index)

            # Generate reconstructed graph
            reconstructed_x = model.decode(z, data.edge_index)
            reconstructed_edge_index = build_graph(reconstructed_x)

            # Save the reconstructed graph
            reconstructed_data = Data(x=reconstructed_x, edge_index=reconstructed_edge_index)

            # Save the graph as a .pt file with a different name for each model
            torch.save(reconstructed_data, f"{output_file}_model_{i}.pt")


reconstructed_data_list = []
for model_index in range(len(vae_models)):
    output_file = f"path/to/output/reconstructed_graph_model_{model_index}.pt"
    reconstructed_data = torch.load(output_file)
    reconstructed_data_list.append(reconstructed_data)

