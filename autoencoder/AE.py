import torch
import torch.nn as nn
import random
import os
from torch_geometric.nn import GCNConv, TopKPooling, GATConv, global_mean_pool
from torch_geometric.data import DataLoader

random_seed = 42
random.seed(random_seed)
torch.manual_seed(random_seed)

# This is the original code that I ran without problems (but without problems I mean it ended with code 0 rather than
# it worked as intended - I only had 32 graphs at the time.

class HierarchicalCoarseGraining(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_levels, coarse_grain_dims, dropout_rate=0.5):
        super(HierarchicalCoarseGraining, self).__init__()

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pooling = nn.ModuleList()
        self.dropout = nn.ModuleList()

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

    def forward(self, x, edge_index):
        outputs = []
        batch_indices = []

        for encoder, decoder, dropout, pool in zip(self.encoders, self.decoders, self.dropout, self.pooling):
            x = dropout(x)
            x = encoder(x, edge_index)
            x, edge_index, _, batch, _, _ = pool(x, edge_index)
            x = decoder(x, edge_index)

            outputs.append(x)
            batch_indices.append(batch)

        return outputs, batch_indices


def random_translate(data, translate_range):
    translation = torch.FloatTensor(data.pos.size()).uniform_(-translate_range, translate_range)
    data.pos += translation
    return data


input_dim = 3
hidden_dim = 128
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

model = HierarchicalCoarseGraining(input_dim, hidden_dim, num_levels, coarse_grain_dims, dropout_rate)

reconstruction_loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
sample_data = data_list[0]  # Choose a sample data element from data_list
print(f"x shape: {sample_data.x.shape}")
print(f"edge_index shape: {sample_data.edge_index.shape}")


for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        x = batch.x
        edge_index = batch.edge_index

        outputs, _ = model(x, edge_index)

        reconstruction_loss = 0.0
        for level_output in outputs:
            reconstruction_loss += reconstruction_loss_function(level_output, x[:level_output.size(0)])

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

            outputs, _ = model(x, edge_index)

            val_loss = 0.0
            for level_output in outputs:
                val_loss += reconstruction_loss_function(level_output, x[:level_output.size(0)])

            total_val_loss += val_loss.item() * x.size(0)

        avg_val_loss = total_val_loss / len(val_loader.dataset)

    print(f"Validation Loss: {avg_val_loss}")

dataloader = DataLoader(data_list, batch_size=batch_size)  # Create a DataLoader for all data


# Save the trained model
torch.save(model.state_dict(), 'best_model.pt')

# Load the trained model
model = HierarchicalCoarseGraining(input_dim, hidden_dim, num_levels, coarse_grain_dims, dropout_rate)
model.load_state_dict(torch.load('best_model.pt'))
model.eval()  # Set the model to evaluation mode


# Load the new protein data
new_data = torch.load('new_protein.pt')

# Apply the model to the new data
new_data_loader = DataLoader([new_data], batch_size=1)

new_coarse_grained_reps = []
for data in new_data_loader:
    x, edge_index, batch = data.x, data.edge_index, data.batch
    outputs, batch_indices = model(x, edge_index)
    last_level_output = outputs[-1]  # Get the output of the last level in the hierarchy

    if last_level_output.numel() == 0:
        # Skip pooling if the tensor or batch is empty
        continue

    # Create a new batch tensor with correct size
    batch_size = last_level_output.size(0)
    new_batch = torch.arange(batch_size).to(x.device)

    # Perform pooling if the tensor is not empty
    rep = global_mean_pool(last_level_output, new_batch)
    new_coarse_grained_reps.append(rep)

# Save the coarse-grained representations
torch.save(new_coarse_grained_reps, 'new_protein_coarse_grained_reps.pt')







