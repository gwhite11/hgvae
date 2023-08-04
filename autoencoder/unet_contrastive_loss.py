import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Batch
from torch_geometric.nn import GraphSAGE, TopKPooling, JumpingKnowledge
from torch.nn import Linear
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from Bio.PDB import PDBIO, Chain, Residue, Atom, Model, Structure
import numpy as np
from torch_geometric.utils import add_self_loops


def contrastive_loss(z, pos_mask, neg_mask):
    sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    pos_sim = sim_matrix[pos_mask].mean()
    neg_sim = sim_matrix[neg_mask].mean()
    loss = -torch.log(pos_sim / neg_sim)
    return loss


def generate_masks(edge_index, num_nodes):
    pos_mask = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    neg_mask = torch.ones((num_nodes, num_nodes), dtype=torch.bool)

    edge_index_t = edge_index.t()
    pos_mask[edge_index_t[0], edge_index_t[1]] = True
    pos_mask[edge_index_t[1], edge_index_t[0]] = True  # In case of undirected graph

    neg_mask[edge_index_t[0], edge_index_t[1]] = False
    neg_mask[edge_index_t[1], edge_index_t[0]] = False  # In case of undirected graph

    # Remove self-connections
    idx = torch.arange(0, num_nodes, dtype=torch.long)
    neg_mask[idx, idx] = False

    return pos_mask, neg_mask


class GraphUNetWithSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, depth, pool_ratios):
        super(GraphUNetWithSAGE, self).__init__()

        self.depth = depth

        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        for i in range(depth):
            in_ch = in_channels if i == 0 else hidden_channels
            self.down_convs.append(GraphSAGE(in_ch, hidden_channels, num_layers=2))
            self.pools.append(TopKPooling(hidden_channels, ratio=pool_ratios[i]))

        self.up_convs = torch.nn.ModuleList()
        for i in range(depth - 1):
            self.up_convs.append(GraphSAGE(2 * hidden_channels, hidden_channels, num_layers=2))

        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear((depth + 1) * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        xs = []
        edge_indices = []
        batch_indices = []
        for i in range(self.depth):
            xs.append(x)
            edge_indices.append(edge_index)
            batch_indices.append(batch)
            print(f"i: {i}, x.shape: {x.shape}, edge_index.shape: {edge_index.shape}")

            x = self.down_convs[i](x, edge_index)
            x, edge_index, _, batch, _, _ = self.pools[i](x, edge_index, None, batch)

        for i in range(self.depth - 2, -1, -1):
            print(f"i: {i}, x.shape: {x.shape}")

            # Generate new edge_index
            row, col = torch.combinations(torch.arange(x.size(0), device=x.device), 2).t()
            edge_index = torch.stack([row, col], dim=0)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

            print(f"edge_index.shape: {edge_index.shape}")

            x = self.up_convs[i]((x, xs[i]), edge_index)


class CustomGraphDataset(Dataset):
    def __init__(self, data_folder, numerical_indices):
        self.data_file_list = [os.path.join(data_folder, filename) for filename in os.listdir(data_folder) if
                               filename.endswith('.pt')]
        self.numerical_indices = numerical_indices
        self.mean, self.std = self._compute_mean_std()

    def __len__(self):
        return len(self.data_file_list)

    def __getitem__(self, index):
        graph_data = torch.load(self.data_file_list[index])
        graph_data.x[:, self.numerical_indices] = (graph_data.x[:, self.numerical_indices] - self.mean) / self.std
        return graph_data

    def _compute_mean_std(self):
        all_data = [torch.load(file) for file in self.data_file_list]
        all_features = torch.cat([data.x[:, self.numerical_indices] for data in all_data], dim=0)
        return torch.mean(all_features, dim=0), torch.std(all_features, dim=0)


def collate_fn(batch):
    return Batch.from_data_list(batch)

if __name__ == '__main__':
    data_folder = "C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//graphs"
    numerical_indicies = [0, 1, 2]
    dataset = CustomGraphDataset(data_folder, numerical_indicies)

    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    batch_size = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0,
                              collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0,
                              collate_fn=collate_fn)

    in_channels = train_dataset[0].num_node_features
    hidden_channels = 420
    out_channels = 68
    depth = 3
    pool_ratios = [0.95, 0.85, 0.75]
    model = GraphUNetWithSAGE(in_channels, hidden_channels, out_channels, depth, pool_ratios)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(50):
        model.train()
        total_loss = 0
        for i, data in enumerate(train_loader):
            print(f"Batch {i}: num_nodes: {data.num_nodes}, edge_index.shape: {data.edge_index.shape}")
            optimizer.zero_grad()

            z = model(data)
            pos_mask, neg_mask = generate_masks(data.edge_index, data.num_nodes)

            loss = contrastive_loss(z, pos_mask, neg_mask)
            loss.backward()

            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch: {epoch}, Loss: {total_loss / len(train_loader)}")

    torch.save(model.state_dict(), 'graph_unet_model.pth')

    # Load the model parameters
    model.load_state_dict(torch.load('graph_unet_model.pth'))

    # Compute embeddings for all nodes in the new graph
    model.eval()
    with torch.no_grad():
        # new_data is your new graph data
        new_graph_data = torch.load(
            'C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//chi_graph//chig.pdb.pt')
        new_graph_data.x[:, numerical_indicies] = (new_graph_data.x[:, numerical_indicies] - dataset.mean) / dataset.std
        new_embeddings = model(new_graph_data).cpu().numpy()
        # Reconstruct the graph using the decoder

    # Define a range of possible cluster numbers
    min_clusters = 10
    max_clusters = min(50, new_embeddings.shape[0] - 1)

    best_score = -1
    best_n_clusters = 2

    for n_clusters in range(min_clusters, max_clusters + 1):
        spectral = SpectralClustering(n_clusters=n_clusters)
        labels = spectral.fit_predict(new_embeddings)
        score = silhouette_score(new_embeddings, labels)

        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters

    print("Best number of clusters:", best_n_clusters)

    spectral = SpectralClustering(n_clusters=best_n_clusters)
    labels = spectral.fit_predict(new_embeddings)

    print(labels)

    graph = to_networkx(new_graph_data)
    plt.figure(figsize=(7, 7))
    nx.draw_networkx(graph, cmap=plt.get_cmap('Set1'), node_color=labels, node_size=75, linewidths=6)
    plt.show()

def visualize_clusters(data, labels, original_pdb_path):
    from Bio.PDB import PDBParser

    # Parse the original PDB file
    parser = PDBParser()
    structure = parser.get_structure("original", original_pdb_path)

    # Extract the coordinates and identities of each atom
    atom_info = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom_info[atom.serial_number] = {'coord': atom.coord.tolist(),
                                                     'residue_name': residue.resname,
                                                     'atom_name': atom.name}

    # Create a color map from cluster labels
    cmap = [labels[node] for node in data.x]

    plt.figure(figsize=(8, 8))
    nx.draw(to_networkx(data), node_color=cmap, with_labels=True, cmap=plt.cm.tab10)
    plt.show()

    # Save the coarse grained graph to a PDB file
    coarse_grained_graph = Structure.Structure("H")
    model = Model.Model(0)  # add a model with model_id = 0
    chain = Chain.Chain("A")
    model.add(chain)
    coarse_grained_graph.add(model)

    # Create clusters of atoms (or 'beads')
    clusters = {}
    for node, label in enumerate(labels):
        if label in clusters:
            clusters[label].append(node)
        else:
            clusters[label] = [node]

    # Assign atom coordinates in each cluster by their centroid
    for label, nodes in clusters.items():
        # Use the coordinates from the original PDB file
        try:
            centroid = np.mean([atom_info[node + 1]['coord'] for node in nodes], axis=0)
        except KeyError as e:
            print(f'KeyError: {e} not found in atom_info')

        # Use the first atom's name and residue name as the name for the cluster
        first_atom_info = atom_info[nodes[0] + 1]
        residue = Residue.Residue((" ", label, " "), first_atom_info['residue_name'], " ")
        atom = Atom.Atom(first_atom_info['atom_name'], centroid.tolist(), 0, 0, " ",
                         first_atom_info['atom_name'], label, "C")
        residue.add(atom)
        chain.add(residue)

    io = PDBIO()
    io.set_structure(coarse_grained_graph)
    io.save("coarse_grained_graph.pdb")

# Call the function
visualize_clusters(new_graph_data, labels, 'C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//input_chig//chig.pdb')