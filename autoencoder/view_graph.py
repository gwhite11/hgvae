import networkx as nx
import matplotlib.pyplot as plt
import torch


def visualize(data):
    G = nx.Graph()  # Create an empty NetworkX graph
    edge_index = data.edge_index.numpy()

    # Add edges to the graph
    for i in range(edge_index.shape[1]):
        G.add_edge(edge_index[0, i], edge_index[1, i])

    # Use the spring layout to position nodes
    pos = nx.spring_layout(G, seed=42)

    # Draw the graph
    nx.draw(G, pos, with_labels=True)
    plt.show()


# Load your graph
data = torch.load('C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//graph_data//1yu5_1.pdb.pt')

# Visualize it
visualize(data)
