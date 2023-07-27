import networkx as nx
import matplotlib.pyplot as plt
import torch
import os
from sklearn.decomposition import PCA

def visualize(data):
    G = nx.Graph()  # Create an empty NetworkX graph
    edge_index = data.edge_index.numpy()
    coordinates = data.x.numpy()  # Get the atom coordinates

    # Use PCA to reduce the dimension from 3 to 2
    pca = PCA(n_components=2)
    coordinates_2d = pca.fit_transform(coordinates)

    # Add nodes and edges to the graph
    for i in range(coordinates_2d.shape[0]):
        G.add_node(i, pos=coordinates_2d[i])

    for i in range(edge_index.shape[1]):
        G.add_edge(edge_index[0, i], edge_index[1, i])

    # Use the coordinates for positions
    pos = nx.get_node_attributes(G, 'pos')

    # Draw the graph
    nx.draw(G, pos, with_labels=True)
    plt.show()

# Define the file path
file_path = 'C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//graphs//1X7B_2.pdb.pt'

# Check if the file exists
if os.path.exists(file_path):
    # Load your graph
    data = torch.load(file_path)

    # Visualize it
    visualize(data)
else:
    print(f"The file {file_path} does not exist.")


