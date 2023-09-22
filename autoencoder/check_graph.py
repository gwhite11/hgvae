import torch


def check_graph_data(data):
    # Check the shape of node features
    print("Node Features (x) Shape:", data.x.shape)

    # Print the features of the first 5 nodes
    print("Features of the first 5 nodes:")
    print(data.x[:50])

    # Check the shape of edge connections
    print("Edge Connections (edge_index) Shape:", data.edge_index.shape)

    # Check the number of nodes and edges
    print("Number of Nodes:", data.num_nodes)
    print("Number of Edges:", data.num_edges)

    # Print the adjacency matrix
    adj_matrix = torch.zeros((data.num_nodes, data.num_nodes))
    adj_matrix[data.edge_index[0], data.edge_index[1]] = 1
    print("Adjacency Matrix:")
    print(adj_matrix)



data = torch.load('C://Users//gemma//PycharmProjects//pythonProject1//autoencoder//pdb_files//graphs_structural_new//PED00001e001.pdb.pt')
check_graph_data(data)
