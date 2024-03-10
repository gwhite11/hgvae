import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGPooling
import torch_geometric


class PoolGraph(torch.nn.Module):
    def __init__(self, num_node_features):
        super().__init__()
        self.pool1 = SAGPooling(num_node_features)

    def forward(self, x, edge_index, batch):
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, batch=batch)
        return torch_geometric.data.Data(x=x, edge_index=edge_index, batch=batch)


def unpool(edge_index, perm):
    # Reverse the permutation
    rev_perm = torch.argsort(perm)

    # This will use the reversed permutation to map the pooled edge indices back to
    # their positions in the unpooled graph - for both rows of edge_index.
    row_0 = torch.index_select(rev_perm, 0, edge_index[0])
    row_1 = torch.index_select(rev_perm, 0, edge_index[1])

    return torch.stack([row_0, row_1], dim=0)


class PoolUnpoolGraph(torch.nn.Module):
    def __init__(self, num_node_features):
        super().__init__()
        self.pool1 = SAGPooling(num_node_features)

    def forward(self, x, edge_index, batch):
        x_cg, edge_index_cg, _, _, perm, _ = self.pool1(x, edge_index, batch=batch)

        new_batch = batch[perm]
        x_unpooled = x[perm]
        edge_index_unpooled = unpool(edge_index_cg, perm)

        return x_unpooled, x_cg, edge_index_unpooled, new_batch


class GraphUNet(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, depth):
        super(GraphUNet, self).__init__()
        self.depth = depth

        # Down part
        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.down_convs.append(GCNConv(input_channels, hidden_channels))
        for i in range(depth - 1):
            self.pools.append(PoolGraph(hidden_channels))
            self.down_convs.append(GCNConv(hidden_channels, hidden_channels))

        # Up part
        self.up_convs = torch.nn.ModuleList()
        self.unpools = torch.nn.ModuleList()
        for i in range(depth - 1):
            self.unpools.append(PoolUnpoolGraph(hidden_channels))
            self.up_convs.append(GCNConv(2 * hidden_channels, hidden_channels))
        self.up_convs.append(GCNConv(hidden_channels, output_channels))

    def forward(self, x, edge_index):
        skip_connections = []
        batch = torch.zeros(x.size(0), dtype=torch.long)

        for i in range(self.depth):
            x = F.relu(self.down_convs[i](x, edge_index))
            skip_connections.append(x)
            print(f"After down_conv {i}, x size: {x.size()}")
            if i != self.depth - 1:
                pool_data = self.pools[i](x, edge_index, batch)
                x, edge_index = pool_data.x, pool_data.edge_index
                batch = pool_data.batch
                print(f"After pooling {i}, x size: {x.size()}")

        for i in range(self.depth - 1):
            unpool_data = self.unpools[i](x, edge_index, batch)
            x, edge_index, batch = unpool_data[0], unpool_data[2], unpool_data[3]
            skip = skip_connections[-(i + 1)]
            print(f"Upsampled x size: {x.size()}, Skip size: {skip.size()}")
            if x.size(0) != skip.size(0):
                skip = skip[:x.size(0)]
            x = torch.cat([x, skip], dim=1)
            x = F.relu(self.up_convs[i](x, edge_index))

        return x


# To check it is working:
model = GraphUNet(input_channels=1, hidden_channels=64, output_channels=2, depth=4)
x = torch.randn((100, 1))
edge_index = torch.randint(100, (2, 200), dtype=torch.long)
out = model(x, edge_index)
print(out.shape)