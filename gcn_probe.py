from torch_geometric.nn import global_mean_pool
from torch.nn import Linear
import torch.nn.functional as F
import torch
import torch.nn
from torch_geometric.nn import GCNConv, SimpleConv


class GCNProbe(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv1 = GCNConv(in_features, in_features)
        self.lin1 = Linear(in_features, out_features)

    def forward(self, x, edge_index, batch=None, edge_weights=None):
        x = self.conv1(x, edge_index, edge_weights)
        if batch is not None:
            x = global_mean_pool(x, batch)
        x = self.lin1(x)
        return x
    
    
class GCNNonlinearProbe(torch.nn.Module):
    def __init__(self, in_features, out_features, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels, out_features)

    def forward(self, x, edge_index, batch=None, edge_weights=None):
        x = self.conv1(x, edge_index, edge_weights)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weights)
        if batch is not None:
            x = global_mean_pool(x, batch)
        x = self.lin1(x)
        return x


class MLPProbe(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.lin1 = Linear(in_features, out_features)

    def forward(self, x, batch):
        x = global_mean_pool(x, batch)
        x = self.lin1(x)
        return x
    