import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool

class GCN(torch.nn.Module):
    def __init__(self, input_shape: int = 3, hidden_shape: int = 16, output_shape: int = 2):
        super().__init__()
        
        self.conv1 = GCNConv(input_shape, hidden_shape)
        self.conv2 = GCNConv(hidden_shape, hidden_shape) # Membuatnya sedikit lebih 'deep'
        self.relu = nn.ReLU()
        self.clf = nn.Linear(in_features=hidden_shape, out_features=output_shape)
    
    def forward(self, feature_matrix, edge_index, batch):
        out = self.conv1(feature_matrix, edge_index)
        out = self.relu(out)
        out = self.conv2(out, edge_index)
        out = self.relu(out)
        out = global_max_pool(out, batch)
        out = self.clf(out)
        
        return out