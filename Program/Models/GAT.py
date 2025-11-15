import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool

class GAT(torch.nn.Module):
    def __init__(self, input_shape: int = 3, hidden_shape: int = 16, output_shape: int = 2, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.hidden_shape = hidden_shape
        
        self.conv1 = GATConv(input_shape, hidden_shape, heads=self.heads)
        self.relu = nn.ReLU()
        
        # Classifier (Inputnya adalah hidden_shape * heads)
        self.clf = nn.Linear(in_features=hidden_shape * self.heads, out_features=output_shape)
    
    def forward(self, feature_matrix, edge_index, batch):
        out = self.conv1(feature_matrix, edge_index)
        out = self.relu(out)
        out = global_max_pool(out, batch)
        out = self.clf(out)
        
        return out