import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_max_pool
from torch_geometric.data import Batch

SEQUENCE_LENGTH = 20 

class GAT_LSTM(torch.nn.Module):
    def __init__(self, input_shape: int = 3, gat_hidden_shape: int = 16, 
                 heads: int = 4, lstm_hidden_shape: int = 32, output_shape: int = 2):
        super().__init__()
        
        self.gat_hidden_shape = gat_hidden_shape
        self.heads = heads
        self.lstm_hidden_shape = lstm_hidden_shape
        
        # --- 1. GAT Encoder (Spasial) ---
        self.gat_conv1 = GATConv(input_shape, gat_hidden_shape, heads=heads)
        self.gat_output_dim = gat_hidden_shape * heads
        self.relu = nn.ReLU()
        
        # --- 2. LSTM Decoder (Temporal) ---
        self.lstm = nn.LSTM(
            input_size=self.gat_output_dim,
            hidden_size=lstm_hidden_shape,
            num_layers=1,
            batch_first=True
        )
        
        # --- 3. Classifier ---
        self.clf = nn.Linear(in_features=lstm_hidden_shape, out_features=output_shape)
    
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        
        B, T, N, F = x.shape
        
        x_flat = x.view(B * T, N, F)
        
        edge_index_list = [edge_index + i * N for i in range(B * T)]
        edge_index_batch = torch.cat(edge_index_list, dim=1).to(x.device)
        
        batch_idx_batch = torch.arange(B * T, device=x.device).view(-1, 1).repeat(1, N).flatten()

        x_gat_input = x_flat.view(B * T * N, F)
        
        # --- 1. Jalankan GAT Encoder (Spasial) ---
        gat_out = self.gat_conv1(x_gat_input, edge_index_batch)
        gat_out = self.relu(gat_out)
        
        # Global Pooling per frame -> [B*T, GATHidden * Heads]
        graph_embeddings = global_max_pool(gat_out, batch_idx_batch)
        
        # --- 2. Jalankan LSTM Decoder (Temporal) ---
        # Ubah shape -> [B, T, Features]
        sequence_input = graph_embeddings.view(B, T, self.gat_output_dim)
        
        lstm_out, _ = self.lstm(sequence_input)
        
        # Ambil output dari time step TERAKHIR
        last_time_step_out = lstm_out[:, -1, :]
        
        # --- 3. Classifier ---
        output = self.clf(last_time_step_out)
        
        return output