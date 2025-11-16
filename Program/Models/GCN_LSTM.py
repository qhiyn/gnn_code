import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_max_pool
from torch_geometric.data import Batch

# Variabel ini akan diimpor oleh file loader
SEQUENCE_LENGTH = 20 

class GCN_LSTM(torch.nn.Module):
    def __init__(self, input_shape: int = 3, gcn_hidden_shape: int = 16, 
                 lstm_hidden_shape: int = 32, output_shape: int = 2):
        super().__init__()
        
        self.gcn_hidden_shape = gcn_hidden_shape
        self.lstm_hidden_shape = lstm_hidden_shape
        
        # --- 1. GCN Encoder (Spasial) ---
        self.gcn_conv1 = GCNConv(input_shape, gcn_hidden_shape)
        self.relu = nn.ReLU()
        
        # --- 2. LSTM Decoder (Temporal) ---
        self.lstm = nn.LSTM(
            input_size=gcn_hidden_shape,
            hidden_size=lstm_hidden_shape,
            num_layers=1,
            batch_first=True
        )
        
        # --- 3. Classifier ---
        self.clf = nn.Linear(in_features=lstm_hidden_shape, out_features=output_shape)
    
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        
        # x shape dari loader: [B, T, N, F]
        B, T, N, F = x.shape
        
        # Ubah (B, T, N, F) -> (B*T, N, F)
        x_flat = x.view(B * T, N, F)
        
        # Buat edge_index & batch index yang baru untuk GCN
        edge_index_list = [edge_index + i * N for i in range(B * T)]
        edge_index_batch = torch.cat(edge_index_list, dim=1).to(x.device)
        
        batch_idx_batch = torch.arange(B * T, device=x.device).view(-1, 1).repeat(1, N).flatten()

        # Ratakan 'x_flat' untuk GCN: (B*T, N, F) -> (B*T*N, F)
        x_gcn_input = x_flat.view(B * T * N, F)
        
        # --- 1. Jalankan GCN Encoder (Spasial) ---
        gcn_out = self.gcn_conv1(x_gcn_input, edge_index_batch)
        gcn_out = self.relu(gcn_out)
        
        # Global Pooling per frame -> [B*T, GCNHidden]
        graph_embeddings = global_max_pool(gcn_out, batch_idx_batch)
        
        # --- 2. Jalankan LSTM Decoder (Temporal) ---
        # Ubah shape -> [B, T, Features]
        sequence_input = graph_embeddings.view(B, T, self.gcn_hidden_shape)
        
        lstm_out, _ = self.lstm(sequence_input)
        
        # Ambil output dari time step TERAKHIR
        last_time_step_out = lstm_out[:, -1, :]
        
        # --- 3. Classifier ---
        output = self.clf(last_time_step_out)
        
        return output