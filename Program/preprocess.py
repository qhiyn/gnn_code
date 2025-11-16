# Program/preprocess.py

import torch
import numpy as np
import os
import pickle
import glob
import argparse  # <-- Tambahkan ini
from torch_geometric.data import Data
from Models.GCN_LSTM import SEQUENCE_LENGTH

# --- KONFIGURASI PATH LOKAL ANDA ---
# Ganti ini dengan path di komputermu
TRAINING_PATH_LOCAL = "/path/ke/folder/Datasets/Training" 
TESTING_PATH_LOCAL = "/path/ke/folder/Datasets/Testing"

# Ini adalah tempat .pkl akan disimpan
OUTPUT_DIR_LOCAL = "../Processed_Data"
# ------------------------------------

def get_static_edge_index(npz_files):
    """Membaca file npz pertama hanya untuk mendapatkan edge_index."""
    print("Membaca edge index statis...")
    try:
        data = np.load(npz_files[0], allow_pickle=True)
        edges = data['edges']
        static_edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        print(f"  > Loaded static edge index with shape: {static_edge_index.shape}")
        return static_edge_index
    except Exception as e:
        print(f"Tidak bisa memuat edge index dari {npz_files[0]}: {e}")
        return None

def process_frame_by_frame(npz_files, edge_index, state):
    """Memproses untuk GCN/GAT (frame-by-frame)."""
    print(f"\n--- Memulai Pemrosesan Frame-by-Frame untuk: {state} ---")
    all_graphs = []
    
    for i, npz_path in enumerate(npz_files):
        print(f"  Processing file {i+1}/{len(npz_files)}: {os.path.basename(npz_path)}")
        try:
            data = np.load(npz_path, allow_pickle=True)
            nodes_all_frames = data['nodes']
            labels_all_frames = data['y']
            mask_all_frames = data['frame_mask']
            
            num_frames = nodes_all_frames.shape[0]
            for i in range(num_frames):
                if mask_all_frames[i]:
                    node_features = torch.tensor(nodes_all_frames[i], dtype=torch.float32)
                    label_val = int(labels_all_frames[i])
                    label_tensor = torch.zeros((1, 2), dtype=torch.long)
                    label_tensor[0, label_val] = 1
                    
                    graph_data = Data(
                        feature_matrix=node_features,
                        edge_index=edge_index,
                        label=label_tensor
                    )
                    all_graphs.append(graph_data)
        except Exception as e:
            print(f"    ! Error processing file {npz_path}: {e}")

    save_path = os.path.join(OUTPUT_DIR_LOCAL, f'graphs_data_{state.lower()}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(all_graphs, f)
    print(f"  > Selesai. Menyimpan {len(all_graphs)} graf ke {save_path}")

def process_sequential(npz_files, edge_index, state):
    """Memproses untuk GCN_LSTM/GAT_LSTM (sekuensial)."""
    print(f"\n--- Memulai Pemrosesan Sekuensial (T={SEQUENCE_LENGTH}) untuk: {state} ---")
    all_sequences = []

    for i, npz_path in enumerate(npz_files):
        print(f"  Processing file {i+1}/{len(npz_files)}: {os.path.basename(npz_path)}")
        try:
            data = np.load(npz_path, allow_pickle=True)
            nodes_all_frames = data['nodes']
            labels_all_frames = data['y']
            mask_all_frames = data['frame_mask']
            
            num_frames = nodes_all_frames.shape[0]
            for i in range(num_frames - SEQUENCE_LENGTH + 1):
                end_index = i + SEQUENCE_LENGTH
                window_nodes = nodes_all_frames[i : end_index]
                window_mask = mask_all_frames[i : end_index]
                
                if not np.all(window_mask):
                    continue
                    
                label_val = int(labels_all_frames[end_index - 1])
                label_tensor = torch.tensor([label_val], dtype=torch.long)
                x_tensor = torch.tensor(window_nodes, dtype=torch.float32)
                
                sequence_data = Data(
                    x=x_tensor,
                    edge_index=edge_index,
                    y=label_tensor
                )
                all_sequences.append(sequence_data)
        except Exception as e:
            print(f"    ! Error processing file {npz_path}: {e}")

    save_path = os.path.join(OUTPUT_DIR_LOCAL, f'graphs_seq{SEQUENCE_LENGTH}_{state.lower()}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(all_sequences, f)
    print(f"  > Selesai. Menyimpan {len(all_sequences)} sekuens ke {save_path}")


def main():
    # --- Tambahkan Logika Argumen ---
    parser = argparse.ArgumentParser(description="Proses data NPZ menjadi file PKL")
    parser.add_argument('--state', type=str, required=True, choices=['Training', 'Testing'],
                        help="Tentukan 'Training' atau 'Testing'")
    parser.add_argument('--type', type=str, required=True, choices=['frame', 'sequence'],
                        help="Tentukan 'frame' (GCN/GAT) atau 'sequence' (LSTM)")
    args = parser.parse_args()
    
    # Pastikan folder output ada
    os.makedirs(OUTPUT_DIR_LOCAL, exist_ok=True)

    # Pilih path data berdasarkan argumen --state
    if args.state == 'Training':
        path_to_scan = TRAINING_PATH_LOCAL
    else:
        path_to_scan = TESTING_PATH_LOCAL

    npz_files = glob.glob(f"{path_to_scan}/**/*.npz", recursive=True)

    if not npz_files:
        print(f"Error: Tidak ada file .npz ditemukan di {path_to_scan}")
        return

    # Dapatkan edge index sekali saja
    static_edge_index = get_static_edge_index(npz_files)
    if static_edge_index is None:
        return

    # --- Jalankan hanya proses yang diminta ---
    if args.type == 'frame':
        process_frame_by_frame(npz_files, static_edge_index, args.state)
    elif args.type == 'sequence':
        process_sequential(npz_files, static_edge_index, args.state)
    
    print("\nPemrosesan selesai.")

if __name__ == "__main__":
    main()