import torch
import numpy as np
import os
import pickle
import glob
from torch_geometric.data import Dataset as G_Dataset
from torch_geometric.data import Data
from Program.Models.GCN_LSTM import SEQUENCE_LENGTH # Ambil dari file model

class getGraphDataSequence(G_Dataset):
    def __init__(self, data_path, state, transform=None, pre_transform=None, pre_filter=None):
        self.data_path_raw = data_path
        self.state = state
        self.writable_root = '/kaggle/working/'
        
        os.makedirs(os.path.join(self.writable_root, 'processed'), exist_ok=True)
        super().__init__(self.writable_root, transform, pre_transform, pre_filter)
        
        path = os.path.join(self.processed_dir, self.processed_file_names[0])
        print(f"Loading processed sequence data from {path}...")
        try:
            with open(path, "rb") as f:
                self.sequences_data = pickle.load(f)
            print(f"Successfully loaded {len(self.sequences_data)} sequences for {self.state}.")
        except FileNotFoundError:
            print(f"Error: Processed file not found at {path}.")
        except Exception as e:
            print(f"Error loading processed file: {e}")
            raise

    @property
    def raw_file_names(self):
        return [self.state]

    @property
    def processed_dir(self):
        return os.path.join(self.writable_root, 'processed')

    @property
    def processed_file_names(self):
        if self.state == 'Training':
            return [f'graphs_seq{SEQUENCE_LENGTH}_train.pkl']
        else:
            return [f'graphs_seq{SEQUENCE_LENGTH}_test.pkl']

    def process(self):
        """
        Membaca file .npz dan membuat 'sliding windows' dari frame.
        """
        print(f"Processed sequence file not found. Starting processing for '{self.state}' data...")
        
        all_sequences = []
        data_folder = self.data_path_raw
        
        npz_files = glob.glob(f"{data_folder}/**/*.npz", recursive=True)
        print(f"Found {len(npz_files)} .npz files in {data_folder}")

        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in {data_folder}.")

        static_edge_index = None
        file_counter = 0 # <-- TAMBAHAN: untuk melacak file

        for npz_path in npz_files:
            file_counter += 1
            print(f"\n--- Processing file {file_counter}/{len(npz_files)} ---")
            print(f"File: {npz_path}")
            
            try:
                data = np.load(npz_path, allow_pickle=True)
                
                nodes_all_frames = data['nodes']
                labels_all_frames = data['y']
                mask_all_frames = data['frame_mask']
                
                if static_edge_index is None:
                    edges = data['edges']
                    static_edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                    print(f"  > Loaded static edge index with shape: {static_edge_index.shape}")
                
                num_frames = nodes_all_frames.shape[0]
                num_sequences_in_file = 0 # <-- TAMBAHAN: melacak sekuens
                print(f"  > Found {num_frames} frames. Creating sliding windows...")

                # --- Logika Sliding Window ---
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
                        edge_index=static_edge_index,
                        y=label_tensor
                    )
                    all_sequences.append(sequence_data)
                    num_sequences_in_file += 1
                
                print(f"  > Done. Created {num_sequences_in_file} valid sequences from this file.")
                        
            except Exception as e:
                print(f"  ! Error processing file {npz_path}: {e}")
        
        # --- TAMBAHAN: Pesan sebelum menyimpan ---
        print("\n--------------------------------------------------")
        print(f"All {len(npz_files)} files processed.")
        print(f"Total sequences created: {len(all_sequences)}")
        print("Now saving to .pkl file. INI AKAN MEMAKAN WAKTU LAMA...")
        print("--------------------------------------------------")

        if self.state == 'Training':
            save_path = os.path.join(self.processed_dir, f'graphs_seq{SEQUENCE_LENGTH}_train.pkl')
        else:
            save_path = os.path.join(self.processed_dir, f'graphs_seq{SEQUENCE_LENGTH}_test.pkl')

        with open(save_path, 'wb') as f:
            pickle.dump(all_sequences, f)
            
        print(f"Processing complete. Saved {len(all_sequences)} sequences to {save_path}.")

    def len(self):
        return len(self.sequences_data)
        
    def get(self, idx):
        return self.sequences_data[idx]