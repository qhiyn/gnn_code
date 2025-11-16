# Program/Utils/getdata_sequence.py
import torch
import os
import pickle
from torch_geometric.data import Dataset as G_Dataset
from Program.Models.GCN_LSTM import SEQUENCE_LENGTH # Ambil dari file model

# --- PERUBAHAN DI SINI ---
# Path ke dataset .pkl baru yang kamu unggah
PROCESSED_DATA_PATH = "/kaggle/input/graph-nthu-ddd/Processed_Data"
# -------------------------

class getGraphDataSequence(G_Dataset):
    def __init__(self, state, transform=None, pre_transform=None, pre_filter=None):
        """
        Versi ini HANYA memuat file .pkl yang sudah diproses.
        processed_data_root: Path ke dataset Kaggle yang berisi file .pkl
        state: 'Training' atau 'Testing'
        """
        self.state = state
        self.processed_data_root = PROCESSED_DATA_PATH # Menggunakan path baru
        
        super().__init__(None, transform, pre_transform, pre_filter)
        
        # Tentukan path .pkl mana yang akan dimuat
        if self.state == 'Training':
            # Pastikan nama file ini SAMA PERSIS dengan yang kamu unggah
            path = os.path.join(self.processed_data_root, f'graphs_seq{SEQUENCE_LENGTH}_training.pkl')
        else:
            path = os.path.join(self.processed_data_root, f'graphs_seq{SEQUENCE_LENGTH}_testing.pkl')
            
        print(f"Loading PRE-PROCESSED sequence data from {path}...")
        try:
            with open(path, "rb") as f:
                self.sequences_data = pickle.load(f)
            print(f"Successfully loaded {len(self.sequences_data)} sequences for {self.state}.")
        except FileNotFoundError:
            print(f"FATAL ERROR: File {path} not found.")
            print(f"Pastikan kamu sudah menambahkan dataset di path yang benar dan nama file sesuai.")
            raise
        except Exception as e:
            print(f"Error loading processed file: {e}")
            raise

    def process(self):
        # Fungsi ini sengaja dikosongkan. Pemrosesan sudah dilakukan di lokal.
        pass

    def len(self):
        return len(self.sequences_data)
        
    def get(self, idx):
        return self.sequences_data[idx]