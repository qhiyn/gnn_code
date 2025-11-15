import torch
import numpy as np
import os
import pickle
import glob
from torch_geometric.data import Dataset as G_Dataset
from torch_geometric.data import Data

class getGraphData(G_Dataset):
    def __init__(self, data_path, state, transform=None, pre_transform=None, pre_filter=None):
        self.data_path_raw = data_path
        self.state = state
        self.writable_root = '/kaggle/working/'
        
        os.makedirs(os.path.join(self.writable_root, 'processed'), exist_ok=True)
        super().__init__(self.writable_root, transform, pre_transform, pre_filter)
        
        path = os.path.join(self.processed_dir, self.processed_file_names[0])
        print(f"Loading processed graph data from {path}...")
        try:
            with open(path, "rb") as f:
                self.graphs_data = pickle.load(f)
            print(f"Successfully loaded {len(self.graphs_data)} graphs for {self.state}.")
        except FileNotFoundError:
            print(f"Error: Processed file not found at {path}. 'process()' method will be called.")
            # process() akan dipanggil otomatis oleh superclass
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
            return ['graphs_data_train.pkl']
        else:
            return ['graphs_data_test.pkl']

    def process(self):
        print(f"Processed file not found. Starting processing for '{self.state}' data...")
        all_graphs = []
        data_folder = self.data_path_raw
        
        npz_files = glob.glob(f"{data_folder}/**/*.npz", recursive=True)
        print(f"Found {len(npz_files)} .npz files in {data_folder}")

        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in {data_folder}.")

        static_edge_index = None
        for npz_path in npz_files:
            print(f"Processing file: {npz_path}")
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
                for i in range(num_frames):
                    if mask_all_frames[i]:
                        node_features = torch.tensor(nodes_all_frames[i], dtype=torch.float32)
                        label_val = int(labels_all_frames[i])
                        label_tensor = torch.zeros((1, 2), dtype=torch.long)
                        label_tensor[0, label_val] = 1
                        
                        graph_data = Data(
                            feature_matrix=node_features,
                            edge_index=static_edge_index,
                            label=label_tensor
                        )
                        all_graphs.append(graph_data)
            except Exception as e:
                print(f"  ! Error processing file {npz_path}: {e}")
        
        save_path = os.path.join(self.processed_dir, self.processed_file_names[0])
        with open(save_path, 'wb') as f:
            pickle.dump(all_graphs, f)
        print(f"Processing complete. Saved {len(all_graphs)} graphs to {save_path}.")

    def len(self):
        return len(self.graphs_data)
        
    def get(self, idx):
        return self.graphs_data[idx]