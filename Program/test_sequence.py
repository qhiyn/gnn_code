# Program/test_sequence.py
import os
import argparse
import torch
import numpy as np
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from Utils.getdata_sequence import getGraphDataSequence
from Models.GCN_LSTM import GCN_LSTM
from Models.GAT_LSTM import GAT_LSTM

# --- PERUBAHAN DI SINI ---
PROCESSED_DATA_PATH = "/kaggle/input/graph-nthu-ddd/Processed_Data"
OUTPUT_DIR = "/kaggle/working/Results"
# -------------------------

def test(model_name):
    torch.manual_seed(42)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"--- Starting SEQUENCE Testing for model: {model_name} ---")

    # --- Muat data .pkl yang sudah diproses ---
    dataset = getGraphDataSequence(
        state='Testing'
    )
    
    if len(dataset) == 0:
        print("Dataset is empty. Exiting.")
        return
        
    test_loader = DataLoader(dataset=dataset, batch_size=16, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model_path = os.path.join(OUTPUT_DIR, f'{model_name}.pth')
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}.")
        return
    
    model = None
    
    if model_name.upper() == 'GCN_LSTM':
        print("Initializing GCN_LSTM model...")
        model = GCN_LSTM(input_shape=3, gcn_hidden_shape=16, lstm_hidden_shape=32, output_shape=2).to(device)
    elif model_name.upper() == 'GAT_LSTM':
        print("Initializing GAT_LSTM model...")
        model = GAT_LSTM(input_shape=3, gat_hidden_shape=16, heads=4, lstm_hidden_shape=32, output_shape=2).to(device)
    else:
        print(f"Error: Model '{model_name}' not recognized.")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded.")

    correct, counter = 0, 0
    all_labels, all_preds = [], []

    model.eval()
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            
            out = model(data)
            out = torch.argmax(out, dim=1)
            all_preds.extend(out.cpu().tolist())

            decoded_label = data.y.squeeze()
            all_labels.extend(decoded_label.cpu().tolist())

            correct_batch = torch.sum(decoded_label == out)
            correct += correct_batch.item()
            counter += len(decoded_label)
     
    acc = correct / counter
    print(f"Test Accuracy : {acc:.4f}")
    
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)
    labels = ['nonsleepy', 'sleepy']

    print("\nClassification Report:")
    print(classification_report(all_labels_np, all_preds_np,
          target_names=labels, zero_division=0))

    cm = confusion_matrix(all_labels_np, all_preds_np)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues") 
    plt.title(f"{model_name} Confusion Matrix")

    plot_save_path = os.path.join(OUTPUT_DIR, f'{model_name}_confusion_matrix.png')
    plt.savefig(plot_save_path, dpi=300, bbox_inches="tight")

    plt.close()
    print(f"Confusion matrix saved to {plot_save_path}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test Sequence GNN model")
    parser.add_argument('--model', type=str, required=True, choices=['GCN_LSTM', 'GAT_LSTM'],
                        help="Model to test (GCN_LSTM or GAT_LSTM)")
    args = parser.parse_args()
    
    test(model_name=args.model)