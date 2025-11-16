import os
import argparse
import torch
import numpy as np
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from Program.Utils.getdata import getGraphData
from Program.Models.GCN import GCN 
from Program.Models.GAT import GAT 

TESTING_PATH = "/kaggle/input/nthu-ddd-graph/NTHU DDD GRAPH/Testing"
OUTPUT_DIR = "/kaggle/working/Results"

def test(model_name):
    torch.manual_seed(42)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"--- Starting Testing for model: {model_name} ---")
    
    dataset = getGraphData(data_path=TESTING_PATH, state='Testing')
    
    if len(dataset) == 0:
        print("Dataset is empty. Exiting.")
        return
        
    test_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model_path = os.path.join(OUTPUT_DIR, f'{model_name}.pth')
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}.")
        return
    
    model = None
    if model_name.upper() == 'GCN':
        model = GCN(input_shape=3, hidden_shape=16, output_shape=2).to(device)
    elif model_name.upper() == 'GAT':
        model = GAT(input_shape=3, hidden_shape=16, output_shape=2, heads=4).to(device)
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
            out = model(data.feature_matrix, data.edge_index, data.batch)
            out = torch.argmax(out, dim=1)
            all_preds.extend(out.cpu().tolist())
            decoded_label = torch.argmax(data.label, dim=1)
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
    parser = argparse.ArgumentParser(description="Test GNN model")
    parser.add_argument('--model', type=str, required=True, choices=['GCN', 'GAT'],
                        help="Model to test (GCN or GAT)")
    args = parser.parse_args()
    test(model_name=args.model)