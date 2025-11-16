# Program/train_sequence.py
import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt

from Utils.getdata_sequence import getGraphDataSequence
from Models.GCN_LSTM import GCN_LSTM
from Models.GAT_LSTM import GAT_LSTM

# --- PERUBAHAN DI SINI ---
# Path ke dataset .pkl baru yang kamu unggah
PROCESSED_DATA_PATH = "/kaggle/input/graph-nthu-ddd/Processed_Data"
OUTPUT_DIR = "/kaggle/working/Results"
# -------------------------

def train(model_name):
    torch.manual_seed(42)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"--- Starting SEQUENCE Training for model: {model_name} ---")
    
    # --- Muat data .pkl yang sudah diproses ---
    # Kita tidak lagi butuh data_path, cukup state
    dataset = getGraphDataSequence(
        state='Training'
    )
    
    if len(dataset) == 0:
        print("Dataset is empty. Exiting.")
        return
        
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    epochs = 100
    model = None

    if model_name.upper() == 'GCN_LSTM':
        print("Initializing GCN_LSTM model...")
        model = GCN_LSTM(input_shape=3, gcn_hidden_shape=16, lstm_hidden_shape=32, output_shape=2).to(device)
    elif model_name.upper() == 'GAT_LSTM':
        print("Initializing GAT_LSTM model...")
        model = GAT_LSTM(input_shape=3, gat_hidden_shape=16, heads=4, lstm_hidden_shape=32, output_shape=2).to(device)
    else:
        print(f"Error: Model '{model_name}' not recognized. Use 'GCN_LSTM' or 'GAT_LSTM'.")
        return
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3) 
    all_loss, all_acc = [], []
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        loss_per_epoch = 0
        correct, counter = 0, 0

        for data in train_loader:
            data = data.to(device)
            out = model(data)
            decoded_label = data.y.squeeze() # Squeeze from [B, 1] to [B]

            loss = loss_fn(out, decoded_label)
            loss_per_epoch += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            out = torch.argmax(out, dim=1)
            correct_batch = torch.sum(decoded_label == out)
            correct += correct_batch.item()
            counter += len(decoded_label)
            
        loss_per_epoch = loss_per_epoch / len(train_loader)
        acc = correct / counter
        all_loss.append(loss_per_epoch)
        all_acc.append(acc) 

        print(f"EPOCH - {epoch+1:03d} --> Train Loss: {loss_per_epoch:.4f} | Train Acc: {acc:.4f}")

    print("Training finished.")

    model_save_path = os.path.join(OUTPUT_DIR, f"{model_name}.pth")
    plot_save_path = os.path.join(OUTPUT_DIR, f"{model_name}_training_performance.png")

    torch.save(model.state_dict(), model_save_path)
    print(f"Model state_dict saved to {model_save_path}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(all_loss, label='loss')
    ax.plot(all_acc, label='acc')
    plt.legend(loc="upper right")
    plt.title(f"{model_name} Training Loss & Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Loss / Accuracy')
    plt.savefig(plot_save_path, dpi=300)
    print(f"Training plot saved to {plot_save_path}")
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Sequence GNN model")
    parser.add_argument('--model', type=str, required=True, choices=['GCN_LSTM', 'GAT_LSTM'],
                        help="Model to train (GCN_LSTM or GAT_LSTM)")
    args = parser.parse_args()
    train(model_name=args.model)