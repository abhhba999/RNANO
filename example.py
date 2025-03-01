"""
Example script for using RNANO to detect RNA modifications
"""

import os
import argparse
import torch
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import Model
from dataset import NpDataset
from utils import load_or_create_normal_data, plot_roc_curve, plot_pr_curve, get_metrics

def run_example(args):
    """Run a complete example of training and prediction"""
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    print(f"Using device: {device}")
    
    # Create result directories
    os.makedirs(f"./result/{args.mod_type}", exist_ok=True)
    os.makedirs(f"./result/normal/{args.cell_line}", exist_ok=True)
    
    # Set paths
    if args.cell_line == "IM95":
        normal_path = f"./result/normal/{args.cell_line}/{args.cell_line}.joblib"
        site_path = f"D:\\WLS\\utils\\sites\\{args.mod_type}\\{args.mod_type}.csv"
        data_path = f"D:\\WLS\\IM95\\rnano\\data_G.csv" if args.mod_type == "m7G" else f"D:\\WLS\\IM95\\rnano\\data_A.csv"
    else:
        print(f"Example not configured for cell line: {args.cell_line}")
        return
    
    # Determine kmer_nums based on modification type
    if args.mod_type == "m6A":
        kmer_nums = 206
    elif args.mod_type == "Nm":
        kmer_nums = 1023
    else:
        kmer_nums = 781
    
    print(f"Running example for {args.mod_type} modification in {args.cell_line} cell line")
    
    # Step 1: Load or create normalization data
    print("\nStep 1: Loading normalization data...")
    final_mean_std_results = load_or_create_normal_data(data_path, normal_path)
    
    # Step 2: Create datasets
    print("\nStep 2: Creating datasets...")
    train_set = NpDataset('Train', data_path, site_path, final_mean_std_results, kmer_nums, args.min_reads, args.mod_type)
    val_set = NpDataset('Val', data_path, site_path, final_mean_std_results, kmer_nums, args.min_reads, args.mod_type)
    
    # Step 3: Create and train model
    print("\nStep 3: Creating and training model...")
    model = Model(kmer_nums).to(device)
    
    # Training parameters
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Train for a few epochs (reduced for example)
    num_epochs = 3  # Reduced for example
    print(f"Training for {num_epochs} epochs (reduced for example)...")
    
    for epoch in range(num_epochs):
        model.train()
        # Get a small batch of data for demonstration
        X, kmers, labels = train_set[0]
        X = X.unsqueeze(0).to(device)
        kmers = kmers.unsqueeze(0).to(device)
        labels = torch.tensor([labels]).float().to(device)
        
        # Forward pass
        outputs = model({"kmer": kmers, "X": X})
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    
    # Step 4: Evaluate model
    print("\nStep 4: Evaluating model...")
    model.eval()
    
    # Get a small batch of validation data
    all_y_true = []
    all_y_pred = []
    
    with torch.no_grad():
        for i in range(min(10, len(val_set))):  # Use at most 10 samples for example
            X, kmers, labels = val_set[i]
            X = X.unsqueeze(0).to(device)
            kmers = kmers.unsqueeze(0).to(device)
            
            outputs = model({"kmer": kmers, "X": X})
            
            all_y_true.append(labels)
            all_y_pred.append(outputs.item())
    
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    # Calculate metrics
    metrics = get_metrics(all_y_true, all_y_pred)
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Step 5: Save model
    print("\nStep 5: Saving model...")
    model_path = f"./result/{args.mod_type}/{args.mod_type}_example_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Step 6: Generate plots
    print("\nStep 6: Generating plots...")
    plot_roc_curve(
        all_y_true, 
        all_y_pred, 
        title=f"Example {args.mod_type} for {args.cell_line}", 
        save=True,
        save_path=f"./result/{args.mod_type}/{args.mod_type}_example_roc.pdf"
    )
    
    plot_pr_curve(
        all_y_true, 
        all_y_pred, 
        title=f"Example {args.mod_type} for {args.cell_line}", 
        save=True,
        save_path=f"./result/{args.mod_type}/{args.mod_type}_example_pr.pdf"
    )
    
    print("\nExample completed successfully!")
    print(f"Check the ./result/{args.mod_type}/ directory for output files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RNANO Example Script")
    
    # Basic parameters
    parser.add_argument("--mod_type", type=str, default="m7G", 
                        choices=["m6A", "m1A", "m5C", "m7G", "ac4C", "Nm", "pU"],
                        help="Modification type")
    parser.add_argument("--cell_line", type=str, default="IM95", 
                        choices=["HEK293t", "Hela", "HepG2", "IM95", "hESCs"],
                        help="Cell line")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.003, help="Learning rate")
    parser.add_argument("--min_reads", type=int, default=20, help="Minimum number of reads")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID (-1 for CPU)")
    
    args = parser.parse_args()
    
    run_example(args)
