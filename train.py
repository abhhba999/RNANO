import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import time
import argparse
import random

from model import Model
from dataset import NpDataset
from utils import get_metrics, get_roc_auc, get_pr_auc, get_accuracy, plot_roc_curve, plot_pr_curve, load_or_create_normal_data, ImbalanceOverSampler

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(args):
    """Main training function"""
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    print(f"Using device: {device}")
    
    # Set random seed
    set_seed(args.seed)
    
    # Determine paths based on cell line and modification type
    if args.cell_line == "Hela" or args.cell_line == "IM95":
        normal_path = f"./result/normal/{args.cell_line}/{args.cell_line}.joblib"
    else:
        normal_path = f"./result/normal/{args.cell_line}/{args.mod_type}.joblib"
    
    # Determine kmer_nums based on modification type
    if args.mod_type == "m6A":
        kmer_nums = 206
    elif args.mod_type == "Nm":
        kmer_nums = 1023
    else:
        kmer_nums = 781
    
    # Set data path based on cell line and modification type
    if args.cell_line == "Hela":
        site_path = f"D:\\WLS\\utils\\sites\\Hela\\{args.mod_type}\\{args.mod_type}.csv"
        if args.mod_type == "m6A":
            data_path = r"C:\WLS\G\TandemMod\wlshela\rnano\m6A\data.csv"
        elif args.mod_type == "m1A":
            data_path = r"C:\WLS\G\TandemMod\wlshela\rnano\data_A.csv"
        elif args.mod_type == "m5C" or args.mod_type == "ac4C":
            data_path = r"C:\WLS\G\TandemMod\wlshela\rnano\data_C.csv"
        elif args.mod_type == "m7G":
            data_path = r"C:\WLS\G\TandemMod\wlshela\rnano\data_G.csv"
        elif args.mod_type == "Nm":
            data_path = r"C:\WLS\G\TandemMod\wlshela\rnano\data_C.csv"
        elif args.mod_type == "pU":
            data_path = r"C:\WLS\G\TandemMod\wlshela\rnano\data_T.csv"
    elif args.cell_line == "IM95":
        site_path = f"D:\\WLS\\utils\\sites\\{args.mod_type}\\{args.mod_type}.csv"
        if args.mod_type == "m6A":
            data_path = f"D:\\WLS\\IM95\\rnano\\m6A\\data.csv"
        elif args.mod_type == "m1A":
            data_path = f"D:\\WLS\\IM95\\rnano\\data_A.csv"
        elif args.mod_type == "m5C" or args.mod_type == "ac4C":
            data_path = f"D:\\WLS\\IM95\\rnano\\data_C.csv"
        elif args.mod_type == "m7G":
            data_path = f"D:\\WLS\\IM95\\rnano\\data_G.csv"
        elif args.mod_type == "Nm":
            data_path = f"D:\\WLS\\IM95\\rnano\\data_C.csv"
        elif args.mod_type == "pU":
            data_path = f"D:\\WLS\\IM95\\rnano\\data_T.csv"
    elif args.cell_line == "HEK293t":
        site_path = f"D:\\WLS\\utils\\sites\\{args.mod_type}\\{args.mod_type}.csv"
        if args.mod_type == "m6A":
            data_path = r"D:\WLS\29311\basecall_result\rnano\m6A\data.csv"
        elif args.mod_type == "m1A":
            data_path = r"D:\WLS\29311\basecall_result\rnano\m1A\data.csv"
        elif args.mod_type == "m5C":
            data_path = r"D:\WLS\29311\basecall_result\rnano\m5C\data.csv"
        elif args.mod_type == "pU":
            data_path = r"C:\WLS\G\29311\pU\data.csv"
    elif args.cell_line == "HepG2":
        site_path = f"D:\\WLS\\utils\\sites\\{args.mod_type}\\{args.mod_type}.csv"
        if args.mod_type == "m7G":
            data_path = r"C:\WLS\G\g221\rnano\m7G\data.json"
        elif args.mod_type == "Nm":
            data_path = r"C:\WLS\G\g221\rnano\Nm\data.json"
    elif args.cell_line == "hESCs":
        site_path = f"D:\\WLS\\utils\\sites\\{args.mod_type}\\{args.mod_type}.csv"
        if args.mod_type == "ac4C":
            data_path = r"C:\WLS\G\h9\rnano\data.csv"
    
    # Create result directory if it doesn't exist
    os.makedirs(f"./result/{args.mod_type}", exist_ok=True)
    os.makedirs(f"./result/normal/{args.cell_line}", exist_ok=True)
    
    # Load or create normalization data
    final_mean_std_results = load_or_create_normal_data(data_path, normal_path)
    
    # Create datasets
    if args.mode == "Train":
        train_set = NpDataset('Train', data_path, site_path, final_mean_std_results, kmer_nums, args.min_reads, args.mod_type)
        val_set = NpDataset('Val', data_path, site_path, final_mean_std_results, kmer_nums, args.min_reads, args.mod_type)
    elif args.mode == "Val":
        val_set = NpDataset('Val', data_path, site_path, final_mean_std_results, kmer_nums, args.min_reads, args.mod_type)
    
    # Create model
    model = Model(kmer_nums).to(device)
    
    # Set up criterion
    criterion = nn.BCELoss()
    
    # Training loop
    if args.mode == "Train":
        print("Starting training...")
        
        # Create dataloaders
        train_dataloader = DataLoader(
            train_set, 
            batch_size=args.batch_size,
            sampler=ImbalanceOverSampler(train_set),
            num_workers=args.num_workers
        )
        
        val_dataloader = DataLoader(
            val_set, 
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        
        # Set up optimizer and scheduler
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.8)
        
        # Initialize variables
        best_pr_auc = 0
        best_metrics = None
        results_list = []
        
        # Training loop
        for epoch in range(args.epochs):
            model.train()
            start = time.time()
            all_y_true = []
            all_y_pred = []
            train_loss_list = []
            
            for batch in train_dataloader:
                y_true = batch.pop().to(device).flatten()
                y_pred = model({"kmer": batch[1].to(device), 'X': batch[0].to(device)})
                
                loss = criterion(y_pred, y_true)
                train_loss_list.append(loss.item())
                
                loss.backward()
                
                # Gradient clipping
                clip_value = 1.0
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                if grad_norm > clip_value:
                    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)
                
                optimizer.step()
                optimizer.zero_grad()
                
                y_true = y_true.detach().cpu().numpy()
                y_pred = y_pred.detach().cpu().numpy()
                all_y_true.extend(y_true)
                all_y_pred.extend(y_pred.flatten())
            
            scheduler.step()
            
            # Calculate training metrics
            compute_time = time.time() - start
            all_y_true = np.array(all_y_true)
            all_y_pred = np.array(all_y_pred)
            
            print(f"Epoch {epoch} train_loss={torch.mean(torch.tensor(train_loss_list)).item():.4f}, "
                  f"ROC={get_roc_auc(all_y_true, all_y_pred):.4f}, "
                  f"PR={get_pr_auc(all_y_true, all_y_pred):.4f}, "
                  f"Acc={get_accuracy(all_y_true, all_y_pred):.4f}")
            
            # Validation
            model.eval()
            all_y_true = None
            all_y_pred = []
            start = time.time()
            
            with torch.no_grad():
                for _ in range(1):
                    y_true_tmp = []
                    y_pred_tmp = []
                    
                    for batch in val_dataloader:
                        y_true = batch.pop().to(device).flatten()
                        y_pred = model({"kmer": batch[1].to(device), 'X': batch[0].to(device)})
                        
                        if all_y_true is None:
                            y_true_tmp.extend(y_true.detach().cpu().numpy())
                        
                        y_pred_tmp.extend(y_pred.flatten().detach().cpu().numpy())
                    
                    if all_y_true is None:
                        all_y_true = y_true_tmp
                    
                    all_y_pred.append(y_pred_tmp)
            
            compute_time = time.time() - start
            y_pred_avg = torch.mean(torch.tensor(all_y_pred), axis=0).detach().cpu().numpy()
            all_y_true = np.array(all_y_true).flatten()
            
            # Calculate validation metrics
            val_loss = criterion(torch.Tensor(y_pred_avg), torch.Tensor(all_y_true)).item()
            val_metrics = get_metrics(all_y_true, y_pred_avg)
            
            print(f"Epoch {epoch} val_loss={val_loss:.4f}, metrics: {val_metrics}")
            
            # Save results
            val_metrics["epoch"] = epoch
            results_list.append(val_metrics)
            
            # Save best model
            if val_metrics['pr_auc'] > best_pr_auc:
                best_pr_auc = val_metrics['pr_auc']
                best_metrics = val_metrics
                torch.save(model.state_dict(), f"./result/{args.mod_type}/{args.mod_type}_model_lr{args.learning_rate}_bs{args.batch_size}.pt")
                print(f"Saved new best model with PR AUC: {best_pr_auc:.4f}")
        
        # Save results to CSV
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(f"./result/{args.mod_type}/{args.mod_type}_results_lr{args.learning_rate}_bs{args.batch_size}.csv", index=False)
        
        # Plot final curves
        plot_roc_curve(
            all_y_true, 
            y_pred_avg, 
            title=f"{args.mod_type} for lr{args.learning_rate} bs{args.batch_size}", 
            save=True,
            save_path=f"./result/{args.mod_type}/{args.mod_type}_train_roc.pdf"
        )
        
        plot_pr_curve(
            all_y_true, 
            y_pred_avg, 
            title=f"{args.mod_type} for lr{args.learning_rate} bs{args.batch_size}", 
            save=True,
            save_path=f"./result/{args.mod_type}/{args.mod_type}_train_pr.pdf"
        )
        
        print(f"Training completed. Best model saved with metrics: {best_metrics}")
    
    elif args.mode == "Val":
        # Load best model
        model_path = f"./result/{args.mod_type}/{args.mod_type}_model_lr{args.learning_rate}_bs{args.batch_size}.pt"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            print(f"Loaded model from {model_path}")
        else:
            print(f"Model file {model_path} not found. Using untrained model.")
        
        # Create dataloader
        val_dataloader = DataLoader(
            val_set, 
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        
        # Validation
        model.eval()
        all_y_true = []
        all_y_pred = []
        
        with torch.no_grad():
            for batch in val_dataloader:
                y_true = batch.pop().to(device).flatten()
                y_pred = model({"kmer": batch[1].to(device), 'X': batch[0].to(device)})
                
                all_y_true.extend(y_true.detach().cpu().numpy())
                all_y_pred.extend(y_pred.flatten().detach().cpu().numpy())
        
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        
        # Calculate validation metrics
        val_metrics = get_metrics(all_y_true, all_y_pred)
        print(f"Validation metrics: {val_metrics}")
        
        # Plot curves
        plot_roc_curve(
            all_y_true, 
            all_y_pred, 
            title=f"Test on {args.cell_line}, {args.mod_type}", 
            save=True,
            save_path=f"./result/{args.mod_type}/{args.mod_type}_val_roc.pdf"
        )
        
        plot_pr_curve(
            all_y_true, 
            all_y_pred, 
            title=f"Test on {args.cell_line}, {args.mod_type}", 
            save=True,
            save_path=f"./result/{args.mod_type}/{args.mod_type}_val_pr.pdf"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RNANO: RNA Modification Detection")
    
    # Basic parameters
    parser.add_argument("--mode", type=str, default="Train", choices=["Train", "Val", "Predict"], 
                        help="Mode: Train, Val, or Predict")
    parser.add_argument("--mod_type", type=str, default="m7G", 
                        choices=["m6A", "m1A", "m5C", "m7G", "ac4C", "Nm", "pU"],
                        help="Modification type")
    parser.add_argument("--cell_line", type=str, default="IM95", 
                        choices=["HEK293t", "Hela", "HepG2", "IM95", "hESCs"],
                        help="Cell line")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.003, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--min_reads", type=int, default=20, help="Minimum number of reads")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID (-1 for CPU)")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
    
    args = parser.parse_args()
    
    train(args)
