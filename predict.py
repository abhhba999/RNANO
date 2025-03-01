import torch
import numpy as np
import pandas as pd
import os
import argparse
from torch.utils.data import DataLoader

from model import Model
from dataset import NpDataset
from utils import load_or_create_normal_data, plot_roc_curve, plot_pr_curve

def predict(args):
    """Main prediction function"""
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    print(f"Using device: {device}")
    
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
    
    # Load normalization data
    final_mean_std_results = load_or_create_normal_data(data_path, normal_path)
    
    # Create dataset for prediction
    pred_set = NpDataset('Eval', data_path, site_path, final_mean_std_results, kmer_nums, args.min_reads, args.mod_type)
    
    # Create dataloader
    pred_dataloader = DataLoader(
        pred_set, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Create model
    model = Model(kmer_nums).to(device)
    
    # Load model
    model_path = f"./result/{args.mod_type}/{args.mod_type}_model_lr{args.learning_rate}_bs{args.batch_size}.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model file {model_path} not found. Using untrained model.")
    
    # Prediction
    model.eval()
    all_predictions = []
    all_site_ids = []
    all_read_ids = []
    
    with torch.no_grad():
        for batch in pred_dataloader:
            labels = batch.pop()
            site_ids = [pred_set.strings[idx.item()] for idx in labels[:, 0].long()]
            read_ids = labels[:, 1].long().tolist()
            
            y_pred = model({"kmer": batch[1].to(device), 'X': batch[0].to(device)})
            
            all_predictions.extend(y_pred.flatten().detach().cpu().numpy())
            all_site_ids.extend(site_ids)
            all_read_ids.extend(read_ids)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'site_id': all_site_ids,
        'read_id': all_read_ids,
        'prediction': all_predictions
    })
    
    # Save predictions
    output_path = f"./result/{args.mod_type}/{args.mod_type}_{args.cell_line}_predictions.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    
    # Aggregate predictions by site
    site_predictions = results_df.groupby('site_id')['prediction'].mean().reset_index()
    site_predictions.to_csv(f"./result/{args.mod_type}/{args.mod_type}_{args.cell_line}_site_predictions.csv", index=False)
    print(f"Site-level predictions saved to ./result/{args.mod_type}/{args.mod_type}_{args.cell_line}_site_predictions.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RNANO: RNA Modification Prediction")
    
    # Basic parameters
    parser.add_argument("--mod_type", type=str, default="m7G", 
                        choices=["m6A", "m1A", "m5C", "m7G", "ac4C", "Nm", "pU"],
                        help="Modification type")
    parser.add_argument("--cell_line", type=str, default="IM95", 
                        choices=["HEK293t", "Hela", "HepG2", "IM95", "hESCs"],
                        help="Cell line")
    
    # Model parameters
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.003, help="Learning rate used in the trained model")
    parser.add_argument("--min_reads", type=int, default=20, help="Minimum number of reads")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID (-1 for CPU)")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
    
    args = parser.parse_args()
    
    predict(args)
