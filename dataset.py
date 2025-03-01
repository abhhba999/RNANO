import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import datatable as dt
from itertools import product
from typing import Dict, List, Tuple, Union, Optional

class NpDataset(Dataset):
    def __init__(self, mode, csvpath, site_path, factor, kmer_nums, min_reads=20, mod_type="m7G"):
        self.mode = mode
        self.min_reads = min_reads
        self.mod_type = mod_type
        
        # Load site information
        site = pd.read_csv(site_path, sep='\t', names=['seqnames', 'starts', 'ends', 'scores', 'strand'])
        site['ind'] = site['seqnames'] + '_' + site['starts'].astype(str)
        site.set_index('ind', inplace=True)
        self.res = site
        
        # Load data
        self.data = self.load_csv(csvpath)
        
        # Create unique identifiers for reads
        li = []
        q = 0
        m = self.data.index.values
        li.append(m[0] + "_" + str(q))
        for i in range(len(m)):
            if i == 0:
                continue
            if m[i] == m[i-1]:
                q = q + 1
                li.append(m[i] + "_" + str(q))
            else:
                q = 0
                li.append(m[i] + "_" + str(q))
                
        self.data["ins_index"] = li
        self.data["pse_label"] = 0
        self.ins_psu_dict = self.data.set_index('ins_index')['pse_label'].to_dict()
        self.sites = list(set(self.data.index.values))
        
        # Define motifs based on modification type
        if mod_type == "m6A":
            motifs = [['G', 'A', 'C', 'T'], ['G', 'A', 'C', 'T'], ['A', 'G', 'T'], ['G', 'A'], ['A'], ['C'], ['A', 'C', 'T'], ['G', 'A', 'C', 'T'], ['G', 'A', 'C', 'T']]
            self.kmer_nums = 206
        elif mod_type == "m1A":
            motifs = [['G', 'A', 'C', 'T'], ['G', 'A', 'C', 'T'], ['G', 'A', 'C', 'T'], ['G', 'A', 'C', 'T'], ['A'], ['G', 'A', 'C', 'T'], ['G', 'A', 'C', 'T'], ['G', 'A', 'C', 'T'], ['G', 'A', 'C', 'T']]
            self.kmer_nums = 781
        elif mod_type == "m5C" or mod_type == "ac4C":
            motifs = [['G', 'A', 'C', 'T'], ['G', 'A', 'C', 'T'], ['G', 'A', 'C', 'T'], ['G', 'A', 'C', 'T'], ['C'], ['G', 'A', 'C', 'T'], ['G', 'A', 'C', 'T'], ['G', 'A', 'C', 'T'], ['G', 'A', 'C', 'T']]
            self.kmer_nums = 781
        elif mod_type == "m7G":
            motifs = [['G', 'A', 'C', 'T'], ['G', 'A', 'C', 'T'], ['G', 'A', 'C', 'T'], ['G', 'A', 'C', 'T'], ['G'], ['G', 'A', 'C', 'T'], ['G', 'A', 'C', 'T'], ['G', 'A', 'C', 'T'], ['G', 'A', 'C', 'T']]
            self.kmer_nums = 781
        elif mod_type == "Nm":
            motifs = [['G', 'A', 'C', 'T'], ['G', 'A', 'C', 'T'], ['G', 'A', 'C', 'T'], ['G', 'A', 'C', 'T'], ['A', 'C', 'G'], ['G', 'A', 'C', 'T'], ['G', 'A', 'C', 'T'], ['G', 'A', 'C', 'T'], ['G', 'A', 'C', 'T']]
            self.kmer_nums = 1023
        elif mod_type == "pU":
            motifs = [['G', 'A', 'C', 'T'], ['G', 'A', 'C', 'T'], ['G', 'A', 'C', 'T'], ['G', 'A', 'C', 'T'], ['T'], ['G', 'A', 'C', 'T'], ['G', 'A', 'C', 'T'], ['G', 'A', 'C', 'T'], ['G', 'A', 'C', 'T']]
            self.kmer_nums = 781
            
        all_kmers = list(["".join(x) for x in product(*motifs)])
        self.all_kmers = np.unique(np.array(list(map(lambda x: [x[i:i+5] for i in range(len(x) - 4)], all_kmers))).flatten())
        self.kmer_to_int = {self.all_kmers[i]: i for i in range(len(self.all_kmers))}
        self.int_to_kmer = {i: self.all_kmers[i] for i in range(len(self.all_kmers))}
        
        self.factor = factor
        self.resind = set(self.res.index.values)
        
        # Process data for batch access
        string_array = self.data.index.values
        change_indices = np.where(string_array[:-1] != string_array[1:])[0]
        
        start_indices = np.concatenate(([0], change_indices + 1))
        end_indices = np.concatenate((change_indices, [len(string_array) - 1]))
        
        self.strings = string_array[start_indices]
        self.labels = [1 if i in self.resind else 0 for i in self.strings]
        self.start_positions = start_indices
        self.end_positions = end_indices + 1
        self.data = self.data.values
        
    def load_csv(self, csvpath):
        data = dt.fread(csvpath, sep=",", columns=[
            "ind", "kmer", 
            "signal_means_1", "signal_stds_1", "signal_length_1", "signal_amplitude_1", "signal_skewness_1", "signal_kurtosis_1",
            "signal_means_2", "signal_stds_2", "signal_length_2", "signal_amplitude_2", "signal_skewness_2", "signal_kurtosis_2",
            "signal_means_3", "signal_stds_3", "signal_length_3", "signal_amplitude_3", "signal_skewness_3", "signal_kurtosis_3",
            "signal_means_4", "signal_stds_4", "signal_length_4", "signal_amplitude_4", "signal_skewness_4", "signal_kurtosis_4",
            "signal_means_5", "signal_stds_5", "signal_length_5", "signal_amplitude_5", "signal_skewness_5", "signal_kurtosis_5",
            "read_ind"
        ]).to_pandas()
        
        data = data[["ind", "kmer", 
                    "signal_means_1", "signal_stds_1", "signal_length_1", "signal_amplitude_1", "signal_skewness_1", "signal_kurtosis_1",
                    "signal_means_2", "signal_stds_2", "signal_length_2", "signal_amplitude_2", "signal_skewness_2", "signal_kurtosis_2",
                    "signal_means_3", "signal_stds_3", "signal_length_3", "signal_amplitude_3", "signal_skewness_3", "signal_kurtosis_3",
                    "signal_means_4", "signal_stds_4", "signal_length_4", "signal_amplitude_4", "signal_skewness_4", "signal_kurtosis_4",
                    "signal_means_5", "signal_stds_5", "signal_length_5", "signal_amplitude_5", "signal_skewness_5", "signal_kurtosis_5",
                    "read_ind"]]
        
        print("all reads in dataset:", data.shape)
        data.set_index("ind", inplace=True)
        
        # Filter data based on mode
        inter = list(set(data.index.values).intersection(set(self.res.index.values)))
        all_outer = list(set(data.index.values).difference(set(self.res.index.values)))
        
        print("all sites:", len(data.index.value_counts()))
        print("nega sites:", len(all_outer))
        print("all posi sites:", len(inter))
        
        # Sample negative sites to balance dataset
        np.random.seed(42)
        all_outer = np.random.choice(all_outer, len(inter), replace=False)
        print("sel nega sites:", len(all_outer))
        print("intersection sites:", len(set(all_outer).intersection(set(inter))))
        
        all = np.concatenate([inter, all_outer])
        np.random.shuffle(all)
        
        if self.mode == "Train":
            data = data.loc[all[0:int(len(all) * 0.7)]]
        if self.mode == "Val":
            data = data.loc[all[int(len(all) * 0.01):]]
            
        all_sites = data.index.value_counts()
        return data.loc[all_sites[all_sites.values >= 20].index.values]
        
    def __getitem__(self, index):
        sam_num = 10
        
        if self.mode == "Train" or self.mode == "Val":
            tmp = self.data[self.start_positions[index]:self.end_positions[index], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32]]
            tmp = tmp[np.random.choice(tmp.shape[0], self.min_reads, replace=False), :-1] 
        else:
            tmp = self.data[self.start_positions[index]:self.end_positions[index], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]]
            if reads == 1:
                tmp = tmp[0:self.min_reads * sam_num, :]
            else:
                tmp = tmp[np.random.choice(tmp.shape[0], self.min_reads * sam_num, replace=True), :]
        
        if self.mode == "Train" or self.mode == "Val":
            pass
        else:
            read_ind = tmp[:, -1].astype(int)
            tmp = tmp[:, :-1]
            
        kmer = tmp[:, 0]
        tmp = tmp[:, 1:]
        
        if self.mode == "Train" or self.mode == "Val":
            for i in range(self.min_reads):
                for j in range(5):
                    u, v = self.factor[kmer[i][j:j+5]]["mean"], self.factor[kmer[i][j:j+5]]["std"]
                    tmp[i, j*6:j*6+6] = (tmp[i, j*6:j*6+6] - u) / v 
        else:
            for i in range(self.min_reads * sam_num):
                for j in range(5):
                    u, v = self.factor[kmer[i][j:j+5]]["mean"], self.factor[kmer[i][j:j+5]]["std"]
                    tmp[i, j*6:j*6+6] = (tmp[i, j*6:j*6+6] - u) / v
      
        kmer = np.array([[self.kmer_to_int[k[i:i+5]] for i in range(9-4)] for k in kmer]).astype(int)
        tmp = tmp[:, :].astype(float)
        
        if self.mode == "Train" or self.mode == "Val":
            label = self.labels[index]
        else:
            label = self.strings[index]
            label = [int(x) for x in label[4:].replace('.', '_').split('_')]
            label.extend(read_ind.tolist())  # Append read_ind to the label
            label = torch.Tensor(label)
            
        return (torch.Tensor(tmp), torch.Tensor(kmer), label)
    
    def update_pse_label(self, pse_dict):
        self.stats = "pseudo"
        self.ins_psu_dict.update(pse_dict)
        print(f"mode = {self.stats}, used {len(pse_dict)} pseudo labels to replace the positive examples in the original {len(self.ins_psu_dict)} pseudo labels")

    def __len__(self):
        return len(self.strings)
