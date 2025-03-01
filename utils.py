import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc, accuracy_score
import matplotlib.pyplot as plt
import joblib
import datatable as dt
import pandas as pd
from numba import jit
import os

def get_roc_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate ROC AUC score"""
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    return roc_auc

def get_pr_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Precision-Recall AUC score"""
    precision, recall, _ = precision_recall_curve(y_true, y_pred, pos_label=1)
    pr_auc = auc(recall, precision)
    return pr_auc

def get_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate accuracy score"""
    y_pred = np.where(y_pred > 0.5, 1, 0)
    return accuracy_score(y_true, y_pred)

def get_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate multiple metrics for evaluation"""
    # Convert predictions to binary
    y_pred_binary = np.where(y_pred > 0.5, 1, 0)
    
    # Calculate TP, TN, FP, FN
    tp = np.sum((y_true == 1) & (y_pred_binary == 1))
    tn = np.sum((y_true == 0) & (y_pred_binary == 0))
    fp = np.sum((y_true == 0) & (y_pred_binary == 1))
    fn = np.sum((y_true == 1) & (y_pred_binary == 0))
    
    # Calculate metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Matthews correlation coefficient
    mcc_numerator = (tp * tn) - (fp * fn)
    mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = mcc_numerator / mcc_denominator if mcc_denominator > 0 else 0
    
    # F1 score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = sensitivity
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'roc_auc': get_roc_auc(y_true, y_pred),
        'pr_auc': get_pr_auc(y_true, y_pred),
        'accuracy': get_accuracy(y_true, y_pred),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'mcc': mcc,
        'f1': f1
    }

def plot_roc_curve(y_true: np.ndarray, y_pred: np.ndarray, title="", save=False, save_path=None):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    lw = 1.5  # Line width
    plt.plot(fpr, tpr, color='#339999', lw=lw, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='#ff6666', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14, fontfamily='Arial')
    plt.ylabel('True Positive Rate', fontsize=14, fontfamily='Arial')
    plt.title(f'ROC Curve of {title}', fontsize=14, fontfamily='Arial')
    plt.legend(loc="lower right", prop={'size': 14})
    plt.tick_params(labelsize=14)
    
    if save and save_path:
        plt.savefig(save_path, format='pdf')
    plt.show()

def plot_pr_curve(y_true: np.ndarray, y_pred: np.ndarray, title="", save=False, save_path=None):
    """Plot Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_pred, pos_label=1)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    lw = 1.5
    plt.plot(recall, precision, color='#339999', lw=lw, label=f'PR curve (area = {pr_auc:.2f})')
    plt.fill_between(recall, precision, alpha=0., color='#99CC66')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=14, fontfamily='Arial')
    plt.ylabel('Precision', fontsize=14, fontfamily='Arial')
    plt.title(f'PR Curve of {title}', fontsize=14, fontfamily='Arial')
    plt.legend(loc="best", prop={'size': 14})
    plt.tick_params(labelsize=14)
    
    if save and save_path:
        plt.savefig(save_path, format='pdf')
    plt.show()

def load_or_create_normal_data(data_path, normal_path):
    """Load or create normal data for normalization"""
    if os.path.exists(normal_path):
        final_mean_std_results = joblib.load(normal_path) 
        print("load normal success")
        return final_mean_std_results
    else:
        print("start creating normal data")
        # Read CSV file and extract relevant columns
        data = dt.fread(data_path, sep=",", columns=[
            "ind", "kmer", 
            "signal_means_1", "signal_stds_1", "signal_length_1", "signal_amplitude_1", "signal_skewness_1", "signal_kurtosis_1",
            "signal_means_2", "signal_stds_2", "signal_length_2", "signal_amplitude_2", "signal_skewness_2", "signal_kurtosis_2",
            "signal_means_3", "signal_stds_3", "signal_length_3", "signal_amplitude_3", "signal_skewness_3", "signal_kurtosis_3",
            "signal_means_4", "signal_stds_4", "signal_length_4", "signal_amplitude_4", "signal_skewness_4", "signal_kurtosis_4",
            "signal_means_5", "signal_stds_5", "signal_length_5", "signal_amplitude_5", "signal_skewness_5", "signal_kurtosis_5",
            "read_ind"
        ]).to_pandas()

        # Merge all feature columns into a numpy array for fast computation
        feature_columns = [
            "signal_means_1", "signal_stds_1", "signal_length_1", "signal_amplitude_1", "signal_skewness_1", "signal_kurtosis_1",
            "signal_means_2", "signal_stds_2", "signal_length_2", "signal_amplitude_2", "signal_skewness_2", "signal_kurtosis_2",
            "signal_means_3", "signal_stds_3", "signal_length_3", "signal_amplitude_3", "signal_skewness_3", "signal_kurtosis_3",
            "signal_means_4", "signal_stds_4", "signal_length_4", "signal_amplitude_4", "signal_skewness_4", "signal_kurtosis_4",
            "signal_means_5", "signal_stds_5", "signal_length_5", "signal_amplitude_5", "signal_skewness_5", "signal_kurtosis_5"
        ]

        features = data[feature_columns].to_numpy()

        # Extract kmer column
        kmers = data['kmer'].to_numpy()
        print("data loaded")
        
        # Use numba's jit to accelerate computation
        @jit
        def process_kmers(kmers, features):
            result_dict = {}
            for idx in range(len(kmers)):
                kmer = kmers[idx]
                for i in range(5):
                    sub_kmer = kmer[i:i+5]
                    if sub_kmer not in result_dict:
                        result_dict[sub_kmer] = []
                    result_dict[sub_kmer].append(features[idx][i*6:i*6+6])
            return result_dict

        # Process data
        processed_data = process_kmers(kmers, features)
        del data
        print("data processed")
        
        # Calculate mean and standard deviation for each 5mer
        final_mean_std_results = {}
        for kmer, feature_list in processed_data.items():
            feature_array = np.array(feature_list)
            means = np.mean(feature_array, axis=0)
            stds = np.std(feature_array, axis=0)
            final_mean_std_results[kmer] = {
                'mean': means,
                'std': stds
            }
        joblib.dump(final_mean_std_results, normal_path)
        
        print("normal data created and saved")
        del processed_data
        return final_mean_std_results

class ImbalanceOverSampler:
    """Sampler to handle imbalanced datasets"""
    def __init__(self, data_source):
        labels_array = np.array(data_source.labels)
        class_counts = np.unique(labels_array, return_counts=True)
        minority_class, majority_class = class_counts[0][np.argmin(class_counts[1])], class_counts[0][np.argmax(class_counts[1])]
        self.minority_class_idx = np.argwhere(labels_array == minority_class).flatten()
        self.majority_class_idx = np.argwhere(labels_array == majority_class).flatten()

    def __iter__(self):
        idx = np.append(self.majority_class_idx, np.random.choice(self.minority_class_idx,
                                                                len(self.majority_class_idx), replace=True))
        np.random.shuffle(idx)
        return iter(idx)

    def __len__(self):
        return len(self.majority_class_idx) + len(self.majority_class_idx)
