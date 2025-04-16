import sys
import torch
import numpy as np
import sklearn
import random
import time
import os
import datetime
import torch.nn.functional as F
from torch.nn import Linear
from sklearn import preprocessing
from sklearn.metrics import  classification_report, confusion_matrix, matthews_corrcoef, precision_recall_fscore_support, adjusted_rand_score
import pandas as pd
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GraphConv, global_mean_pool
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

import threading

def validate_model_on_fold(fold, input_dir, edges_fn, model_fn, output_dir, epochs):
    """Validate a pretrained model on a specific fold"""
    log_file = os.path.join(output_dir, f"validation_fold_{fold}_log.txt")
    log_message(f"Validating model on fold {fold}", log_file)
    
    # Call main_single_experiment with the validation parameters
    return main_single_experiment(
        experiment_id=fold,
        output_dir=output_dir,
        input_dir=input_dir,
        edges_fn_override=edges_fn,
        model_fn=model_fn,
        epochs_override=epochs
    )

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 51  # Original model output size
HIDDEN_CHANNELS = 64
BATCH_SIZE = 64
EPOCHS = 500

# # Default file paths (will be overridden by command-line arguments when provided)
# edges_fn = '/mnt/home/yuankeji/RanceLab/reticula_new/reticula/data/GEO_model_training/input/edges_all.txt'
# node_features_fn = '/mnt/home/yuankeji/RanceLab/reticula_new/reticula/data/GEO_model_training/input/node_features_all.txt'
# graph_targets_fn = '/mnt/home/yuankeji/RanceLab/reticula_new/reticula/data/GEO_model_training/input/graph_targets_all.txt'

log_lock = threading.Lock()

def log_message(message, filename, lock=None):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message_with_timestamp = f"[{timestamp}] {message}"
    
    if lock:
        with lock:
            with open(filename, 'a') as f:
                f.write(f"{message_with_timestamp}\n")
            print(message_with_timestamp)
    else:
        with open(filename, 'a') as f:
            f.write(f"{message_with_timestamp}\n")
        print(message_with_timestamp)

def read_reactome_graph(edges_fn, node_features_fn, log_file):
    edge_v1 = []
    edge_v2 = []

    log_message(f"Reading edges from {edges_fn}", log_file, log_lock)
    with open(edges_fn, 'r') as f:
        for line in f:
            data = line.split()
            node1 = int(data[0]) - 1  
            node2 = int(data[1]) - 1  
            edge_v1.append(node1)
            edge_v2.append(node2)

    log_message(f"Loaded {len(edge_v1)} edges", log_file, log_lock)
    return edge_v1, edge_v2

def build_reactome_graph_datalist(edge_v1, edge_v2, node_features_fn, graph_targets_fn, log_file):
    edge_index = torch.tensor([edge_v1, edge_v2], dtype=torch.long)
    
    log_message(f"Loading node features from {node_features_fn}", log_file, log_lock)
    feature_v = np.loadtxt(node_features_fn)
    
    log_message(f"Loading targets from {graph_targets_fn}", log_file, log_lock)
    target_v = np.loadtxt(graph_targets_fn, dtype=str, delimiter=",")
    
    target_encoder = sklearn.preprocessing.LabelEncoder()
    target_v = target_encoder.fit_transform(target_v)
    num_classes = len(np.unique(target_v))
    log_message(f"Detected {num_classes} unique classes", log_file, log_lock)

    class_mapping = {i: label for i, label in enumerate(target_encoder.classes_)}
    
    log_message(f"Building data list with {len(feature_v)} samples", log_file, log_lock)
    data_list = []
    for row_idx in range(len(feature_v)):
        features = feature_v[row_idx, :]
        x = torch.tensor(features, dtype=torch.float).unsqueeze(1)
        y = torch.tensor(target_v[row_idx], dtype=torch.long)
        data_list.append(Data(x=x, y=y, edge_index=edge_index))

    class_counts = np.bincount(target_v)
    for i, count in enumerate(class_counts):
        log_message(f"Class {i} ({class_mapping[i]}): {count} samples", log_file, log_lock)
        
    return data_list, target_v, target_encoder, num_classes, class_mapping

# Function to change key names in state dict
def change_key(state_dict, old, new):
    for _ in range(len(state_dict)):
        k, v = state_dict.popitem(False)
        state_dict[new if old == k else k] = v
    return state_dict

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_classes):
        super(GNN, self).__init__()
        self.conv1 = GraphConv(INPUT_CHANNELS, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight).relu()
        x = self.conv3(x, edge_index, edge_weight)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, training=self.training)
        return self.lin(x)

def train(loader, model, optimizer, device, epoch, total_epochs, experiment_id, log_file):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    desc = f"Exp {experiment_id} - Epoch {epoch+1}/{total_epochs} [Train]"

    for batch in loader:
        x, edge_index, batch_idx, y = batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device), batch.y.to(device)
        out = model(x, edge_index, batch_idx)
        loss = torch.nn.CrossEntropyLoss()(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        pred = out.argmax(dim=1)
        correct += int((pred == y).sum()) 

    return correct / len(loader.dataset)

def test(loader, model, device, experiment_id, desc, log_file, output_fn=None):
    model.eval()

    targets = []
    predictions = []
    for batch in loader:
        x, edge_index, batch_idx, y = batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device), batch.y.to(device)
        targets += torch.Tensor.tolist(y)
        out = model(x, edge_index, batch_idx)
        pred = out.argmax(dim=1)
        
        predictions += torch.Tensor.tolist(pred)
    
    # Calculate adjusted rand score
    ari = adjusted_rand_score(targets, predictions)
    log_message(f"Adjusted Rand Index: {ari:.4f}", log_file)

    # Get class mapping (assuming it's available from the training data)
    target_encoder = sklearn.preprocessing.LabelEncoder()
    target_encoder.classes_ = np.unique(targets)
    class_mapping = {i: str(cls) for i, cls in enumerate(target_encoder.classes_)}
    
    # Calculate detailed metrics
    metrics_results = calculate_metrics(np.array(targets), np.array(predictions), class_mapping)
    
    # Save metrics results to CSV
    metrics_df = pd.DataFrame(metrics_results)
    metrics_file = os.path.join(os.path.dirname("/mnt/home/yuankeji/RanceLab/validation/k-fold-gnn-val/"), f"k_fold_detailed_metrics_{experiment_id}.csv")
    metrics_df.to_csv(metrics_file, index=False)
    
    return ari

def calculate_metrics(y_true, y_pred, class_mapping):
    """计算每个组织类型的详细指标"""
    num_classes = len(class_mapping)

    results = []

    for class_idx in range(num_classes):
        class_name = class_mapping[class_idx]
        y_true_binary = (y_true == class_idx).astype(int)
        y_pred_binary = (y_pred == class_idx).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]).ravel()
        positives = tp + fn
        tpr = tp / positives if positives else 0
        fnr = fn / positives if positives else 0
        negatives = tn + fp
        tnr = tn / negatives if negatives else 0
        fpr = fp / negatives if negatives else 0
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tpr
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
        mcc = matthews_corrcoef(y_true_binary, y_pred_binary)
        results.append({
            'tissue': class_name,
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'POSITIVES': positives,
            'MCC': mcc,
            'TPR': tpr,
            'FPR': fpr,
            'TNR': tnr,
            'FNR': fnr,
            'Precision': precision,
            'Recall': recall,
            'F1_score': f1
        })
    return results

def run_experiment(experiment_id, data_list, num_classes, class_mapping, model_save_dir, pretrained_model_path=None):
    log_file = os.path.join(model_save_dir, f"experiment_{experiment_id}_log.txt")
    
    seed = 42 + experiment_id
    set_seed(seed)
    
    # Always use CPU
    device = torch.device('cpu')
    
    log_message(f"=== Starting experiment {experiment_id} ===", log_file)
    log_message(f"Using device: {device}", log_file)
    log_message(f"Using seed: {seed}", log_file)
    
    # Use alternating indices for train/test split as in the second code
    train_data_list = data_list[0::2]
    test_data_list = data_list[1::2]
    
    log_message(f"Experiment {experiment_id}: {len(train_data_list)} train samples, {len(test_data_list)} test samples", log_file)
    
    # Use shuffle=True for train loader, shuffle=False for test loader
    train_loader = DataLoader(train_data_list, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data_list, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model for either fresh training or transfer learning
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        log_message(f"Loading pretrained model from {pretrained_model_path}", log_file)
        
        # Initialize with OUTPUT_CHANNELS first (original model shape)
        model = GNN(hidden_channels=HIDDEN_CHANNELS, num_classes=num_classes).to(device)
        
        # Change the output layer to match the new dataset
        model.lin = Linear(HIDDEN_CHANNELS, num_classes).to(device)
        
        # Load the pretrained state dict
        state_dict = torch.load(pretrained_model_path, map_location=device)
        
        # Update key names if needed
        change_key(state_dict, 'conv1.lin_l.weight', 'conv1.lin_rel.weight')
        change_key(state_dict, 'conv1.lin_l.bias', 'conv1.lin_rel.bias')
        change_key(state_dict, 'conv1.lin_r.weight', 'conv1.lin_root.weight')
        change_key(state_dict, 'conv2.lin_l.weight', 'conv2.lin_rel.weight')
        change_key(state_dict, 'conv2.lin_l.bias', 'conv2.lin_rel.bias')
        change_key(state_dict, 'conv2.lin_r.weight', 'conv2.lin_root.weight')
        change_key(state_dict, 'conv3.lin_l.weight', 'conv3.lin_rel.weight')
        change_key(state_dict, 'conv3.lin_l.bias', 'conv3.lin_rel.bias')
        change_key(state_dict, 'conv3.lin_r.weight', 'conv3.lin_root.weight')
        change_key(state_dict, 'lin.weight', 'lin.weight')
        change_key(state_dict, 'lin.bias', 'lin.bias')

        # Remove linear layer weights (will be newly initialized)
        state_dict.pop('lin.weight', None)
        state_dict.pop('lin.bias', None)
        
        # Load state dict with strict=False to allow for the new linear layer
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        # Freeze convolutional layers
        model.conv1.lin_rel.weight.requires_grad = False
        model.conv1.lin_rel.bias.requires_grad = False
        model.conv1.lin_root.weight.requires_grad = False
        model.conv2.lin_rel.weight.requires_grad = False
        model.conv2.lin_rel.bias.requires_grad = False
        model.conv2.lin_root.weight.requires_grad = False
        model.conv3.lin_rel.weight.requires_grad = False
        model.conv3.lin_rel.bias.requires_grad = False
        model.conv3.lin_root.weight.requires_grad = False
        model.lin.weight.requires_grad = True
        model.lin.bias.requires_grad = True

        log_message("Transfer learning setup: convolutional layers frozen, only training linear layer", log_file)
        
        # Use AdamW optimizer for consistency
        optimizer = torch.optim.AdamW(model.parameters())
    else:
        # For fresh training, initialize the full model
        model = GNN(hidden_channels=HIDDEN_CHANNELS, num_classes=num_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters())
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        train_acc = train(train_loader, model, optimizer, device, epoch, EPOCHS, experiment_id, log_file)
        if train_acc == 1.0:
            break
        # Use the updated test function that calculates ARI
        output_fn = os.path.join(model_save_dir, f"predictions_epoch_{epoch+1}_experiment_{experiment_id}.tsv") if (epoch + 1) % 50 == 0 else None
        test_ari = test(test_loader, model, device, experiment_id, f"Epoch {epoch+1}/{EPOCHS} [Test]", log_file, output_fn)
        
        log_message(f"Experiment {experiment_id}, Epoch {epoch+1}/{EPOCHS}, Train Acc: {train_acc:.4f}, Test ARI: {test_ari:.4f}", log_file)
        

        model_save_path = os.path.join(model_save_dir, f"model_experiment_{experiment_id}.pt")
        torch.save(model.state_dict(), model_save_path)
    
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    log_message(f"Experiment {experiment_id} completed in {int(hours)}h {int(minutes)}m {int(seconds)}s", log_file)


def main_single_experiment(experiment_id, output_dir, input_dir=None, edges_fn_override=None, model_fn=None, epochs_override=None):
    """运行单个实验"""
    model_save_dir = output_dir
    os.makedirs(model_save_dir, exist_ok=True)
    
    main_log_file = os.path.join(model_save_dir, f"validation_fold_{experiment_id}_log.txt")
    
    log_message(f"Starting GNN validation for fold {experiment_id}", main_log_file)
    
    # Override file paths if provided
    global edges_fn, node_features_fn, graph_targets_fn, EPOCHS
    
    # Update file paths based on input_dir and fold number
    if input_dir is not None:
        if edges_fn_override is not None:
            edges_fn = edges_fn_override
        else:
            edges_fn = os.path.join(input_dir, f"edges_val_{experiment_id}.txt")
        
        node_features_fn = os.path.join(input_dir, f"node_features_val_{experiment_id}.txt")
        graph_targets_fn = os.path.join(input_dir, f"graph_targets_val_{experiment_id}.txt")
    
    # Override epochs if provided
    if epochs_override is not None:
        EPOCHS = epochs_override
    
    log_message(f"Using edges file: {edges_fn}", main_log_file)
    log_message(f"Using node features file: {node_features_fn}", main_log_file)
    log_message(f"Using graph targets file: {graph_targets_fn}", main_log_file)
    log_message(f"Epochs: {EPOCHS}", main_log_file)
    if model_fn is not None:
        log_message(f"Using pretrained model: {model_fn}", main_log_file)
    
    edge_v1, edge_v2 = read_reactome_graph(edges_fn, node_features_fn, main_log_file)
    data_list, labels, target_encoder, num_classes, class_mapping = build_reactome_graph_datalist(edge_v1, edge_v2, node_features_fn, graph_targets_fn, main_log_file)
    
    mapping_file = os.path.join(model_save_dir, f"class_mapping_fold_{experiment_id}.csv")
    with open(mapping_file, 'w') as f:
        f.write("class_id,tissue_name\n")
        for class_id, tissue_name in class_mapping.items():
            f.write(f"{class_id},{tissue_name}\n")
    log_message(f"Class mapping saved to {mapping_file}", main_log_file)
    
    log_message(f"Number of classes: {num_classes}", main_log_file)
    
    # 直接运行单次验证实验，不进行多次实验
    start_time = time.time()
    log_message(f"Running validation for fold {experiment_id} with pretrained model", main_log_file)
    run_experiment(experiment_id, data_list, num_classes, class_mapping, model_save_dir, model_fn)
    
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    log_message(f"Fold {experiment_id} validation total runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s", main_log_file)
    

def main():
    parser = argparse.ArgumentParser(description='GNN with Parallel Random Subsampling Validation')
    
    # Original arguments
    parser.add_argument('--experiment_id', type=int, default=None, help='Specific experiment ID to run (1-10)')
    parser.add_argument('--num_experiments', type=int, default=10, help='Number of parallel experiments to run')
    parser.add_argument('--output_dir', type=str, default="/mnt/home/yuankeji/RanceLab/reticula_new/reticula/data/gtex/GNN/k_fold_val/", help='Output directory')
    
    # New arguments for validation script
    parser.add_argument('--fold', type=int, help='Fold number for validation')
    parser.add_argument('--input_dir', type=str, help='Input directory containing validation data files')
    parser.add_argument('--edges_fn', type=str, help='Path to edges file for validation')
    parser.add_argument('--model_fn', type=str, help='Path to pretrained model file')
    parser.add_argument('--epochs', type=int, help='Number of epochs for training/validation')
    
    global args
    args = parser.parse_args()
    
    # If fold is provided, run validation on that fold
    if args.fold is not None:
        validate_model_on_fold(
            fold=args.fold,
            input_dir=args.input_dir,
            edges_fn=args.edges_fn,
            model_fn=args.model_fn,
            output_dir=args.output_dir,
            epochs=args.epochs
        )
        return
    
    # If experiment_id is provided, run that specific experiment
    if args.experiment_id is not None:
        main_single_experiment(args.experiment_id, args.output_dir)
        return
    
if __name__ == "__main__":
    main()