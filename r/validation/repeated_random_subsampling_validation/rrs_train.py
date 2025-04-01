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
from sklearn.metrics import confusion_matrix, matthews_corrcoef, precision_recall_fscore_support
import pandas as pd
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GraphConv, global_mean_pool
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import concurrent.futures
import threading

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Default hyperparameters
INPUT_CHANNELS = 1
HIDDEN_CHANNELS = 64
BATCH_SIZE = 64
EPOCHS = 500
TEST_RATIO = 0.2

# File paths
edges_fn = '/mnt/home/yuankeji/RanceLab/reticula_new/reticula/data/gtex/input/edges_all.txt'
node_features_fn = '/mnt/home/yuankeji/RanceLab/reticula_new/reticula/data/gtex/input/node_features_all.txt'
graph_targets_fn = '/mnt/home/yuankeji/RanceLab/reticula_new/reticula/data/gtex/input/graph_targets_all.txt'

def split_data(edges_fn, node_features_fn, graph_targets_fn, test_ratio=0.2, run_id=1, output_dir="/mnt/home/yuankeji/RanceLab/reticula_new/reticula/data/gtex/input/"):
    """
    Randomly split the dataset while maintaining sample correspondence across the three files.
    
    Args:
        edges_fn (str): Path to the edges file.
        node_features_fn (str): Path to the node features file.
        graph_targets_fn (str): Path to the graph targets file.
        test_ratio (float): Test set ratio.
        output_dir (str): Directory to save the split datasets.
        run_id (int): Unique identifier for the current run, used to differentiate output files.
    
    Returns:
        tuple: Training and test set data.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Read data using the correct methods
    # Read node features
    features = np.loadtxt(node_features_fn)
    num_samples = features.shape[0]
    
    # Read graph targets 
    targets = np.loadtxt(graph_targets_fn, dtype=str, delimiter=",")
    
    # Read edges data
    def read_reactome_graph(edges_fn):
        edge_v1 = []
        edge_v2 = []
        for line in open(edges_fn, 'r'):
            data = line.split()
            node1 = int(data[0]) - 1  # subtracting to convert R idx to python idx
            node2 = int(data[1]) - 1  # " "
            edge_v1.append(node1)
            edge_v2.append(node2)
        return edge_v1, edge_v2
    
    edge_v1, edge_v2 = read_reactome_graph(edges_fn)
    
    # 2. Create and shuffle sample indices
    indices = list(range(num_samples))
    np.random.seed(run_id)  # Use run_id as random seed for reproducibility
    np.random.shuffle(indices)
    
    # 3. Split dataset using shuffled indices
    test_size = int(test_ratio * num_samples)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    # 4. Split and save the features
    train_features = features[train_indices]
    test_features = features[test_indices]
    np.savetxt(os.path.join(output_dir, f"train_features_{run_id}.txt"), train_features)
    np.savetxt(os.path.join(output_dir, f"test_features_{run_id}.txt"), test_features)
    
    # 5. Split and save the targets
    train_targets = targets[train_indices]
    test_targets = targets[test_indices]
    np.savetxt(os.path.join(output_dir, f"train_targets_{run_id}.txt"), train_targets, fmt='%s', delimiter=",")
    np.savetxt(os.path.join(output_dir, f"test_targets_{run_id}.txt"), test_targets, fmt='%s', delimiter=",")
    
    # 6. Split and save the edges
    # Since edges represent relationships between nodes, we'll just save the original edges file
    # for both train and test sets, as this is the common approach with graph data
    with open(os.path.join(output_dir, f"train_edges_{run_id}.txt"), 'w') as f:
        for v1, v2 in zip(edge_v1, edge_v2):
            # Converting back to R indexing (1-based) when saving
            f.write(f"{v1+1} {v2+1}\n")
    
    with open(os.path.join(output_dir, f"test_edges_{run_id}.txt"), 'w') as f:
        for v1, v2 in zip(edge_v1, edge_v2):
            # Converting back to R indexing (1-based) when saving
            f.write(f"{v1+1} {v2+1}\n")
    
    # 7. Return the split data
    return (
        (edge_v1, edge_v2, train_features, train_targets),
        (edge_v1, edge_v2, test_features, test_targets)
    )


def build_reactome_graph_datalist(edge_v1, edge_v2, features, targets):
    edge_index = torch.tensor([edge_v1, edge_v2], dtype=torch.long)
    
    target_encoder = sklearn.preprocessing.LabelEncoder()
    target_v = target_encoder.fit_transform(targets)
    num_classes = len(np.unique(target_v))
    
    data_list = []
    for row_idx in range(len(features)):
        x = torch.tensor(features[row_idx], dtype=torch.float).unsqueeze(1)
        y = torch.tensor(target_v[row_idx], dtype=torch.long)
        data_list.append(Data(x=x, y=y, edge_index=edge_index))
    
    return data_list, num_classes

def build_reactome_graph_loader(data_list, batch_size):
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)
    return loader

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

def train(loader, model, optimizer, criterion, device, log_file):
    model.train()
    for batch in loader:
        x = batch.x.to(device)
        e = batch.edge_index.to(device)
        b = batch.batch.to(device)
        y = batch.y.to(device)
        out = model(x, e, b)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test(loader, model, criterion, device, log_file):
    model.eval()
    correct = 0
    total_loss = 0
    for batch in loader:
        x = batch.x.to(device)
        e = batch.edge_index.to(device)
        b = batch.batch.to(device)
        y = batch.y.to(device)
        out = model(x, e, b)
        loss = criterion(out, y)
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += int((pred == y).sum())
    return correct / len(loader.dataset), total_loss / len(loader)

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

def run_trial(trial_id, seed):
    set_seed(seed)
    output_dir = f"/mnt/home/yuankeji/RanceLab/validation/rrs-train/trial_{trial_id}"
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "log.txt")
    with open(log_file, "w") as f:
        f.write(f"Trial {trial_id} started at {datetime.datetime.now()}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Parameters: INPUT_CHANNELS={INPUT_CHANNELS}, HIDDEN_CHANNELS={HIDDEN_CHANNELS}, "
                f"BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}\n")
        f.flush()

    train_data, test_data = split_data(edges_fn, node_features_fn, graph_targets_fn, TEST_RATIO, trial_id)
    train_data_list, num_classes = build_reactome_graph_datalist(
        train_data[0],  # edge_v1
        train_data[1],  # edge_v2
        train_data[2],  # train_features
        train_data[3]   # train_targets
    )
    train_loader = build_reactome_graph_loader(train_data_list, BATCH_SIZE)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNN(hidden_channels=HIDDEN_CHANNELS, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    acc_str = ''
    for epoch in range(EPOCHS):
        train(train_loader, model, optimizer, criterion, device, log_file)
        train_acc, train_loss = test(train_loader, model, criterion, device, log_file)
        acc_str += f'{train_acc:.4f}\n'
        with open(log_file, "a") as f:
            f.write(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}\n')
            f.flush()
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}')

    training_acc_fn = f"graph_classification_acc_trial_{trial_id}.txt"
    path = os.path.join(output_dir, training_acc_fn)
    with open(path, 'w') as writefile:
        writefile.write(acc_str)

    model_save_name = f"trained_pytorch_model_trial_{trial_id}.pt"
    path = os.path.join(output_dir, model_save_name)
    torch.save(model.state_dict(), path)
    with open(log_file, "a") as f:
        f.write(f"model saved as {path}\n")
        f.flush()
    print(f"model saved as {path}")

    with open(log_file, "a") as f:
        f.write(f"Trial {trial_id} completed at {datetime.datetime.now()}\n")
        f.flush()
    
    
    
    target_encoder = sklearn.preprocessing.LabelEncoder()
    target_v = np.array([target.strip() for target in train_data[3]])
    target_encoder.fit(target_v)
    class_mapping = {i: label for i, label in enumerate(target_encoder.classes_)}

    test_data_list, _ = build_reactome_graph_datalist(test_data[0], test_data[1], test_data[2], test_data[3])
    test_loader = build_reactome_graph_loader(test_data_list, BATCH_SIZE)
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch.x.to(device)
            e = batch.edge_index.to(device)
            b = batch.batch.to(device)
            out = model(x, e, b)
            preds = out.argmax(dim=1).cpu().numpy()
            targets = batch.y.cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets)

    metrics = calculate_metrics(np.array(all_targets), np.array(all_preds), class_mapping)

    df = pd.DataFrame(metrics)
    metrics_file = os.path.join(output_dir, f"metrics_trial_{trial_id}.csv")
    df.to_csv(metrics_file, index=False)
    with open(log_file, "a") as f:
        f.write(f"Metrics saved to {metrics_file}\n")
    print(f"Metrics saved to {metrics_file}")

    return {"trial_id": trial_id, "status": "completed"}

def main():
    parser = argparse.ArgumentParser(description="GNN model training with multiple trials")
    parser.add_argument("--num_trials", type=int, default=10, help="Number of trials to run")
    parser.add_argument("--base_seed", type=int, default=42, help="Base seed for trials")
    parser.add_argument("--trial_id", type=int, default=None, help="Specific trial ID to run (for SLURM array jobs)")
    args = parser.parse_args()

    if args.trial_id is not None:
        seed = args.base_seed + args.trial_id
        result = run_trial(args.trial_id, seed)
        print(f"Trial {args.trial_id} completed with status: {result['status']}")
    else:
        results = []
        for trial_id in range(args.num_trials):
            seed = args.base_seed + trial_id
            result = run_trial(trial_id, seed)
            results.append(result)
            print(f"Trial {trial_id} completed with status: {result['status']}")

        print(f"All {args.num_trials} trials completed")

if __name__ == "__main__":
    main()