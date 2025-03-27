import torch
import numpy as np
import sklearn
import random
import json
import os
import datetime
import torch.nn.functional as F
from torch.nn import Linear
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, matthews_corrcoef, precision_recall_fscore_support, adjusted_rand_score
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
output_dir2 = "/mnt/home/yuankeji/RanceLab/reticula_new/reticula/data/gtex/input/"

def validate_splits(output_dir, run_id):
    test_files = {
        "edges": f"{output_dir}test_edges_{run_id}.txt",
        "features": f"{output_dir}test_features_{run_id}.txt",
        "targets": f"{output_dir}test_targets_{run_id}.txt"
    }
    
    return test_files["edges"], test_files["features"], test_files["targets"]

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_classes):
        super(GNN, self).__init__()

        self.conv1 = GraphConv(INPUT_CHANNELS, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weight)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_weight)
        x = global_mean_pool(x, batch) 
        x = self.lin(x)
        return x
    
def read_reactome_graph(e_fn):
    e_v1 = []
    e_v2 = []

    for line in open(e_fn, 'r'):
        dt = line.split()
        node1 = int(dt[0]) - 1  # subtracting to convert R idx to python idx
        node2 = int(dt[1]) - 1  # " "
        e_v1.append(node1)
        e_v2.append(node2)

    return e_v1, e_v2

def build_reactome_graph_datalist(e_v1, e_v2, n_fn, g_fn):
    edge_index = torch.tensor([e_v1, e_v2], dtype=torch.long)
    
    feature_v = np.loadtxt(n_fn)
    target_v = np.loadtxt(g_fn, dtype=str, delimiter=",")
    num_classes = len(np.unique(target_v))

    target_encoder = sklearn.preprocessing.LabelEncoder()
    target_v = target_encoder.fit_transform(target_v)

    d_list = []
    for row_idx in range(len(feature_v)):
        features = feature_v[row_idx, :]
        x = torch.tensor(features, dtype=torch.float)
        x = x.unsqueeze(1)
        y = torch.tensor([target_v[row_idx]])
        d_list.append(Data(x=x, y=y, edge_index=edge_index))

    return d_list, num_classes


def build_reactome_graph_loader(d_list, batch_size):
    loader = DataLoader(d_list, batch_size=batch_size, shuffle=True)
    return loader


def train(loader, model, optimizer, criterion, dv):
    model.train()

    correct = 0
    for batch in loader:  # Iterate in batches over the training dataset.
        x = batch.x.to(dv)
        e = batch.edge_index.to(dv)
        b = batch.batch.to(dv)
        y = batch.y.to(dv)
        out = model(x, e, b)  # Perform a single forward pass.
        loss = criterion(out, y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


def test(loader, model, criterion, dv):
    model.eval()

    targets = []
    predictions = []
    for batch in loader:  # Iterate in batches over the test dataset.
        x = batch.x.to(dv)
        e = batch.edge_index.to(dv)
        b = batch.batch.to(dv)
        y = batch.y.to(dv)
        targets += torch.Tensor.tolist(y)
        out = model(x, e, b)  # Perform a single forward pass.
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        predictions += torch.Tensor.tolist(pred)
    
    # Save raw predictions
    np.savetxt(output_dir2, np.transpose([targets, predictions]),
               fmt='%d', delimiter='\t', header='target\tprediction')
    
    # Calculate ARI
    ari = adjusted_rand_score(targets, predictions)
    print(f'ari: {ari}')
    
    # Get class mapping (assuming it's available from the training data)
    target_encoder = sklearn.preprocessing.LabelEncoder()
    target_encoder.classes_ = np.unique(targets)
    class_mapping = {i: str(cls) for i, cls in enumerate(target_encoder.classes_)}
    
    # Calculate detailed metrics
    metrics_results = calculate_metrics(np.array(targets), np.array(predictions), class_mapping)
    
    # Save metrics results to CSV
    metrics_df = pd.DataFrame(metrics_results)
    metrics_file = os.path.join(os.path.dirname(output_dir2), "rrs_detailed_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)
    
    # Print summary
    print(f"Detailed metrics saved to {metrics_file}")
    
    return ari

# from https://stackoverflow.com/questions/12150872/change-key-in-ordereddict-without-losing-order
def change_key(self, old, new):
    for _ in range(len(self)):
        k, v = self.popitem(False)
        self[new if old == k else k] = v

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

    output_dir = f"rrs_trial_{trial_id}"
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "log.txt")

    with open(log_file, "w") as f:
        f.write(f"Trial {trial_id} started at {datetime.datetime.now()}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Parameters: INPUT_CHANNELS={INPUT_CHANNELS}, HIDDEN_CHANNELS={HIDDEN_CHANNELS}, "
                f"BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}\n")
    edges_fn, node_features_fn, graph_targets_fn = validate_splits(output_dir2, trial_id)
    (edge_v1, edge_v2) = read_reactome_graph(edges_fn)
    
    device = cpu = torch.device('cpu')
    data_list, num_classes = build_reactome_graph_datalist(edge_v1, edge_v2, node_features_fn, graph_targets_fn)
    model = GNN(hidden_channels=HIDDEN_CHANNELS, num_classes=num_classes)

    target_v = np.loadtxt(graph_targets_fn, dtype=str, delimiter=",")
    target_encoder = sklearn.preprocessing.LabelEncoder()
    target_encoder.fit(target_v)
    class_mapping = {i: str(cls) for i, cls in enumerate(target_encoder.classes_)}


    model.lin = Linear(HIDDEN_CHANNELS, num_classes)

    model_fn = f'/mnt/scratch/yuankeji/RanceLab/reticula_new/gtex/output/exp_${trial_id}/trained_pytorch_model_trial_{trial_id}.pt'

    sd = torch.load(model_fn, map_location=device)
    change_key(sd, 'conv1.lin_l.weight', 'conv1.lin_rel.weight')
    change_key(sd, 'conv1.lin_l.bias', 'conv1.lin_rel.bias')
    change_key(sd, 'conv1.lin_r.weight', 'conv1.lin_root.weight')
    change_key(sd, 'conv2.lin_l.weight', 'conv2.lin_rel.weight')
    change_key(sd, 'conv2.lin_l.bias', 'conv2.lin_rel.bias')
    change_key(sd, 'conv2.lin_r.weight', 'conv2.lin_root.weight')
    change_key(sd, 'conv3.lin_l.weight', 'conv3.lin_rel.weight')
    change_key(sd, 'conv3.lin_l.bias', 'conv3.lin_rel.bias')
    change_key(sd, 'conv3.lin_r.weight', 'conv3.lin_root.weight')
    change_key(sd, 'lin.weight', 'lin.weight')
    change_key(sd, 'lin.bias', 'lin.bias')
    sd.pop('lin.weight',None)
    sd.pop('lin.bias',None)
    
    model.load_state_dict(sd, strict=False)
    model.eval()

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

    optimizer = torch.optim.AdamW(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    train_data_list = data_list[0::2]
    train_data_loader = build_reactome_graph_loader(train_data_list, BATCH_SIZE)
    for epoch in range(EPOCHS):
        train_acc = train(model, train_data_loader, optimizer, criterion, device)
        print(f'Epoch: {epoch}, Train Acc: {train_acc}')
        if train_acc == 1.0:
            break

    test_data_list = data_list[1::2]
    print(f'Number of test graphs: {len(test_data_list)}')

    test_data_loader = build_reactome_graph_loader(test_data_list, BATCH_SIZE)
    test_ari = test(test_data_loader, model, criterion, device)
    print(f'test_ari: {test_ari}')
    
    # Save the class mapping to a file
    mapping_file = os.path.join(output_dir, "class_mapping.json")
    with open(mapping_file, "w") as f:
        json.dump(class_mapping, f)
    print(f"Class mapping saved to {mapping_file}")

    model_save_name = f'rrs_val_model.pt'
    path = f'/mnt/home/yuankeji/RanceLab/reticula_new/reticula/data/tcga/GNN/{model_save_name}'
    torch.save(model.state_dict(), path)

    with open(log_file, "a") as f:
        f.write(f"Trial {trial_id} completed at {datetime.datetime.now()}\n")
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
    print(f"All {args.num_trials} trials completed")

if __name__ == "__main__":
    main()
