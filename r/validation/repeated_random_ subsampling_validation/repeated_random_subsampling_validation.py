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

INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 26
HIDDEN_CHANNELS = 64
BATCH_SIZE = 64
EPOCHS = 500
TEST_RATIO = 0.2  

edges_fn = '/mnt/home/yuankeji/RanceLab/reticula_new/reticula/data/gtex/input/edges_all.txt'
node_features_fn = '/mnt/home/yuankeji/RanceLab/reticula_new/reticula/data/gtex/input/node_features_all.txt'
graph_targets_fn = '/mnt/home/yuankeji/RanceLab/reticula_new/reticula/data/gtex/input/graph_targets_all.txt'

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
        x = global_mean_pool(x, batch)  # 池化
        x = F.dropout(x, training=self.training)
        return self.lin(x)

def train(loader, model, optimizer, criterion, device, epoch, total_epochs, experiment_id, log_file):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    desc = f"Exp {experiment_id} - Epoch {epoch+1}/{total_epochs} [Train]"
    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        optimizer.zero_grad()
        x, edge_index, batch_idx, y = batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device), batch.y.to(device)
        out = model(x, edge_index, batch_idx)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        total_loss += loss.item()
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{correct/total:.4f}"})
    
    avg_loss = total_loss / len(loader)
    acc = correct / total
    return avg_loss, acc

def test(loader, model, device, experiment_id, desc, log_file):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    targets = []
    
    pbar = tqdm(loader, desc=f"Exp {experiment_id} - {desc}", leave=False)
    with torch.no_grad():
        for batch in pbar:
            x, edge_index, batch_idx, y = batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device), batch.y.to(device)
            out = model(x, edge_index, batch_idx)
            pred = out.argmax(dim=1)
            
            correct += (pred == y).sum().item()
            total += y.size(0)
            
            predictions.extend(pred.cpu().numpy())
            targets.extend(y.cpu().numpy())
            
            acc = correct / total
            pbar.set_postfix({'acc': f"{acc:.4f}"})
    
    return acc, np.array(predictions), np.array(targets)

def calculate_per_class_metrics(y_true, y_pred, class_mapping, output_dir, experiment_id):
    """计算每个组织类型的详细指标"""
    num_classes = len(class_mapping)
    
    results = []
    
    for class_idx in range(num_classes):
        class_name = class_mapping[class_idx]
        
        y_true_binary = (y_true == class_idx).astype(int)
        y_pred_binary = (y_pred == class_idx).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]).ravel()
        
        positives = tp + fn
        
        if positives == 0:
            tpr = 0
            fnr = 0
        else:
            tpr = tp / positives  
            fnr = fn / positives
        
        negatives = tn + fp
        if negatives == 0:
            tnr = 0
            fpr = 0
        else:
            tnr = tn / negatives 
            fpr = fp / negatives
        
        if (tp + fp) == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)
        
        if (tp + fn) == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)  
        
        if (precision + recall) == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        if (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) == 0:
            mcc = 0
        else:
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
    
    df = pd.DataFrame(results)
    output_file = os.path.join(output_dir, f"tissue_metrics_experiment_{experiment_id}.csv")
    df.to_csv(output_file, index=False)
    
    print(f"Per-tissue metrics saved to {output_file}")
    
    return df

def run_experiment(experiment_id, data_list, num_classes, class_mapping, model_save_dir, gpu_id=None):
    log_file = os.path.join(model_save_dir, f"experiment_{experiment_id}_log.txt")
    
    seed = 42 + experiment_id
    set_seed(seed)
    
    if gpu_id is not None:
        device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    log_message(f"=== Starting experiment {experiment_id} ===", log_file)
    log_message(f"Using device: {device}", log_file)
    log_message(f"Using seed: {seed}", log_file)
    
    n_samples = len(data_list)
    indices = np.random.permutation(n_samples)
    test_size = int(TEST_RATIO * n_samples)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    train_data_list = [data_list[i] for i in train_indices]
    test_data_list = [data_list[i] for i in test_indices]
    
    log_message(f"Experiment {experiment_id}: {len(train_data_list)} train samples, {len(test_data_list)} test samples", log_file)
    
    train_loader = DataLoader(train_data_list, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data_list, batch_size=BATCH_SIZE, shuffle=False)
    
    model = GNN(hidden_channels=HIDDEN_CHANNELS, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_test_acc = 0.0
    best_epoch = 0
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train(train_loader, model, optimizer, criterion, device, epoch, EPOCHS, experiment_id, log_file)
        test_acc, _, _ = test(test_loader, model, device, experiment_id, f"Epoch {epoch+1}/{EPOCHS} [Test]", log_file)
        
        log_message(f"Experiment {experiment_id}, Epoch {epoch+1}/{EPOCHS}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}", log_file)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            model_save_path = os.path.join(model_save_dir, f"best_model_experiment_{experiment_id}.pt")
            torch.save(model.state_dict(), model_save_path)
            log_message(f"Experiment {experiment_id}: New best model saved at {model_save_path}", log_file)
    
    model.load_state_dict(torch.load(os.path.join(model_save_dir, f"best_model_experiment_{experiment_id}.pt")))
    final_test_acc, predictions, targets = test(test_loader, model, device, experiment_id, f"Final evaluation", log_file)
    
    from sklearn.metrics import classification_report, confusion_matrix
    
    confusion = confusion_matrix(targets, predictions)
    report = classification_report(targets, predictions, target_names=[class_mapping[i] for i in range(num_classes)])
    
    with open(os.path.join(model_save_dir, f"results_experiment_{experiment_id}.txt"), 'w') as f:
        f.write(f"Best epoch: {best_epoch+1}\n")
        f.write(f"Best test accuracy: {best_test_acc:.4f}\n")
        f.write(f"Final test accuracy: {final_test_acc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(confusion) + "\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    tissue_metrics = calculate_per_class_metrics(targets, predictions, class_mapping, model_save_dir, experiment_id)
    
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    log_message(f"Experiment {experiment_id} completed in {int(hours)}h {int(minutes)}m {int(seconds)}s with best test accuracy: {best_test_acc:.4f} at epoch {best_epoch+1}", log_file)
    
    return best_test_acc, tissue_metrics

def combine_tissue_metrics(all_tissue_metrics, class_mapping, model_save_dir):
    if not all_tissue_metrics:
        return
    
    all_metrics_df = pd.concat([df.assign(experiment=i+1) for i, df in enumerate(all_tissue_metrics)])
    
    avg_metrics = all_metrics_df.groupby('tissue').agg({
        'TP': 'mean',
        'TN': 'mean',
        'FP': 'mean',
        'FN': 'mean',
        'POSITIVES': 'mean',
        'MCC': 'mean',
        'TPR': 'mean',
        'FPR': 'mean',
        'TNR': 'mean',
        'FNR': 'mean',
        'Precision': 'mean',
        'Recall': 'mean',
        'F1_score': 'mean'
    }).reset_index()
    
    std_metrics = all_metrics_df.groupby('tissue').agg({
        'MCC': 'std',
        'TPR': 'std',
        'FPR': 'std',
        'TNR': 'std',
        'FNR': 'std',
        'Precision': 'std',
        'Recall': 'std',
        'F1_score': 'std'
    }).reset_index()
    
    std_metrics.columns = ['tissue'] + [f'{col}_std' for col in std_metrics.columns if col != 'tissue']
    
    final_metrics = pd.merge(avg_metrics, std_metrics, on='tissue')
    
    final_metrics.to_csv(os.path.join(model_save_dir, "tissue_metrics_all_experiments.csv"), index=False)
    print(f"Combined tissue metrics saved to {os.path.join(model_save_dir, 'tissue_metrics_all_experiments.csv')}")
    
    readable_df = final_metrics.copy()
    for col in ['TPR', 'TNR', 'FPR', 'FNR', 'Precision', 'Recall', 'F1_score', 'MCC']:
        readable_df[col] = readable_df[col].apply(lambda x: f"{x*100:.2f}%" if col != 'MCC' else f"{x:.4f}")
        if f'{col}_std' in readable_df.columns:
            readable_df[f'{col}_std'] = readable_df[f'{col}_std'].apply(lambda x: f"±{x*100:.2f}%" if col != 'MCC' else f"±{x:.4f}")
    
    readable_df.to_csv(os.path.join(model_save_dir, "tissue_metrics_all_experiments_readable.csv"), index=False)
    print(f"Human-readable tissue metrics saved to {os.path.join(model_save_dir, 'tissue_metrics_all_experiments_readable.csv')}")
    
    performance_df = final_metrics[['tissue', 'F1_score', 'MCC', 'Precision', 'Recall']]
    performance_df = performance_df.sort_values('F1_score', ascending=False)
    performance_df.to_csv(os.path.join(model_save_dir, "tissue_performance_ranking.csv"), index=False)
    print(f"Tissue performance ranking saved to {os.path.join(model_save_dir, 'tissue_performance_ranking.csv')}")

def main():
    parser = argparse.ArgumentParser(description='GNN with Parallel Random Subsampling Validation')
    parser.add_argument('--num_experiments', type=int, default=10, help='Number of parallel experiments to run')
    parser.add_argument('--output_dir', type=str, default="/mnt/home/yuankeji/RanceLab/reticula_new/reticula/data/gtex/GNN", help='Output directory')
    args = parser.parse_args()
    
    model_save_dir = args.output_dir
    os.makedirs(model_save_dir, exist_ok=True)
    
    main_log_file = os.path.join(model_save_dir, "training_log.txt")
    
    start_time = time.time()
    log_message("Starting GNN training with parallel random subsampling validation", main_log_file)
    
    edge_v1, edge_v2 = read_reactome_graph(edges_fn, node_features_fn, main_log_file)
    data_list, labels, target_encoder, num_classes, class_mapping = build_reactome_graph_datalist(edge_v1, edge_v2, node_features_fn, graph_targets_fn, main_log_file)
    
    mapping_file = os.path.join(model_save_dir, "class_mapping.csv")
    with open(mapping_file, 'w') as f:
        f.write("class_id,tissue_name\n")
        for class_id, tissue_name in class_mapping.items():
            f.write(f"{class_id},{tissue_name}\n")
    log_message(f"Class mapping saved to {mapping_file}", main_log_file)
    
    log_message(f"Number of classes: {num_classes}", main_log_file)
    
    num_experiments = args.num_experiments
    log_message(f"Running {num_experiments} parallel experiments", main_log_file)
    
    num_gpus = torch.cuda.device_count()
    log_message(f"Available GPUs: {num_gpus}", main_log_file)
    
    all_results = []
    all_tissue_metrics = []
    
    if num_gpus > 1:
        log_message(f"Running experiments on {num_gpus} GPUs", main_log_file)
        
        gpu_assignments = [i % num_gpus for i in range(num_experiments)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_experiments) as executor:
            future_to_experiment = {
                executor.submit(run_experiment, i+1, data_list, num_classes, class_mapping, model_save_dir, gpu_assignments[i]): i+1 
                for i in range(num_experiments)
            }
            
            for future in concurrent.futures.as_completed(future_to_experiment):
                experiment_id = future_to_experiment[future]
                try:
                    result, tissue_metrics = future.result()
                    all_results.append(result)
                    all_tissue_metrics.append(tissue_metrics)
                    log_message(f"Experiment {experiment_id} completed with accuracy: {result:.4f}", main_log_file)
                except Exception as exc:
                    log_message(f"Experiment {experiment_id} generated an exception: {exc}", main_log_file)
    else:
        log_message("Running experiments sequentially on a single GPU", main_log_file)
        for i in range(num_experiments):
            result, tissue_metrics = run_experiment(i+1, data_list, num_classes, class_mapping, model_save_dir)
            all_results.append(result)
            all_tissue_metrics.append(tissue_metrics)
    
    mean_acc = np.mean(all_results)
    std_acc = np.std(all_results)
    log_message(f"=== Final Results ===", main_log_file)
    log_message(f"Mean Test Accuracy: {mean_acc:.4f} ± {std_acc:.4f}", main_log_file)
    
    with open(os.path.join(model_save_dir, "final_results.txt"), 'w') as f:
        f.write(f"=== Final Results ===\n")
        f.write(f"Mean Test Accuracy: {mean_acc:.4f} ± {std_acc:.4f}\n\n")
        f.write("Individual Experiment Results:\n")
        for exp, acc in enumerate(all_results):
            f.write(f"Experiment {exp+1}: {acc:.4f}\n")
    
    combine_tissue_metrics(all_tissue_metrics, class_mapping, model_save_dir)
    
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    log_message(f"Total runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s", main_log_file)

if __name__ == "__main__":
    main()