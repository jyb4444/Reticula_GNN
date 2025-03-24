import torch
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, matthews_corrcoef, precision_recall_fscore_support
import random
import time
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GraphConv, global_mean_pool
import os
import pandas as pd
import argparse

random.seed(88888888)
np.random.seed(88888888)
torch.manual_seed(88888888)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cpu = torch.device('cpu')

INPUT_CHANNELS = 1
HIDDEN_CHANNELS = 64
BATCH_SIZE = 64
EPOCHS = 500
BENCHMARKING = False

def read_reactome_graph(edges_fn):
    """读取边数据"""
    edge_v1 = []
    edge_v2 = []

    for line in open(edges_fn, 'r'):
        data = line.split()
        node1 = int(data[0]) - 1  
        node2 = int(data[1]) - 1
        edge_v1.append(node1)
        edge_v2.append(node2)

    return edge_v1, edge_v2

def build_reactome_graph_datalist(edge_v1, edge_v2, node_features_fn, graph_targets_fn):
    edge_index = torch.tensor([edge_v1, edge_v2], dtype=torch.long)
    feature_v = np.loadtxt(node_features_fn)
    target_v = np.loadtxt(graph_targets_fn, dtype=str, delimiter=",")
    
    target_encoder = sklearn.preprocessing.LabelEncoder()
    target_v = target_encoder.fit_transform(target_v)
    
    class_mapping = {i: label for i, label in enumerate(target_encoder.classes_)}
    class_mapping_reverse = {label: i for i, label in enumerate(target_encoder.classes_)}

    data_list = []
    for row_idx in range(len(feature_v)):
        features = feature_v[row_idx, :]
        x = torch.tensor(features, dtype=torch.float).unsqueeze(1)
        y = torch.tensor([target_v[row_idx]], dtype=torch.long)
        data_list.append(Data(x=x, y=y, edge_index=edge_index))

    return data_list, target_v, class_mapping, class_mapping_reverse

def build_reactome_graph_loader(data_list, batch_size, shuffle=True):
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)
    return loader

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, output_channels):
        super(GNN, self).__init__()

        self.conv1 = GraphConv(INPUT_CHANNELS, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, output_channels)

    def forward(self, x, edge_index, batch, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weight)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_weight)

        x = global_mean_pool(x, batch)  

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

def train(loader, model, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_batches = len(loader)
    
    print(f"Training on {len(loader.dataset)} samples in {total_batches} batches")
    
    for batch_idx, batch in enumerate(loader):
        x = batch.x.to(device)
        e = batch.edge_index.to(device)
        b = batch.batch.to(device)
        y = batch.y.to(device)
        
        out = model(x, e, b)
        loss = criterion(out, y)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        batch_loss = loss.item() * batch.num_graphs
        total_loss += batch_loss
        
        if (batch_idx + 1) % max(1, total_batches // 10) == 0:
            print(f"  Batch {batch_idx + 1}/{total_batches} ({(batch_idx + 1) / total_batches * 100:.1f}%) - Loss: {batch_loss / batch.num_graphs:.4f}")
    
    return total_loss / len(loader.dataset)

def test(loader, model, criterion, device):
    model.eval()
    correct = 0
    total_loss = 0
    
    with torch.no_grad():
        for batch in loader:
            x = batch.x.to(device)
            e = batch.edge_index.to(device)
            b = batch.batch.to(device)
            y = batch.y.to(device)
            
            out = model(x, e, b)
            loss = criterion(out, y)
            
            total_loss += loss.item() * batch.num_graphs
            pred = out.argmax(dim=1)
            correct += int((pred == y).sum())
    
    return correct / len(loader.dataset), total_loss / len(loader.dataset)

def test_with_predictions(loader, model, device):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in loader:
            x = batch.x.to(device)
            e = batch.edge_index.to(device)
            b = batch.batch.to(device)
            y = batch.y
            
            out = model(x, e, b)
            pred = out.argmax(dim=1).cpu().numpy()
            
            predictions.extend(pred)
            true_labels.extend(y.numpy().flatten())
    
    return np.array(predictions), np.array(true_labels)

def calculate_per_class_metrics(y_true, y_pred, class_mapping, output_dir, fold_idx):
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
    output_file_csv = os.path.join(output_dir, f"tissue_metrics_fold_{fold_idx}.csv")
    df.to_csv(output_file_csv, index=False)
    
    output_file_txt = os.path.join(output_dir, f"tissue_metrics_fold_{fold_idx}.txt")
    with open(output_file_txt, 'w') as f:
        f.write(f"Per-tissue Performance Summary:\n")
        
        correct_predictions = sum(y_pred == y_true)
        total_samples = len(y_true)
        f.write(f"Number of samples: {total_samples}\n")
        f.write(f"Correct predictions: {correct_predictions}\n")
        f.write(f"Overall Accuracy: {correct_predictions/total_samples:.4f}\n\n")
        
        for result in results:
            f.write(f"Tissue: \"{result['tissue']}\"\n")
            f.write(f"  Precision: {result['Precision']:.4f}\n")
            f.write(f"  Recall: {result['Recall']:.4f}\n")
            f.write(f"  F1 Score: {result['F1_score']:.4f}\n")
            f.write(f"  MCC: {result['MCC']:.4f}\n")
            f.write(f"  TP: {result['TP']}, FP: {result['FP']}, TN: {result['TN']}, FN: {result['FN']}\n\n")
    
    print(f"Per-tissue metrics saved to {output_file_csv} and {output_file_txt}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Train GNN model using R-generated fold data')
    parser.add_argument('--fold', type=int, default=0, help='Fold index to use (0-9)')
    parser.add_argument('--input_dir', type=str, default="/mnt/home/yuankeji/RanceLab/reticula_new/reticula/data/gtex/input", 
                        help='Directory with input data created by R script')
    parser.add_argument('--output_dir', type=str, default="/mnt/home/yuankeji/RanceLab/reticula_new/reticula/data/gtex/GNN", 
                        help='Directory to save GNN outputs')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    fold_idx = args.fold
    
    edges_fn = os.path.join(args.input_dir, f"edges_train_{fold_idx}.txt")
    node_features_fn = os.path.join(args.input_dir, f"node_features_train_{fold_idx}.txt")
    graph_targets_fn = os.path.join(args.input_dir, f"graph_targets_train_{fold_idx}.txt")
    
    print(f"Using fold {fold_idx} for training")
    print(f"Reading data from:")
    print(f"  Edges file: {edges_fn}")
    print(f"  Node features file: {node_features_fn}")
    print(f"  Graph targets file: {graph_targets_fn}")
    
    if not os.path.exists(edges_fn) or not os.path.exists(node_features_fn) or not os.path.exists(graph_targets_fn):
        print("Error: Input files not found. Please check your file paths and ensure R script completed successfully.")
        return
    
    try:
        edge_v1, edge_v2 = read_reactome_graph(edges_fn)
        data_list, target_labels, class_mapping, class_mapping_reverse = build_reactome_graph_datalist(
            edge_v1, edge_v2, node_features_fn, graph_targets_fn)
        
        output_channels = len(class_mapping)
        print(f"Number of tissue classes: {output_channels}")
        
        mapping_file = os.path.join(args.output_dir, f"class_mapping_fold_{fold_idx}.csv")
        with open(mapping_file, 'w') as f:
            f.write("class_id,tissue_name\n")
            for class_id, tissue_name in class_mapping.items():
                f.write(f"{class_id},{tissue_name}\n")
        print(f"Class mapping saved to {mapping_file}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    train_loader = build_reactome_graph_loader(data_list, BATCH_SIZE)
    
    model = GNN(hidden_channels=HIDDEN_CHANNELS, output_channels=output_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_train_acc = 0
    best_epoch = 0
    train_losses = []
    train_accs = []
    
    start_time = time.time()
    
    print(f"\nStarting training...")
    for epoch in range(EPOCHS):
        train_loss = train(train_loader, model, optimizer, criterion, device)
        train_acc, _ = test(train_loader, model, criterion, device)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            best_epoch = epoch
            model_save_name = f"trained_pytorch_model_fold_{fold_idx}.pt"
            path = os.path.join(args.output_dir, model_save_name)
            torch.save(model.state_dict(), path)
            print(f"Model saved to {path}")
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch + 1}/{EPOCHS}, '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Best Train Acc: {best_train_acc:.4f} (Epoch {best_epoch})')
        
        if train_acc >= 0.999:
            print(f"Reached near-perfect training accuracy of {train_acc:.4f} at epoch {epoch}. Stopping training.")
            break
    
    end_time = time.time()
    
    print(f'\nTraining for fold {fold_idx} completed in {end_time - start_time:.2f}s')
    print(f'Best Train Acc: {best_train_acc:.4f} (Epoch {best_epoch})')
    
    results_fn = f"train_results_fold_{fold_idx}.txt"
    path = os.path.join(args.output_dir, results_fn)
    
    with open(path, 'w') as f:
        f.write('epoch,train_loss,train_acc\n')
        for i in range(len(train_losses)):
            f.write(f'{i},{train_losses[i]:.6f},{train_accs[i]:.6f}\n')
    
    print(f"Training results saved to {path}")

    print(f'\nTraining completed in {end_time - start_time:.2f}s')
    print(f'Best Train Acc: {best_train_acc:.4f} (Epoch {best_epoch})')

    train_predictions, train_true_labels = test_with_predictions(train_loader, model, device)

    train_metrics = calculate_per_class_metrics(
        train_true_labels, 
        train_predictions, 
        class_mapping, 
        args.output_dir, 
        fold_idx=1
    )

if __name__ == "__main__":
    main()