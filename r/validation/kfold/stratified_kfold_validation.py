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

def read_reactome_graph(edges_fn):
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

def build_reactome_graph_loader(data_list, batch_size, shuffle=False):
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
    all_probs = []
    
    with torch.no_grad():
        for batch in loader:
            x = batch.x.to(device)
            e = batch.edge_index.to(device)
            b = batch.batch.to(device)
            y = batch.y
            
            out = model(x, e, b)
            probs = F.softmax(out, dim=1).cpu().numpy()
            pred = out.argmax(dim=1).cpu().numpy()
            
            predictions.extend(pred)
            true_labels.extend(y.numpy().flatten())
            all_probs.extend(probs)
    
    return np.array(predictions), np.array(true_labels), np.array(all_probs)

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
    output_file = os.path.join(output_dir, f"val_tissue_metrics_fold_{fold_idx}.csv")
    df.to_csv(output_file, index=False)
    
    print(f"Validation per-tissue metrics saved to {output_file}")
    
    return df

def save_detailed_predictions(predictions, true_labels, probabilities, class_mapping, output_dir, fold_idx):
    results = []
    
    for i in range(len(true_labels)):
        result = {
            'true_label': true_labels[i],
            'true_tissue': class_mapping[true_labels[i]],
            'predicted_label': predictions[i],
            'predicted_tissue': class_mapping[predictions[i]],
            'correct': true_labels[i] == predictions[i]
        }
        
        for class_idx, class_name in class_mapping.items():
            result[f'prob_{class_name}'] = probabilities[i][class_idx]
        
        results.append(result)
    
    df = pd.DataFrame(results)
    output_file = os.path.join(output_dir, f"detailed_predictions_fold_{fold_idx}.csv")
    df.to_csv(output_file, index=False)
    
    print(f"Detailed predictions saved to {output_file}")

def compute_confusion_matrix(y_true, y_pred, class_mapping, output_dir, fold_idx):
    num_classes = len(class_mapping)
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    cm_df = pd.DataFrame(cm)
    cm_df.index = [class_mapping[i] for i in range(num_classes)]
    cm_df.columns = [class_mapping[i] for i in range(num_classes)]
    
    output_file = os.path.join(output_dir, f"confusion_matrix_fold_{fold_idx}.csv")
    cm_df.to_csv(output_file)
    
    print(f"Confusion matrix saved to {output_file}")
    
    return cm_df

def main():
    parser = argparse.ArgumentParser(description='Validate GNN model using R-generated fold data')
    parser.add_argument('--fold', type=int, default=0, help='Fold index to use (0-9)')
    parser.add_argument('--input_dir', type=str, default="/mnt/home/yuankeji/RanceLab/reticula_new/reticula/data/gtex/input", 
                        help='Directory with input data created by R script')
    parser.add_argument('--output_dir', type=str, default="/mnt/home/yuankeji/RanceLab/reticula_new/reticula/data/gtex/GNN", 
                        help='Directory to save GNN outputs')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    fold_idx = args.fold
    
    edges_fn = os.path.join(args.input_dir, f"edges_train_{fold_idx}.txt")  # 边结构与训练相同
    node_features_fn = os.path.join(args.input_dir, f"node_features_val_{fold_idx}.txt")
    graph_targets_fn = os.path.join(args.input_dir, f"graph_targets_val_{fold_idx}.txt")
    
    model_path = os.path.join(args.output_dir, f"trained_pytorch_model_fold_{fold_idx}.pt")
    
    print(f"Using fold {fold_idx} for validation")
    print(f"Reading data from:")
    print(f"  Edges file: {edges_fn}")
    print(f"  Node features file: {node_features_fn}")
    print(f"  Graph targets file: {graph_targets_fn}")
    print(f"  Model path: {model_path}")
    
    if not os.path.exists(edges_fn) or not os.path.exists(node_features_fn) or not os.path.exists(graph_targets_fn):
        print("Error: Input files not found. Please check your file paths and ensure R script completed successfully.")
        return
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Please ensure training completed successfully.")
        return
    
    try:
        edge_v1, edge_v2 = read_reactome_graph(edges_fn)
        data_list, target_labels, class_mapping, class_mapping_reverse = build_reactome_graph_datalist(
            edge_v1, edge_v2, node_features_fn, graph_targets_fn)
        
        output_channels = len(class_mapping)
        print(f"Number of tissue classes: {output_channels}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    val_loader = build_reactome_graph_loader(data_list, BATCH_SIZE)
    
    model = GNN(hidden_channels=HIDDEN_CHANNELS, output_channels=output_channels).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    criterion = torch.nn.CrossEntropyLoss()
    
    start_time = time.time()
    
    print("\nStarting validation...")
    
    val_acc, val_loss = test(val_loader, model, criterion, device)
    print(f"Validation Accuracy: {val_acc:.4f}, Validation Loss: {val_loss:.4f}")
    
    predictions, true_labels, probabilities = test_with_predictions(val_loader, model, device)
    
    tissue_metrics = calculate_per_class_metrics(true_labels, predictions, class_mapping, args.output_dir, fold_idx)
    
    confusion_matrix_df = compute_confusion_matrix(true_labels, predictions, class_mapping, args.output_dir, fold_idx)
    
    save_detailed_predictions(predictions, true_labels, probabilities, class_mapping, args.output_dir, fold_idx)
    
    summary_file = os.path.join(args.output_dir, f"validation_summary_fold_{fold_idx}.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Fold: {fold_idx}\n")
        f.write(f"Validation Accuracy: {val_acc:.6f}\n")
        f.write(f"Validation Loss: {val_loss:.6f}\n")
        f.write(f"Number of samples: {len(data_list)}\n")
        f.write(f"Correct predictions: {int(val_acc * len(data_list))}\n")
        
        f.write("\nPer-tissue Performance Summary:\n")
        for class_idx in range(len(class_mapping)):
            row = tissue_metrics[tissue_metrics['tissue'] == class_mapping[class_idx]]
            f.write(f"Tissue: {class_mapping[class_idx]}\n")
            f.write(f"  Precision: {row['Precision'].values[0]:.4f}\n")
            f.write(f"  Recall: {row['Recall'].values[0]:.4f}\n")
            f.write(f"  F1 Score: {row['F1_score'].values[0]:.4f}\n")
    
    end_time = time.time()
    print(f"Validation completed in {end_time - start_time:.2f} seconds")
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()