import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import numpy as np

tissue_mapping = {
    "0": "Adipose", "1": "Adrenal", "2": "Aorta", "3": "Bone", "4": "Brain",
    "5": "Eye", "6": "Heart", "7": "Intestine", "8": "Kidney", "9": "Liver",
    "10": "Lung", "11": "MOE", "12": "Mammary gland", "13": "Muscle", "14": "Nerve",
    "15": "Olfactory", "16": "Ovary", "17": "Pancreas", "18": "Skin", "19": "Sperm",
    "20": "Spinal cord", "21": "Stomach", "22": "Tendon", "23": "Testes", "24": "Thymus",
    "25": "Tongue", "26": "Uterus", "27": "VNO"
}

tissue_counts_from_log = {
    "Adipose": 119,
    "Adrenal": 5,
    "Aorta": 9,
    "Bone": 8,
    "Brain": 340,
    "Eye": 33,
    "Heart": 41,
    "Intestine": 118,
    "Kidney": 88,
    "Liver": 432,
    "Lung": 63,
    "MOE": 26,
    "Mammary gland": 6,
    "Muscle": 73,
    "Nerve": 8,
    "Olfactory": 7,
    "Ovary": 17,
    "Pancreas": 45,
    "Skin": 29,
    "Sperm": 13,
    "Spinal cord": 19,
    "Stomach": 7,
    "Tendon": 7,
    "Testes": 44,
    "Thymus": 5,
    "Tongue": 6,
    "Uterus": 11,
    "VNO": 12,
}

training_tissue_counts = {
    "Adipose": 476,
    "Adrenal": 18,
    "Aorta": 34,
    "Bone": 30,
    "Brain": 1356,
    "Eye": 132,
    "Heart": 163,
    "Intestine": 472,
    "Kidney": 348,
    "Liver": 1727,
    "Lung": 252,
    "MOE": 101,
    "Mammary gland": 21,
    "Muscle": 289,
    "Nerve": 31,
    "Olfactory": 26,
    "Ovary": 66,
    "Pancreas": 177,
    "Skin": 112,
    "Sperm": 51,
    "Spinal cord": 73,
    "Stomach": 28,
    "Tendon": 25,
    "Testes": 174,
    "Thymus": 17,
    "Tongue": 21,
    "Uterus": 44,
    "VNO": 48,
}

tissue_counts_from_log = {tissue: tissue_counts_from_log.get(tissue, 0) + training_tissue_counts.get(tissue, 0)
                           for tissue in set(tissue_counts_from_log) | set(training_tissue_counts)}

all_data = []

for i in range(1, 11):
    file_name = f"/Users/kejiyuan/Desktop/gnn_validation/k-fold-gnn-val/k_fold_detailed_metrics_{i}.csv"
    try:
        df = pd.read_csv(file_name)
        df["tissue_code"] = df["tissue"].astype(str)
        df["tissue"] = df["tissue_code"].map(tissue_mapping)
        all_data.append(df)
    except FileNotFoundError:
        print(f"File {file_name} not found.")

combined_data = pd.concat(all_data, ignore_index=True)

unique_tissues_sorted = sorted(combined_data['tissue'].unique())

xticklabels = [f"{tissue}/{tissue_counts_from_log.get(tissue, 'N/A')}"
               for tissue in unique_tissues_sorted]

plt.figure(figsize=(16, 8))
sns.boxplot(x="tissue", y="F1_score", data=combined_data, order=unique_tissues_sorted)
plt.xticks(rotation=90, labels=xticklabels, ticks=np.arange(len(unique_tissues_sorted)), fontsize=12, fontweight='bold', ha='center')
plt.yticks(fontsize=12, fontweight='bold')
plt.title("F1 Score Distribution by Tissue in Stratified K-fold Validation", fontsize=16)
plt.xlabel("Count/Tissue", fontsize=16)
plt.ylabel("F1 Score", fontsize=16)
plt.tight_layout()
plt.savefig("f1_score_by_tissue_skv_with_counts_violin.pdf")
plt.show()