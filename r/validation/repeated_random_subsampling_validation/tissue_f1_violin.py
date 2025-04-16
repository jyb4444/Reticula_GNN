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

tissue_mean_counts = {
    "Adipose": 116,
    "Adrenal": 5,
    "Aorta": 10,
    "Bone": 8,
    "Brain": 332,
    "Eye": 34,
    "Heart": 38,
    "Intestine": 112,
    "Kidney": 86,
    "Liver": 436,
    "Lung": 62,
    "MOE": 26,
    "Mammary gland": 6,
    "Muscle": 76,
    "Nerve": 10,
    "Olfactory": 8,
    "Ovary": 17,
    "Pancreas": 46,
    "Skin": 28,
    "Sperm": 12,
    "Spinal cord": 18,
    "Stomach": 9,
    "Tendon": 7,
    "Testes": 46,
    "Thymus": 4,
    "Tongue": 6,
    "Uterus": 11,
    "VNO": 11,
}

training_rrs_counts = {
    "Adipose": 479,
    "Adrenal": 18,
    "Aorta": 33,
    "Bone": 30,
    "Brain": 1364,
    "Eye": 131,
    "Heart": 166,
    "Intestine": 478,
    "Kidney": 350,
    "Liver": 1723,
    "Lung": 253,
    "MOE": 101,
    "Mammary gland": 21,
    "Muscle": 286,
    "Nerve": 29,
    "Olfactory": 25,
    "Ovary": 66,
    "Pancreas": 176,
    "Skin": 112,
    "Sperm": 52,
    "Spinal cord": 74,
    "Stomach": 26,
    "Tendon": 25,
    "Testes": 172,
    "Thymus": 18,
    "Tongue": 22,
    "Uterus": 44,
    "VNO": 49,
}

tissue_mean_counts = {tissue: tissue_mean_counts.get(tissue, 0) + training_rrs_counts.get(tissue, 0)
                       for tissue in set(tissue_mean_counts) | set(training_rrs_counts)}


all_data = []

for i in range(1, 11):
    file_name = f"/Users/kejiyuan/Desktop/gnn_validation/rrs-val/rrs_detailed_metrics_{i}.csv"  # 替换为您的文件路径
    try:
        df = pd.read_csv(file_name)
        df["tissue_code"] = df["tissue"].astype(str)
        df["tissue"] = df["tissue_code"].map(tissue_mapping)
        all_data.append(df)
    except FileNotFoundError:
        print(f"File {file_name} not found.")

combined_data = pd.concat(all_data, ignore_index=True)

unique_tissues_sorted = sorted(combined_data['tissue'].unique())

xticklabels = [f"{tissue}/{tissue_mean_counts.get(tissue, 'N/A')}"
               for tissue in unique_tissues_sorted]

plt.figure(figsize=(16, 8))
sns.boxplot(x="tissue", y="F1_score", data=combined_data, order=unique_tissues_sorted)
plt.xticks(rotation=90, labels=xticklabels, ticks=np.arange(len(unique_tissues_sorted)), fontsize=12, fontweight='bold', ha='center')
plt.yticks(fontsize=12, fontweight='bold')
plt.title("F1 Score Distribution by Tissue in Repeated Random Subsampling Validation", fontsize=16)
plt.xlabel("Count/Tissue", fontsize=16)
plt.ylabel("F1 Score", fontsize=16)
plt.tight_layout()
plt.savefig("f1_score_by_tissue_rrs_with_counts_boxplot.pdf")
plt.show()