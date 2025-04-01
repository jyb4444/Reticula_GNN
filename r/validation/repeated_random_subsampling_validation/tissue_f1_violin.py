import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json

tissue_mapping = {
    "0": "Adipose", "1": "Adrenal", "2": "Aorta", "3": "Bone", "4": "Brain",
    "5": "Eye", "6": "Heart", "7": "Intestine", "8": "Kidney", "9": "Liver",
    "10": "Lung", "11": "MOE", "12": "Mammary gland", "13": "Muscle", "14": "Nerve",
    "15": "Olfactory", "16": "Ovary", "17": "Pancreas", "18": "Skin", "19": "Sperm",
    "20": "Spinal cord", "21": "Stomach", "22": "Tendon", "23": "Testes", "24": "Thymus",
    "25": "Tongue", "26": "Uterus", "27": "VNO"
}

all_data = []

for i in range(1, 11):
    file_name = f"/Users/kejiyuan/Desktop/gnn_validation/rrs-val/rrs_detailed_metrics_{i}.csv"  # 替换为您的文件路径
    try:
        df = pd.read_csv(file_name)
        df["tissue"] = df["tissue"].astype(str).map(tissue_mapping)
        all_data.append(df)
    except FileNotFoundError:
        print(f"File {file_name} not found.")

combined_data = pd.concat(all_data, ignore_index=True)

plt.figure(figsize=(12, 6))
sns.violinplot(x="tissue", y="F1_score", data=combined_data)
plt.xticks(rotation=90)  
plt.title("F1 Score Distribution by Tissue")
plt.tight_layout()
plt.show()