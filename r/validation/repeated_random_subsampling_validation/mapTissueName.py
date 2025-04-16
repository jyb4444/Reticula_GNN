import os
import pandas as pd

tissue_mapping = {
    "0": "Adipose", "1": "Adrenal", "2": "Aorta", "3": "Bone", "4": "Brain",
    "5": "Eye", "6": "Heart", "7": "Intestine", "8": "Kidney", "9": "Liver",
    "10": "Lung", "11": "MOE", "12": "Mammary gland", "13": "Muscle", "14": "Nerve",
    "15": "Olfactory", "16": "Ovary", "17": "Pancreas", "18": "Skin", "19": "Sperm",
    "20": "Spinal cord", "21": "Stomach", "22": "Tendon", "23": "Testes", "24": "Thymus",
    "25": "Tongue", "26": "Uterus", "27": "VNO"
}

directory_path = "/Users/kejiyuan/Desktop/article/supplement_final/supplement/File_S3/stratified_kfold_validation"

# Get a list of all files in the directory
all_files = os.listdir(directory_path)

# Filter for files (not directories)
data_files = [f for f in all_files if os.path.isfile(os.path.join(directory_path, f))]

for filename in data_files:
    file_path = os.path.join(directory_path, filename)
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path, sep=',')  # Let pandas infer the header
        # Check if the 'tissue' column exists
        if 'tissue' in df.columns:
            # Replace the values in the 'tissue' column using the mapping
            df['tissue'] = df['tissue'].astype(str).map(tissue_mapping)

            # Save the modified DataFrame back to the same file, overwriting the original
            df.to_csv(file_path, sep='\t', index=False)
            print(f"Successfully processed and updated: {filename}")
        else:
            print(f"Warning: Column 'tissue' not found in {filename}. Skipping.")

    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
    except Exception as e:
        print(f"Error processing {filename}: {e}")

print("Finished processing files.")