import numpy as np
import pandas as pd

def count_targets(graph_targets_fn):
    try:
        target_v = np.loadtxt(graph_targets_fn, dtype=str, delimiter=",")
        unique_targets, counts = np.unique(target_v, return_counts=True)
        target_counts = dict(zip(unique_targets, counts))
        return target_counts
    except FileNotFoundError:
        print(f"Error: Target file not found at {graph_targets_fn}")
        return {}
    except Exception as e:
        print(f"An error occurred while reading the target file: {e}")
        return {}

def main():
    output_dir3 = "/mnt/home/yuankeji/RanceLab/reticula_new/reticula/data/gtex/input/"
    num_trials = 10
    all_tissue_counts = {}

    for i in range(1, num_trials + 1):
        graph_targets_fn = f"{output_dir3}test_targets_{i}.txt"
        target_counts = count_targets(graph_targets_fn)
        print(f"Trial {i} Target Counts:")
        for target, count in target_counts.items():
            print(f"  {target}: {count}")
            if target not in all_tissue_counts:
                all_tissue_counts[target] = []
            all_tissue_counts[target].append(count)
        print("-" * 30)

    # Calculate the mean count for each tissue
    mean_tissue_counts = {}
    for tissue, counts in all_tissue_counts.items():
        mean_tissue_counts[tissue] = int(np.round(np.mean(counts)))

    # Create a Pandas DataFrame for the table
    df = pd.DataFrame(list(mean_tissue_counts.items()), columns=['Tissue', 'Mean Count'])
    print("\nMean Target Counts Across 10 Trials:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()