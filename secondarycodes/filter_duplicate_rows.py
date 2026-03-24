import numpy as np
import os
import pandas as pd

def remove_duplicates(input_file: str):
    # Read the file (tab-separated)
    df = pd.read_csv(input_file, sep="\t")
    
    # Drop duplicates based on the first column (keep first occurrence)
    df = df.drop_duplicates(subset=df.columns[0], keep="first")
    
    # Overwrite the same file (without index column)
    df.to_csv(input_file, sep="\t", index=False)

# Example usage
if __name__ == "__main__":
    Pe_ = [1000]
    Gamma_ = [10**a for a in np.arange(-5., -3.99, 10.)]
    beta_ = [10**a for a in np.arange(-3.125, -5.001, -0.125)]
    for Pe in Pe_:
        for Gamma in Gamma_:
            for beta in beta_:
                filename = f"results/results/output_Pe_{Pe:.10g}_Gamma_{Gamma:.10g}_beta_{beta:.10g}/gamma_linear_plot.txt"  # Change this to your actual file path
                if os.path.isfile(filename):
                    remove_duplicates(filename)
                    print(f"Duplicates removed and file '{filename}' overwritten.")
