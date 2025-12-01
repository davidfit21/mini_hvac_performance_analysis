import pandas as pd
import os

def load_fluid_files(fluid_name, txt_files, columns_to_keep):
    all_dfs = []
    for file_path in txt_files:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}. Skipping.")
            continue
        try:
            df = pd.read_csv(file_path, sep='\t', decimal=',', skiprows=1, header=None)
            df.columns = columns_to_keep
            all_dfs.append(df[columns_to_keep])
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        print(f"No valid data for {fluid_name}")
        return pd.DataFrame()
