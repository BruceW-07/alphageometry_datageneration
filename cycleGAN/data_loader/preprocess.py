from pathlib import Path
import tqdm
import pandas as pd
import os

# Directory containing all the CSV files
directory = '/is/cluster/scratch/pghosh/dataset/alpha_geo/geometry/'
# directory = '/home/mmordig/formalization_proj/LLM_Formalizer/runs/datasets/arithmetic_small'

# List to hold all individual DataFrames
dataframes = []

# Iterate over each file in the directory
num_csvs = 0
for filename in tqdm.tqdm(Path(directory).iterdir()):
    if filename.endswith('.csv'):
        num_csvs += 1
        # if num_csvs > 2:
        #     break
        file_path = os.path.join(directory, filename)
        df = pd.read_csv(file_path)
        dataframes.append(df)

# Concatenate all DataFrames into one
combined_df = pd.concat(dataframes, ignore_index=True)

# Save the combined DataFrame to a new CSV file
destination = '/is/cluster/fast/scratch/pghosh/dataset/alpha_geo/geometry/nl_fl.csv'
# destination = 'runs/datasets/arithmetic/nl_fl.csv'
combined_df.to_csv(destination, index=False)
print(f'File written to {destination}')
