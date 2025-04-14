import os
import pandas as pd
from collections import defaultdict

def process_csv_files(source_dir, target_dir, target_dir_with_aux):
  # Create target directories if they don't exist
  for dir_path in [target_dir, target_dir_with_aux]:
    if not os.path.exists(dir_path):
      os.makedirs(dir_path)

  # Dictionary to store DataFrames for each subdirectory
  subdir_data = defaultdict(list)
  subdir_data_with_aux = defaultdict(list)
  original_counts = defaultdict(int)

  # Walk through all subdirectories
  for root, dirs, files in os.walk(source_dir):
    csv_files = [f for f in files if f.endswith('.csv')]
    if not csv_files:
      continue

    # Get the relative path from source directory
    rel_path = os.path.relpath(root, source_dir)
    
    # Create corresponding target subdirectories
    target_subdir = os.path.join(target_dir, rel_path)
    target_subdir_with_aux = os.path.join(target_dir_with_aux, rel_path)
    for dir_path in [target_subdir, target_subdir_with_aux]:
      if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Process all CSV files in the current directory
    for file in csv_files:
      source_file = os.path.join(root, file)
      # Skip empty files
      if os.path.getsize(source_file) == 0:
        print(f"Skipping empty file: {source_file}")
        continue
      df = pd.read_csv(source_file)
      
      # Count original items
      original_counts[rel_path] += len(df)

      # First filter: fl_solution condition
      filtered_df = df[df['fl_solution'] != "\n\n * Formal Proof steps:\n"]
      
      # Second filter: nl_solution condition for auxiliary constructions
      aux_cons_df = filtered_df[
        filtered_df['nl_solution'].apply(lambda x: 
          'Auxiliary Constructions:' in str(x) and 
          not str(x).split('Auxiliary Constructions:')[1].strip().startswith(': Points')
        )
      ]
      
      # Append filtered data to the respective lists
      subdir_data[rel_path].append(filtered_df)
      subdir_data_with_aux[rel_path].append(aux_cons_df)

  # Process and save combined data for each subdirectory
  for rel_path in subdir_data:
    # Combine all DataFrames for this subdirectory
    combined_df = pd.concat(subdir_data[rel_path], ignore_index=True)
    combined_df_with_aux = pd.concat(subdir_data_with_aux[rel_path], ignore_index=True)

    # Save combined CSV files
    target_file = os.path.join(target_dir, rel_path, 'combined_filtered.csv')
    target_file_with_aux = os.path.join(target_dir_with_aux, rel_path, 'combined_filtered_with_aux.csv')
    
    combined_df.to_csv(target_file, index=False)
    combined_df_with_aux.to_csv(target_file_with_aux, index=False)

    # Create statistics files
    stats_file = os.path.join(target_dir, rel_path, 'statistics.txt')
    stats_file_with_aux = os.path.join(target_dir_with_aux, rel_path, 'statistics.txt')

    # Write statistics for filtered data
    with open(stats_file, 'w') as f:
      filtered_count = len(combined_df)
      original_count = original_counts[rel_path]
      percentage = (filtered_count / original_count * 100) if original_count > 0 else 0
      f.write(f"Total items: {filtered_count}\n")
      f.write(f"Percentage of original data: {percentage:.2f}%\n")

    # Write statistics for data with auxiliary constructions
    with open(stats_file_with_aux, 'w') as f:
      aux_count = len(combined_df_with_aux)
      percentage_of_original = (aux_count / original_count * 100) if original_count > 0 else 0
      percentage_of_filtered = (aux_count / filtered_count * 100) if filtered_count > 0 else 0
      f.write(f"Total items: {aux_count}\n")
      f.write(f"Percentage of original data: {percentage_of_original:.2f}%\n")
      f.write(f"Percentage of filtered data: {percentage_of_filtered:.2f}%\n")

if __name__ == "__main__":
  dataset_dir = "dataset"  # Source directory containing CSV files
  filtered_dataset_dir = "filtered_dataset"  # Target directory for filtered files
  filtered_dataset_with_aux_dir = "filtered_dataset_with_aux"  # Target directory for files with auxiliary constructions
  
  process_csv_files(dataset_dir, filtered_dataset_dir, filtered_dataset_with_aux_dir)
  print("Filtering completed. Results saved in 'filtered_dataset' and 'filtered_dataset_with_aux' directories.")