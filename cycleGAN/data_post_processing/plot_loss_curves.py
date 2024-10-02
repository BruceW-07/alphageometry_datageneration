import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
import math

def main():
    # Initialize data structures
    batch_indices = []

    # Define the loss columns
    loss_columns = ['avg_encoder_loss', 'avg_perplexity_loss', 'avg_decoder_loss', 'decoder_losses_from_natural']

    # Dictionaries to hold validation losses for normal and rephrased problems
    validation_losses_normal = {loss_type: [] for loss_type in loss_columns}
    validation_losses_rephrase = {loss_type: [] for loss_type in loss_columns}

    root_dir = ('/is/cluster/fast/pghosh/ouputs/alpha_geo/cycle_gan/geometry'
                '/meta-llama/Meta-Llama-3.1-8B_dec_only_11/validation_outputs/')
    csv_files = glob.glob(os.path.join(root_dir, '*_fl_fl.csv'))
    csv_files.sort()  # Ensure files are processed in order

    for csv_file in csv_files:
        # Extract batch index from filename (e.g., '0_24899_fl_fl.csv' -> 24899)
        base_name = os.path.basename(csv_file)
        batch_index = int(base_name.split('_')[1])

        # Read the CSV file into a DataFrame
        try:
            df = pd.read_csv(csv_file, sep=',', engine='python')
        except Exception as e:
            print(f"Error reading file {csv_file}: {e}")
            continue  # Skip this file if there is an error

        # Find the index of the separator row (where all columns are NaN)
        separator_idx = df[df.isnull().all(axis=1)].index.tolist()

        if not separator_idx:
            print(f"No separator line found in file {csv_file}, skipping.")
            continue  # Skip files without a separator line
        else:
            separator_idx = separator_idx[0]

        # Split the DataFrame into validation data for normal and rephrased problems
        validation_data_normal = df.loc[:separator_idx - 1].reset_index(drop=True)
        validation_data_rephrase = df.loc[separator_idx + 1:].reset_index(drop=True)

        # Convert loss columns to numeric, handling possible non-numeric entries
        for col in loss_columns:
            validation_data_normal[col] = pd.to_numeric(validation_data_normal[col], errors='coerce')
            validation_data_rephrase[col] = pd.to_numeric(validation_data_rephrase[col], errors='coerce')

        # Compute average losses and append to the lists
        for col in loss_columns:
            loss_normal = validation_data_normal[col].mean(skipna=True)
            loss_rephrase = validation_data_rephrase[col].mean(skipna=True)

            # Only append if the loss is a valid number
            validation_losses_normal[col].append(loss_normal if pd.notnull(loss_normal) else None)
            validation_losses_rephrase[col].append(loss_rephrase if pd.notnull(loss_rephrase) else None)

        # Append batch index
        batch_indices.append(batch_index)

    # Sort data by batch indices
    sorted_indices = sorted(range(len(batch_indices)), key=lambda k: batch_indices[k])
    batch_indices = [batch_indices[i] for i in sorted_indices]

    for col in loss_columns:
        validation_losses_normal[col] = [validation_losses_normal[col][i] for i in sorted_indices]
        validation_losses_rephrase[col] = [validation_losses_rephrase[col][i] for i in sorted_indices]

    # Plot the losses over batch indices in subplots
    num_losses = len(loss_columns)
    num_cols = 2
    num_rows = math.ceil(num_losses / num_cols)

    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, num_rows * 4))
    axs = axs.flatten()  # Flatten the array for easy indexing

    for idx, col in enumerate(loss_columns):
        ax = axs[idx]

        # Filter out None values for plotting
        valid_normal = pd.notnull(validation_losses_normal[col])
        valid_rephrase = pd.notnull(validation_losses_rephrase[col])

        # Plot only if there are valid data points
        if any(valid_normal):
            ax.plot([batch_indices[i] for i, valid in enumerate(valid_normal) if valid],
                    [validation_losses_normal[col][i] for i, valid in enumerate(valid_normal) if valid],
                    label='Validation Loss (Normal)')
        if any(valid_rephrase):
            ax.plot([batch_indices[i] for i, valid in enumerate(valid_rephrase) if valid],
                    [validation_losses_rephrase[col][i] for i, valid in enumerate(valid_rephrase) if valid],
                    label='Validation Loss (Rephrased)')

        ax.set_xlabel('Batch Index')
        ax.set_ylabel(col)
        ax.set_title(f'Validation {col} over Batches')
        ax.legend()
        ax.grid(True)

    # Hide any unused subplots
    for idx in range(len(loss_columns), len(axs)):
        fig.delaxes(axs[idx])

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
