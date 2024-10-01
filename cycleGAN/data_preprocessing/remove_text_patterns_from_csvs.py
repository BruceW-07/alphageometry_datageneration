import pandas as pd
import csv
import re
from tqdm import tqdm

# Define the input and output CSV file names
input_filename = '/is/cluster/fast/scratch/pghosh/dataset/alpha_geo/geometry/svg_fl_long.csv'
output_filename = '/is/cluster/fast/scratch/pghosh/dataset/alpha_geo/geometry/output_file.csv'

# Adjust 'nl_statement' to the name of the column containing the XML data
xml_column = 'nl_statement'

# Read the CSV with appropriate parameters
df = pd.read_csv(
    input_filename,
    sep=',',
    quoting=csv.QUOTE_ALL,
    encoding='utf-8',
    dtype=str,
    keep_default_na=False,
    engine='python'
)

# Replace NaN with empty strings to avoid issues
df = df.fillna('')

# Define a function to remove 'opacity="0.8"' from a string
def remove_opacity(text):
    return re.sub(r'opacity="0\.8"', '', text)

# Enable tqdm for pandas
tqdm.pandas()

# Apply the function to the XML column with progress bar
df[xml_column] = df[xml_column].progress_apply(remove_opacity)

# Write the updated DataFrame back to a CSV file
df.to_csv(
    output_filename,
    index=False,
    quoting=csv.QUOTE_ALL,
    encoding='utf-8'
)
