#!/bin/bash
export PYTHONPATH=$PYTHONPATH:../
samples_per_run=10

# Check if the first argument is provided and is a numeric value
if [[ -z "$1" || ! "$1" =~ ^[0-9]+$ ]]; then
    echo "Error: Argument not provided or not a valid number."
    echo "Usage: $0 <numeric_argument>"
    exit 1
fi

# Use $(( )) for arithmetic calculations
seed_value=$(($1 * samples_per_run / 100))

# Base path for the Python environment and output directory
# python_env="/home/pghosh/miniconda3/envs/alpha_geo/bin/python"
# output_base="/is/cluster/scratch/pghosh/dataset/alpha_geo/arithmetic"
python_env="python"
output_base="dataset/alpha_geo/arithmetic"

# Run the Python script
expr_depth=1
for i in 5 25 45 65
do
  export NUM_FUNCS_TO_USE=$i
  out_dir="${output_base}/arithmetic_probs_depth_${expr_depth}_base_funcs_${NUM_FUNCS_TO_USE}"
  mkdir -p "$out_dir"  # Create the directory if it does not exist
  if ! $python_env demo.py \
  --out ${out_dir}/$1.csv \
  --n $samples_per_run \
  --seed $seed_value \
  --pbar False \
  --depth $expr_depth \
  --overwrite True; then
      echo "Python script failed at depth 1 with NUM_FUNCS_TO_USE $i"
      exit 1
  fi
done

expr_depth=2
export NUM_FUNCS_TO_USE=65
out_dir="${output_base}/arithmetic_probs_depth_${expr_depth}_base_funcs_${NUM_FUNCS_TO_USE}"
mkdir -p "$out_dir"  # Create the directory if it does not exist
if ! $python_env demo.py \
--out ${out_dir}/$1.csv \
--n $samples_per_run \
--seed $seed_value \
--pbar False \
--depth $expr_depth \
--overwrite True; then
    echo "Python script failed at depth 2"
    exit 1
fi
